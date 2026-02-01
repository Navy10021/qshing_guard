# src/eval/eval_fusion_operational.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..train.datasets import FusionDataset, QRDatasetConfig
from ..train.modeling_fusion import build_fusion_model
from ..train.modeling_qr import load_model_state

from .calibration import apply_temperature, fit_temperature
from .metrics import binary_report, save_report, threshold_at_fpr
from .plots import save_roc, save_pr, save_confusion, save_calibration_curve


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _collect_logits(model, loader, dev: torch.device):
    model.eval()
    ys, lgs = [], []
    for ximg, xurl, xlex, xctx, yb in loader:
        ximg = ximg.to(dev)
        xurl = xurl.to(dev)
        xlex = xlex.to(dev)
        xctx = xctx.to(dev)
        yb = yb.to(dev)
        logits = model(ximg, xurl, xlex, xctx)
        ys.append(yb.detach().cpu().numpy())
        lgs.append(logits.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(lgs)


@torch.no_grad()
def _collect_probs(model, loader, dev: torch.device):
    model.eval()
    ys, ps = [], []
    for ximg, xurl, xlex, xctx, yb in loader:
        ximg = ximg.to(dev)
        xurl = xurl.to(dev)
        xlex = xlex.to(dev)
        xctx = xctx.to(dev)
        yb = yb.to(dev)
        logits = model(ximg, xurl, xlex, xctx)
        prob = torch.softmax(logits, dim=1)[:, 1]
        ys.append(yb.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _vectorize_urls(vec, csv_path: str) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "url_norm" not in df.columns:
        raise ValueError(f"{csv_path} must include url_norm")
    X = vec.transform(df["url_norm"].astype(str).tolist()).astype(np.float32)
    return X.toarray()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="", help="optional(only to infer url_dim if needed)")
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)

    ap.add_argument("--model_dir", type=str, required=True, help="folder containing tfidf_vectorizer.joblib, best.pt")
    ap.add_argument("--ckpt", type=str, default="", help="default: {model_dir}/best.pt")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--fusion_mode", type=str, default="", choices=["", "concat", "gated"],
                    help="If empty, try to read from calibration.json else default gated")
    ap.add_argument("--calibration_json", type=str, default="", help="If provided, use it (T + thresholds).")
    ap.add_argument("--fit_temperature_on_val", action="store_true",
                    help="If set, fit temperature on val and compute thresholds on calibrated probs.")

    ap.add_argument("--warn_fpr", type=float, default=0.01)
    ap.add_argument("--block_fpr", type=float, default=0.001)

    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--augment_strength", type=str, default="default", choices=["light", "default", "strong"])
    ap.add_argument("--use_context", action="store_true", help="Use context feature branch (recommended).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = _device()
    print("Device:", dev)

    model_dir = Path(args.model_dir)
    ckpt_path = Path(args.ckpt) if args.ckpt else (model_dir / "best.pt")

    # vectorizer load
    vec_path = model_dir / "tfidf_vectorizer.joblib"
    if not vec_path.exists():
        raise FileNotFoundError(f"Missing: {vec_path}")
    vec = joblib.load(vec_path)

    # url vec
    Xva = _vectorize_urls(vec, args.val_csv)
    Xte = _vectorize_urls(vec, args.test_csv)

    # dataset/loader
    cfg_eval = QRDatasetConfig(image_size=args.image_size, augment=False, augment_strength=args.augment_strength, use_context=args.use_context)
    va_ds = FusionDataset(args.val_csv, url_vec=Xva, cfg=cfg_eval, seed=args.seed)
    te_ds = FusionDataset(args.test_csv, url_vec=Xte, cfg=cfg_eval, seed=args.seed)

    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # infer dims
    url_dim = Xva.shape[1]
    lex_dim = va_ds.lex.shape[1]
    ctx_dim = 7

    # fusion_mode resolve
    fusion_mode = args.fusion_mode.strip()
    calib_from_file = None
    if not fusion_mode:
        cand = model_dir / "calibration.json"
        if cand.exists():
            calib_from_file = json.loads(cand.read_text(encoding="utf-8"))
            fusion_mode = str(calib_from_file.get("fusion_mode", "gated"))
        else:
            fusion_mode = "gated"

    # build model + load weights
    model = build_fusion_model(fusion_mode=fusion_mode, url_dim=url_dim, lex_dim=lex_dim, ctx_dim=ctx_dim, num_classes=2).to(dev)
    load_model_state(model, str(ckpt_path), device=dev)

    # thresholds/calibration
    used_mode = "uncalibrated_thresholds_from_val"
    temperature = 1.0
    warn_thr, block_thr = 0.5, 0.5

    if args.calibration_json:
        calib = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
        temperature = float(calib.get("temperature", 1.0))
        th = calib.get("thresholds", {})
        warn_thr = float(th.get("warn", calib.get("warn_threshold", 0.5)))
        block_thr = float(th.get("block", calib.get("block_threshold", 0.5)))
        used_mode = "use_calibration_json"
    elif args.fit_temperature_on_val:
        yv, lv = _collect_logits(model, va_loader, dev)
        res = fit_temperature(lv, yv)
        temperature = float(res.temperature)
        pv = apply_temperature(lv, temperature)
        warn_thr = threshold_at_fpr(yv, pv, args.warn_fpr)
        block_thr = threshold_at_fpr(yv, pv, args.block_fpr)
        used_mode = "fit_temperature_on_val"
        (out_dir / "calibration_fitted.json").write_text(
            json.dumps(
                {
                    "temperature": temperature,
                    "nll_before": res.nll_before,
                    "nll_after": res.nll_after,
                    "warn_fpr": args.warn_fpr,
                    "block_fpr": args.block_fpr,
                    "thresholds": {"warn": warn_thr, "block": block_thr},
                    "fusion_mode": fusion_mode,
                    "use_context": bool(args.use_context),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        yv, pv = _collect_probs(model, va_loader, dev)
        warn_thr = threshold_at_fpr(yv, pv, args.warn_fpr)
        block_thr = threshold_at_fpr(yv, pv, args.block_fpr)

    # TEST inference
    if used_mode in ("use_calibration_json", "fit_temperature_on_val"):
        yt, lt = _collect_logits(model, te_loader, dev)
        pt = apply_temperature(lt, temperature)
    else:
        yt, pt = _collect_probs(model, te_loader, dev)

    # WARN report
    pred_warn = (pt >= warn_thr).astype(int)
    rep_warn = binary_report(yt, pred_warn, pt)
    save_report(str(out_dir / "report_test_warn.json"), rep_warn,
                extra={"mode": used_mode, "fusion_mode": fusion_mode, "warn_threshold": float(warn_thr), "warn_fpr": args.warn_fpr})

    # BLOCK report
    pred_block = (pt >= block_thr).astype(int)
    rep_block = binary_report(yt, pred_block, pt)
    save_report(str(out_dir / "report_test_block.json"), rep_block,
                extra={"mode": used_mode, "fusion_mode": fusion_mode, "block_threshold": float(block_thr), "block_fpr": args.block_fpr})

    (out_dir / "thresholds.json").write_text(
        json.dumps({"warn_threshold": float(warn_thr), "block_threshold": float(block_thr), "mode": used_mode, "fusion_mode": fusion_mode}, indent=2),
        encoding="utf-8",
    )

    # plots
    save_roc(yt, pt, str(out_dir / "roc.png"))
    save_pr(yt, pt, str(out_dir / "pr.png"))
    save_calibration_curve(yt, pt, str(out_dir / "calibration.png"), n_bins=15)

    try:
        from sklearn.metrics import confusion_matrix
        cm_warn = confusion_matrix(yt, pred_warn, labels=[0, 1])
        cm_block = confusion_matrix(yt, pred_block, labels=[0, 1])
    except Exception:
        cm_warn = np.zeros((2, 2), dtype=int)
        cm_block = np.zeros((2, 2), dtype=int)

    save_confusion(cm_warn, str(out_dir / "cm_warn.png"), title=f"Confusion (WARN thr={warn_thr:.4f})")
    save_confusion(cm_block, str(out_dir / "cm_block.png"), title=f"Confusion (BLOCK thr={block_thr:.4f})")

    print("[OK] Saved:", out_dir)
    print("  mode:", used_mode, "| fusion_mode:", fusion_mode)
    print("  warn_thr:", warn_thr, "block_thr:", block_thr)


if __name__ == "__main__":
    main()
