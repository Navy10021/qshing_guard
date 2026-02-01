# src/app/demo_fusion_predict.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

from ..train.datasets import FusionDataset, QRDatasetConfig
from ..train.modeling_fusion import build_fusion_model
from ..train.modeling_qr import load_model_state
from ..eval.calibration import apply_temperature


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_thresholds(path: str) -> tuple[float, float]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "thresholds" in d and isinstance(d["thresholds"], dict):
        return float(d["thresholds"].get("warn", 0.5)), float(d["thresholds"].get("block", 0.5))
    return float(d.get("warn_threshold", 0.5)), float(d.get("block_threshold", 0.5))


def _load_temperature(calib_json: str) -> float:
    d = json.loads(Path(calib_json).read_text(encoding="utf-8"))
    return float(d.get("temperature", 1.0))


def _decision(prob: float, warn_thr: float, block_thr: float) -> str:
    if prob >= block_thr:
        return "BLOCK"
    if prob >= warn_thr:
        return "WARN"
    return "ALLOW"


def _looks_like_ctx_and_y(b3, b4) -> bool:
    # (ximg,xurl,xlex,xctx,y): b3 float tensor (B,7) or (B,7?) or (7,) collated to (B,7)
    #                         b4 long tensor (B,)
    if torch.is_tensor(b3) and b3.dtype.is_floating_point and torch.is_tensor(b4) and (not b4.dtype.is_floating_point):
        return True
    return False


def _unpack_fusion_batch(batch):
    """
    Compatible with current FusionDataset in your repo:
      - (ximg, xurl, xlex, xctx, y)
      - (ximg, xurl, xlex, xctx, y, payload)
    Also supports older variants:
      - (ximg, xurl, xlex, y)
      - (ximg, xurl, xlex, y, payload)
    """
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    n = len(batch)
    if n == 4:
        ximg, xurl, xlex, y = batch
        xctx, payload = None, None
    elif n == 5:
        ximg, xurl, xlex, b3, b4 = batch
        if _looks_like_ctx_and_y(b3, b4):
            xctx, y, payload = b3, b4, None
        else:
            xctx, y, payload = None, b3, b4
    elif n >= 6:
        ximg, xurl, xlex, xctx, y, payload = batch[:6]
    else:
        raise ValueError(f"Unexpected batch length: {n}")
    return ximg, xurl, xlex, xctx, y, payload


@torch.no_grad()
def _predict_probs(model, loader, dev: torch.device, temperature: float | None, use_context: bool):
    model.eval()
    probs = []
    for batch in loader:
        ximg, xurl, xlex, xctx, _y, _payload = _unpack_fusion_batch(batch)

        ximg = ximg.to(dev, non_blocking=True)
        xurl = xurl.to(dev, non_blocking=True)
        xlex = xlex.to(dev, non_blocking=True)
        if use_context and xctx is not None:
            xctx = xctx.to(dev, non_blocking=True)

        try:
            logits = model(ximg, xurl, xlex, xctx) if (use_context and xctx is not None) else model(ximg, xurl, xlex)
        except TypeError:
            logits = model(ximg, xurl, xlex)

        if temperature is not None:
            p = apply_temperature(logits.detach().cpu().numpy(), float(temperature))
            probs.append(p)
        else:
            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)


def _save_per_item_viz(qr_path: str, url: str, prob: float, decision: str, out_path: Path):
    img_bgr = cv2.imread(qr_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    title = f"{Path(qr_path).name} | prob={prob:.4f} | {decision}\n{url}"
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _save_summary(decisions: list[str], probs: list[float], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    order = ["ALLOW", "WARN", "BLOCK"]
    counts = [decisions.count(k) for k in order]

    plt.figure()
    plt.bar(order, counts)
    plt.title("Decision Counts (Fusion)")
    plt.xlabel("Decision")
    plt.ylabel("Count")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(out_dir / "decision_counts.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(np.array(probs, dtype=float), bins=20)
    plt.title("Predicted prob(phish) Histogram (Fusion)")
    plt.xlabel("prob")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "prob_hist.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True, help="CSV with columns: qr_path,url_norm (label optional)")
    ap.add_argument("--model_dir", type=str, required=True, help="folder containing tfidf_vectorizer.joblib, best.pt")
    ap.add_argument("--ckpt", type=str, default="", help="default: {model_dir}/best.pt")
    ap.add_argument("--thresholds_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--fusion_mode", type=str, default="gated", choices=["concat", "gated"])
    ap.add_argument("--calibration_json", type=str, default="", help="optional temperature scaling json")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_context", action="store_true")
    ap.add_argument("--auto_context", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_item").mkdir(parents=True, exist_ok=True)

    dev = _device()
    warn_thr, block_thr = _load_thresholds(args.thresholds_json)

    model_dir = Path(args.model_dir)
    ckpt_path = Path(args.ckpt) if args.ckpt else (model_dir / "best.pt")

    vec_path = model_dir / "tfidf_vectorizer.joblib"
    if not vec_path.exists():
        raise FileNotFoundError(f"Missing: {vec_path}")
    vec = joblib.load(vec_path)

    df = pd.read_csv(args.input_csv)
    for c in ("qr_path", "url_norm"):
        if c not in df.columns:
            raise ValueError(f"input_csv must include {c}")

    X = vec.transform(df["url_norm"].astype(str).tolist()).astype(np.float32).toarray()

    cfg_eval = QRDatasetConfig(image_size=args.image_size, augment=False, augment_strength="default")

    ds = FusionDataset(args.input_csv, url_vec=X, cfg=cfg_eval, seed=args.seed, require_label=False)

    context_cols = {"qr_area_ratio", "qr_x", "qr_y", "blur_score", "contrast", "bg_complexity", "occlusion_ratio"}
    has_ctx_cols = any(c in df.columns for c in context_cols)
    use_context = bool(args.use_context or (args.auto_context and has_ctx_cols))

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    url_dim = X.shape[1]
    lex_dim = ds.lex.shape[1]

    try:
        model = build_fusion_model(args.fusion_mode, url_dim=url_dim, lex_dim=lex_dim, num_classes=2, use_context=use_context).to(dev)
    except TypeError:
        model = build_fusion_model(args.fusion_mode, url_dim=url_dim, lex_dim=lex_dim, num_classes=2).to(dev)

    load_model_state(model, str(ckpt_path), device=dev)

    temperature = None
    if args.calibration_json:
        temperature = _load_temperature(args.calibration_json)

    probs = _predict_probs(model, loader, dev, temperature, use_context=use_context)

    decisions = []
    rows = []
    for i, p in enumerate(probs):
        prob = float(p)
        dec = _decision(prob, warn_thr, block_thr)
        decisions.append(dec)
        rows.append({"qr_path": df.loc[i, "qr_path"], "url_norm": df.loc[i, "url_norm"], "prob": prob, "decision": dec})
        _save_per_item_viz(df.loc[i, "qr_path"], str(df.loc[i, "url_norm"]), prob, dec, out_dir / "per_item" / f"{i:03d}_viz.png")

    out_df = pd.DataFrame(rows).sort_values("prob", ascending=False)
    out_df.to_csv(out_dir / "predictions.csv", index=False)

    _save_summary(decisions, [float(x) for x in probs], out_dir)

    print("[OK] Saved:", out_dir / "predictions.csv")
    print("[OK] use_context:", use_context)


if __name__ == "__main__":
    main()
