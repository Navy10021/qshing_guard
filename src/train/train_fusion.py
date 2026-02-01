from __future__ import annotations

"""Baseline #3: Fusion model (with context feature branch).

Inputs:
 - QR image (CNN backbone)
 - URL TF-IDF (dense, low-dim)
 - URL lexical features (small numeric vector)
 - QR context features (7-dim, derived from the image via QR detector)

This baseline is robust: URL dominates when informative, image helps when URL is noisy,
and context helps to model "real-world placement/quality" (poster/document/screen, blur, etc.).
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from ..eval.calibration import apply_temperature, fit_temperature
from ..eval.metrics import binary_report, save_report, threshold_at_fpr
from .datasets import FusionDataset, QRDatasetConfig
from .modeling_fusion import build_fusion_model

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("qr_path", "url_norm", "label"):
        if col not in df.columns:
            raise ValueError(f"CSV must include {col}")
    return df


@torch.no_grad()
def _eval(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for ximg, xurl, xlex, xctx, yb in loader:
        ximg = ximg.to(device)
        xurl = xurl.to(device)
        xlex = xlex.to(device)
        xctx = xctx.to(device)
        yb = yb.to(device)
        logits = model(ximg, xurl, xlex, xctx)
        prob = torch.softmax(logits, dim=1)[:, 1]
        ys.append(yb.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def _collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, lgs = [], []
    for ximg, xurl, xlex, xctx, yb in loader:
        ximg = ximg.to(device)
        xurl = xurl.to(device)
        xlex = xlex.to(device)
        xctx = xctx.to(device)
        yb = yb.to(device)
        logits = model(ximg, xurl, xlex, xctx)
        ys.append(yb.detach().cpu().numpy())
        lgs.append(logits.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(lgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts/models/fusion")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--augment_strength", type=str, default="default", choices=["light", "default", "strong"])
    ap.add_argument("--use_context", action="store_true", help="Use context feature branch (recommended).")
    ap.add_argument("--tfidf_max_features", type=int, default=6000)
    ap.add_argument("--tfidf_ngram_max", type=int, default=5)
    ap.add_argument(
        "--fusion_mode",
        type=str,
        default="gated",
        choices=["concat", "gated"],
        help="How to fuse modalities: concat vs gated fusion (recommended)",
    )
    ap.add_argument("--balance_sampler", action="store_true", help="Use WeightedRandomSampler to mitigate imbalance")
    ap.add_argument("--calibrate", action="store_true", help="Fit temperature scaling on val and save calibration.json")
    ap.add_argument("--warn_fpr", type=float, default=0.01)
    ap.add_argument("--block_fpr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = _device()
    print("Device:", dev)

    df_tr = _load_df(args.train_csv)
    df_va = _load_df(args.val_csv)
    df_te = _load_df(args.test_csv)

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, args.tfidf_ngram_max),
        min_df=2,
        max_features=args.tfidf_max_features,
    )
    Xtr = vec.fit_transform(df_tr["url_norm"].astype(str).tolist()).astype(np.float32)
    Xva = vec.transform(df_va["url_norm"].astype(str).tolist()).astype(np.float32)
    Xte = vec.transform(df_te["url_norm"].astype(str).tolist()).astype(np.float32)

    # densify (manageable when max_features is small)
    Xtr_d = Xtr.toarray()
    Xva_d = Xva.toarray()
    Xte_d = Xte.toarray()

    joblib.dump(vec, out_dir / "tfidf_vectorizer.joblib")

    cfg_train = QRDatasetConfig(image_size=args.image_size, augment=True, augment_strength=args.augment_strength, use_context=args.use_context)
    cfg_eval = QRDatasetConfig(image_size=args.image_size, augment=False, augment_strength=args.augment_strength, use_context=args.use_context)

    tr_ds = FusionDataset(args.train_csv, url_vec=Xtr_d, cfg=cfg_train, seed=args.seed)
    va_ds = FusionDataset(args.val_csv, url_vec=Xva_d, cfg=cfg_eval, seed=args.seed)
    te_ds = FusionDataset(args.test_csv, url_vec=Xte_d, cfg=cfg_eval, seed=args.seed)

    if args.balance_sampler:
        labels = tr_ds.labels
        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())
        w0 = 1.0 / max(1, n0)
        w1 = 1.0 / max(1, n1)
        weights = torch.tensor([w1 if int(y) == 1 else w0 for y in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=2, pin_memory=True)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    url_dim = Xtr_d.shape[1]
    lex_dim = tr_ds.lex.shape[1]
    ctx_dim = 7

    model = build_fusion_model(fusion_mode=args.fusion_mode, url_dim=url_dim, lex_dim=lex_dim, ctx_dim=ctx_dim, num_classes=2).to(dev)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        it = tr_loader
        if _HAS_TQDM:
            it = tqdm(tr_loader, desc=f"Train ep{ep}", unit="batch")

        running = 0.0
        n_batches = 0
        for ximg, xurl, xlex, xctx, yb in it:
            ximg = ximg.to(dev, non_blocking=True)
            xurl = xurl.to(dev, non_blocking=True)
            xlex = xlex.to(dev, non_blocking=True)
            xctx = xctx.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(ximg, xurl, xlex, xctx)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            running += float(loss.item())
            n_batches += 1
            if _HAS_TQDM and hasattr(it, "set_postfix"):
                it.set_postfix(loss=running / max(1, n_batches))

        yv, pv = _eval(model, va_loader, dev)
        pred = (pv >= 0.5).astype(int)
        rep = binary_report(yv, pred, pv)
        save_report(str(out_dir / f"report_val_ep{ep}.json"), rep, extra={"epoch": ep})
        print(f"[VAL ep{ep}]", rep)

        rep_f1 = float(rep.get("f1", 0.0))
        if rep_f1 > best_f1:
            best_f1 = rep_f1
            torch.save({"model": model.state_dict(), "epoch": ep, "fusion_mode": args.fusion_mode, "use_context": bool(args.use_context)}, out_dir / "best.pt")

    ckpt = torch.load(out_dir / "best.pt", map_location=dev)
    model.load_state_dict(ckpt["model"])

    # calibration / thresholds
    temperature = 1.0
    thresholds = {"warn": 0.5, "block": 0.5}
    if args.calibrate:
        yv, lv = _collect_logits(model, va_loader, dev)
        res = fit_temperature(lv, yv)
        temperature = float(res.temperature)
        pv_cal = apply_temperature(lv, temperature)
        thresholds["warn"] = threshold_at_fpr(yv, pv_cal, args.warn_fpr)
        thresholds["block"] = threshold_at_fpr(yv, pv_cal, args.block_fpr)
        (out_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "temperature": temperature,
                    "nll_before": res.nll_before,
                    "nll_after": res.nll_after,
                    "warn_fpr": args.warn_fpr,
                    "block_fpr": args.block_fpr,
                    "thresholds": thresholds,
                    "fusion_mode": args.fusion_mode,
                    "use_context": bool(args.use_context),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # test
    if args.calibrate:
        yt, lt = _collect_logits(model, te_loader, dev)
        pt = apply_temperature(lt, temperature)
        pred = (pt >= thresholds["warn"]).astype(int)
    else:
        yt, pt = _eval(model, te_loader, dev)
        pred = (pt >= 0.5).astype(int)

    rep = binary_report(yt, pred, pt)
    save_report(str(out_dir / "report_test.json"), rep, extra={"best_epoch": int(ckpt.get("epoch", -1)), "fusion_mode": args.fusion_mode, "use_context": bool(args.use_context)})
    print("[TEST]", rep)

    torch.save({"model": model.state_dict(), "fusion_mode": args.fusion_mode, "use_context": bool(args.use_context)}, out_dir / "last.pt")
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
