from __future__ import annotations

"""Baseline #2: QR image model (with context feature branch).

ResNet18-like classifier trained on QR images with realistic augmentation.

Input CSV must include: qr_path, label.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from ..eval.calibration import apply_temperature, fit_temperature
from ..eval.metrics import binary_report, save_report, threshold_at_fpr
from .datasets import QRDatasetConfig, QRImageDataset
from .modeling_qr import build_qr_model

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, logits) for calibration."""
    model.eval()
    ys, lgs = [], []
    for batch in loader:
        if len(batch) == 3:
            xb, xctx, yb = batch
        else:
            xb, yb = batch
            xctx = None
        xb = xb.to(device)
        yb = yb.to(device)
        if xctx is not None:
            xctx = xctx.to(device)
            logits = model(xb, xctx)
        else:
            logits = model(xb)
        ys.append(yb.detach().cpu().numpy())
        lgs.append(logits.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(lgs)


@torch.no_grad()
def _eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        if len(batch) == 3:
            xb, xctx, yb = batch
        else:
            xb, yb = batch
            xctx = None
        xb = xb.to(device)
        yb = yb.to(device)
        if xctx is not None:
            xctx = xctx.to(device)
            logits = model(xb, xctx)
        else:
            logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1]
        ys.append(yb.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts/models/qr")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--augment_strength", type=str, default="default", choices=["light", "default", "strong"])
    ap.add_argument(
        "--use_context",
        action="store_true",
        help="Use QR context feature branch (recommended). If unset, fall back to image-only model.",
    )
    ap.add_argument(
        "--balance_sampler",
        action="store_true",
        help="Use WeightedRandomSampler to mitigate class imbalance",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit temperature scaling on the validation set and save calibration.json",
    )
    ap.add_argument(
        "--warn_fpr",
        type=float,
        default=0.01,
        help="Target FPR for WARN threshold (e.g., 0.01=1%%)",
    )
    ap.add_argument(
        "--block_fpr",
        type=float,
        default=0.001,
        help="Target FPR for BLOCK threshold (e.g., 0.001=0.1%%)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = _device()
    print("Device:", dev)

    cfg_train = QRDatasetConfig(image_size=args.image_size, augment=True, augment_strength=args.augment_strength, use_context=args.use_context)
    cfg_eval = QRDatasetConfig(image_size=args.image_size, augment=False, augment_strength=args.augment_strength, use_context=args.use_context)

    train_ds = QRImageDataset(args.train_csv, cfg=cfg_train, seed=args.seed)
    val_ds = QRImageDataset(args.val_csv, cfg=cfg_eval, seed=args.seed)
    test_ds = QRImageDataset(args.test_csv, cfg=cfg_eval, seed=args.seed)

    if args.balance_sampler:
        # weights inversely proportional to class frequency
        labels = train_ds.labels
        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())
        w0 = 1.0 / max(1, n0)
        w1 = 1.0 / max(1, n1)
        weights = torch.tensor([w1 if int(y) == 1 else w0 for y in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_qr_model(num_classes=2, use_context=args.use_context).to(dev)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        it = train_loader
        if _HAS_TQDM:
            it = tqdm(train_loader, desc=f"Train ep{ep}", unit="batch")

        running = 0.0
        n_batches = 0
        for batch in it:
            if len(batch) == 3:
                xb, xctx, yb = batch
            else:
                xb, yb = batch
                xctx = None

            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            if xctx is not None:
                xctx = xctx.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb, xctx) if xctx is not None else model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            running += float(loss.item())
            n_batches += 1
            if _HAS_TQDM and hasattr(it, "set_postfix"):
                it.set_postfix(loss=running / max(1, n_batches))

        # val
        yv, pv = _eval_epoch(model, val_loader, dev)
        pred = (pv >= 0.5).astype(int)
        rep = binary_report(yv, pred, pv)
        save_report(str(out_dir / f"report_val_ep{ep}.json"), rep, extra={"epoch": ep})
        print(f"[VAL ep{ep}]", rep)

        rep_f1 = float(rep.get("f1", 0.0))
        if rep_f1 > best_f1:
            best_f1 = rep_f1
            torch.save({"model": model.state_dict(), "epoch": ep, "use_context": bool(args.use_context)}, out_dir / "best.pt")

    # Load best checkpoint before calibration/testing
    ckpt = torch.load(out_dir / "best.pt", map_location=dev)
    model.load_state_dict(ckpt["model"])

    # (optional) temperature scaling on val
    temperature = 1.0
    thresholds = {"warn": 0.5, "block": 0.5}
    if args.calibrate:
        yv, lv = _collect_logits(model, val_loader, dev)
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
                    "use_context": bool(args.use_context),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # test best
    if args.calibrate:
        yt, lt = _collect_logits(model, test_loader, dev)
        pt = apply_temperature(lt, temperature)
        pred = (pt >= thresholds["warn"]).astype(int)
    else:
        yt, pt = _eval_epoch(model, test_loader, dev)
        pred = (pt >= 0.5).astype(int)

    rep = binary_report(yt, pred, pt)
    save_report(str(out_dir / "report_test.json"), rep, extra={"best_epoch": int(ckpt.get("epoch", -1)), "use_context": bool(args.use_context)})
    print("[TEST]", rep)

    torch.save({"model": model.state_dict(), "use_context": bool(args.use_context)}, out_dir / "last.pt")
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
