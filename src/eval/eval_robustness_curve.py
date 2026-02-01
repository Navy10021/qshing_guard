# src/eval/eval_robustness_curve.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ..models.context_gan import ContextAttacker
from ..models.context_encoder import ContextEncoder
from ..models.detector import DetectorConfig, QuishingDetector
from ..train.datasets import QRDatasetConfig, QRImageDataset, FusionDataset
from ..features.url_vectorizer import fit_url_vectorizer
from ..utils.backgrounds import BackgroundSampler
from ..utils.context_ops import context_attack
from ..utils.qr_decode import decode_qr_array


def _auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def _softmax_prob_phish(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


def _load_thresholds(path: str) -> tuple[float, float]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "thresholds" in d and isinstance(d["thresholds"], dict):
        return float(d["thresholds"].get("warn", 0.5)), float(d["thresholds"].get("block", 0.5))
    return float(d.get("warn_threshold", 0.5)), float(d.get("block_threshold", 0.5))


def _decision(prob: float, warn_thr: float, block_thr: float) -> int:
    if prob >= block_thr:
        return 2
    if prob >= warn_thr:
        return 1
    return 0


def _tensor_to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
    t = x.detach().float().cpu()
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(f"Expected CHW with 3 channels, got shape={tuple(t.shape)}")
    vmin = float(t.min())
    vmax = float(t.max())
    if vmin < -0.1 and vmax <= 1.1:
        t = (t + 1.0) / 2.0
    t = t.clamp(0.0, 1.0)
    rgb = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return rgb[:, :, ::-1].copy()


def _looks_like_ctx_and_y(b1, b2) -> bool:
    # ctx: float tensor (B,7) or (7,) collated; y: long/int tensor (B,)
    return (
        torch.is_tensor(b1) and b1.dtype.is_floating_point and
        torch.is_tensor(b2) and (not b2.dtype.is_floating_point)
    )


def _unpack_qr_batch(batch, use_context: bool):
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    n = len(batch)
    if n == 2:
        x, y = batch
        ctx, payload = None, None
    elif n == 3:
        x, b1, b2 = batch
        if _looks_like_ctx_and_y(b1, b2):
            ctx, y, payload = b1, b2, None
        else:
            ctx, y, payload = None, b1, b2
    else:
        x, ctx, y, payload = batch[:4]

    if not use_context:
        ctx = None
    return x, ctx, y, payload


def _unpack_fusion_batch(batch, use_context: bool):
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
    else:
        ximg, xurl, xlex, xctx, y, payload = batch[:6]

    if not use_context:
        xctx = None
    return ximg, xurl, xlex, xctx, y, payload


def _confusion_from_probs(probs: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pred = (probs >= thr).astype(np.int32)
    y = y.astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / max(1, (tp + fn))
    fpr = fp / max(1, (fp + tn))
    precision = tp / max(1, (tp + fp))
    return {"tpr": float(tpr), "fpr": float(fpr), "precision": float(precision)}


def _plot_curves(df: pd.DataFrame, out_dir: Path, warn_thr: float | None, block_thr: float | None):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["strength"].values, df["accuracy"].values, marker="o")
    plt.title("Robustness Curve: Accuracy vs Attack Strength")
    plt.xlabel("attack strength")
    plt.ylabel("accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "robustness_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()

    if {"warn_rate", "block_rate", "allow_rate"}.issubset(df.columns):
        plt.figure()
        plt.plot(df["strength"].values, df["allow_rate"].values, marker="o", label="ALLOW rate")
        plt.plot(df["strength"].values, df["warn_rate"].values, marker="o", label="WARN rate")
        plt.plot(df["strength"].values, df["block_rate"].values, marker="o", label="BLOCK rate")
        title = "Operational Decision Rates vs Attack Strength"
        if warn_thr is not None and block_thr is not None:
            title += f" (warn={warn_thr:.3f}, block={block_thr:.3f})"
        plt.title(title)
        plt.xlabel("attack strength")
        plt.ylabel("rate")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(out_dir / "robustness_decision_rates.png", dpi=200, bbox_inches="tight")
        plt.close()

    if {"warn_tpr", "warn_fpr", "block_tpr", "block_fpr"}.issubset(df.columns):
        plt.figure()
        plt.plot(df["strength"].values, df["warn_tpr"].values, marker="o", label="WARN TPR")
        plt.plot(df["strength"].values, df["warn_fpr"].values, marker="o", label="WARN FPR")
        plt.plot(df["strength"].values, df["block_tpr"].values, marker="o", label="BLOCK TPR")
        plt.plot(df["strength"].values, df["block_fpr"].values, marker="o", label="BLOCK FPR")
        plt.title("Operational TPR/FPR vs Attack Strength")
        plt.xlabel("attack strength")
        plt.ylabel("rate")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(out_dir / "robustness_tpr_fpr.png", dpi=200, bbox_inches="tight")
        plt.close()

    if "decode_rate" in df.columns:
        plt.figure()
        plt.plot(df["strength"].values, df["decode_rate"].values, marker="o")
        plt.title("QR Decode Success Rate vs Attack Strength")
        plt.xlabel("attack strength")
        plt.ylabel("decode success rate")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / "robustness_decode_rate.png", dpi=200, bbox_inches="tight")
        plt.close()

    if "payload_match_rate" in df.columns:
        plt.figure()
        plt.plot(df["strength"].values, df["payload_match_rate"].values, marker="o")
        plt.title("Payload Match Rate vs Attack Strength")
        plt.xlabel("attack strength")
        plt.ylabel("payload match rate")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / "robustness_payload_match_rate.png", dpi=200, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--detector_mode", choices=["qr", "fusion"], default="qr")
    ap.add_argument("--fusion_mode", choices=["gated", "concat"], default="gated")
    ap.add_argument("--ckpt_detector", required=True)
    ap.add_argument("--ckpt_attacker", required=True)
    ap.add_argument("--use_context", action="store_true")

    ap.add_argument("--background_dir", default="assets/backgrounds")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_features", type=int, default=8000)
    ap.add_argument("--strength_grid", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--thresholds_json", type=str, default="")
    ap.add_argument("--decode_check", action="store_true")
    ap.add_argument("--payload_match", action="store_true")
    ap.add_argument("--payload_col", type=str, default="url_norm")
    ap.add_argument("--decode_subset", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    warn_thr = block_thr = None
    if args.thresholds_json:
        warn_thr, block_thr = _load_thresholds(args.thresholds_json)

    # IMPORTANT: enable payload return when payload_match is requested.
    cfg = QRDatasetConfig(
        image_size=args.image_size,
        augment=False,
        use_context=bool(args.use_context),
        return_payload=bool(args.payload_match),
        payload_col=str(args.payload_col),
    )

    if args.detector_mode == "qr":
        ds = QRImageDataset(args.test_csv, cfg=cfg, seed=args.seed, require_label=True)
        det_cfg = DetectorConfig(mode="qr", use_context=bool(args.use_context))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        fusion = False
    else:
        df0 = pd.read_csv(args.test_csv)
        bundle = fit_url_vectorizer(df0["url_norm"].astype(str).tolist(), max_features=args.max_features)
        ds = FusionDataset(args.test_csv, url_vec=bundle.X, cfg=cfg, seed=args.seed, require_label=True)
        det_cfg = DetectorConfig(
            mode="fusion",
            fusion_mode=args.fusion_mode,
            url_dim=bundle.X.shape[1],
            lex_dim=ds.lex.shape[1],
            use_context=bool(args.use_context),
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        fusion = True

    detector = QuishingDetector(det_cfg).to(device)
    ck = torch.load(args.ckpt_detector, map_location=device)
    detector.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    detector.eval()

    attacker = ContextAttacker(style_dim=16).to(device)
    ckA = torch.load(args.ckpt_attacker, map_location=device)
    attacker.load_state_dict(ckA["model"] if isinstance(ckA, dict) and "model" in ckA else ckA)
    attacker.eval()

    ctx_enc = ContextEncoder().to(device)
    bg = BackgroundSampler(args.background_dir)

    strengths = np.array([float(s) for s in args.strength_grid.split(",")], dtype=np.float32)
    rows = []

    with torch.no_grad():
        for s in strengths:
            correct = 0
            n = 0
            probs_all = []
            y_all = []

            allow_n = warn_n = block_n = 0

            decode_ok = 0
            decode_n = 0
            payload_match_ok = 0
            payload_match_n = 0

            for batch in loader:
                if not fusion:
                    x, ctx_clean, y, payload = _unpack_qr_batch(batch, use_context=bool(args.use_context))
                    url_vec = lex = None
                    if ctx_clean is not None:
                        ctx_clean = ctx_clean.to(device)
                else:
                    x, url_vec, lex, ctx_clean, y, payload = _unpack_fusion_batch(batch, use_context=bool(args.use_context))
                    url_vec = url_vec.to(device)
                    lex = lex.to(device)
                    if ctx_clean is not None:
                        ctx_clean = ctx_clean.to(device)

                x = x.to(device)
                y = y.to(device)

                style = torch.rand((x.shape[0], 16), device=device)
                p = attacker(x, style)
                bg_t = bg.sample(x.shape[0], size=x.shape[-1], device=device) if bg.available() else None
                x_adv = context_attack(x, bg_t, p, difficulty=float(s))

                ctx_adv = ctx_enc(x_adv) if args.use_context else ctx_clean

                logits = detector(x_adv, url_vec=url_vec, lex=lex, ctx=ctx_adv)
                pred = logits.argmax(dim=1)

                correct += (pred == y).sum().item()
                n += x.shape[0]

                prob = _softmax_prob_phish(logits).detach().cpu().numpy().astype(np.float32)
                probs_all.append(prob)
                y_all.append(y.detach().cpu().numpy().astype(np.int32))

                if warn_thr is not None and block_thr is not None:
                    for pr in prob:
                        d = _decision(float(pr), warn_thr, block_thr)
                        if d == 0:
                            allow_n += 1
                        elif d == 1:
                            warn_n += 1
                        else:
                            block_n += 1

                if args.decode_check or args.payload_match:
                    b = x_adv.shape[0]
                    idxs = list(range(b))
                    if args.decode_subset and args.decode_subset > 0:
                        random.shuffle(idxs)
                        idxs = idxs[: min(b, args.decode_subset)]

                    for i in idxs:
                        try:
                            bgr = _tensor_to_bgr_uint8(x_adv[i])
                            dec = decode_qr_array(bgr)
                        except Exception:
                            dec = None

                        decode_n += 1
                        if dec is not None and len(str(dec)) > 0:
                            decode_ok += 1

                        if args.payload_match:
                            gt = None
                            if payload is not None:
                                try:
                                    gt = payload[i] if isinstance(payload, (list, tuple)) else payload
                                except Exception:
                                    gt = None
                            if gt is None:
                                continue  # don't count
                            payload_match_n += 1
                            if dec is not None and str(dec) == str(gt):
                                payload_match_ok += 1

            acc = correct / max(1, n)
            probs_np = np.concatenate(probs_all) if len(probs_all) > 0 else np.zeros((0,), dtype=np.float32)
            y_np = np.concatenate(y_all) if len(y_all) > 0 else np.zeros((0,), dtype=np.int32)

            row = {"strength": float(s), "accuracy": float(acc), "n": int(n)}

            if warn_thr is not None and block_thr is not None and n > 0:
                row.update(
                    {"allow_rate": float(allow_n / n), "warn_rate": float(warn_n / n), "block_rate": float(block_n / n)}
                )
                warn_cf = _confusion_from_probs(probs_np, y_np, warn_thr)
                block_cf = _confusion_from_probs(probs_np, y_np, block_thr)
                row.update(
                    {
                        "warn_tpr": warn_cf["tpr"],
                        "warn_fpr": warn_cf["fpr"],
                        "warn_precision": warn_cf["precision"],
                        "block_tpr": block_cf["tpr"],
                        "block_fpr": block_cf["fpr"],
                        "block_precision": block_cf["precision"],
                    }
                )

            if (args.decode_check or args.payload_match) and decode_n > 0:
                row["decode_rate"] = float(decode_ok / decode_n)

            if args.payload_match and payload_match_n > 0:
                row["payload_match_rate"] = float(payload_match_ok / payload_match_n)

            rows.append(row)
            msg = f"[strength={s:.2f}] acc={acc:.4f}"
            if "decode_rate" in row:
                msg += f" decode={row['decode_rate']:.3f}"
            if "payload_match_rate" in row:
                msg += f" payload_match={row['payload_match_rate']:.3f}"
            print(msg)

    df = pd.DataFrame(rows)
    out_csv = out_dir / "robustness_curve.csv"
    df.to_csv(out_csv, index=False)

    auc = _auc_trapz(df["strength"].to_numpy(np.float32), df["accuracy"].to_numpy(np.float32))

    meta = {
        "auc_accuracy": float(auc),
        "strength_grid": [float(v) for v in strengths.tolist()],
        "detector_mode": args.detector_mode,
        "fusion_mode": args.fusion_mode,
        "use_context": bool(args.use_context),
        "thresholds_json": args.thresholds_json if args.thresholds_json else None,
        "warn_threshold": float(warn_thr) if warn_thr is not None else None,
        "block_threshold": float(block_thr) if block_thr is not None else None,
        "decode_check": bool(args.decode_check),
        "payload_match": bool(args.payload_match),
        "payload_col": str(args.payload_col),
        "decode_subset": int(args.decode_subset),
        "ckpt_detector": args.ckpt_detector,
        "ckpt_attacker": args.ckpt_attacker,
    }
    (out_dir / "robustness_auc.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_curves(df, out_dir, warn_thr, block_thr)

    print("Saved:", out_csv)
    print("AUC:", auc)
    print("Plots:", out_dir / "robustness_accuracy.png")


if __name__ == "__main__":
    main()
