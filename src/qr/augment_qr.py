# qshing_guard/src/qr/augment_qr.py
from __future__ import annotations

"""Offline augmentation generator (REAL-WORLD focused).

Adds three layers:
A) Context-aware composition: QR + unrelated background + camera effects
B) Camera/print effects only (legacy realistic)
C) Context feature extraction: save per-aug image context features (Strategy C)

Example:
  python -m src.qr.augment_qr \
    --input_dir data/qr_images \
    --out_dir data/qr_images_aug \
    --n_per_image 2 \
    --strength strong \
    --background_dir assets/backgrounds \
    --context_mode mix \
    --output_size 512 \
    --save_meta_csv data/processed/qr_aug_meta.csv \
    --decode_filter \
    --seed 42
"""

import argparse
from pathlib import Path
import random
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from .augmentations import (
    AugmentConfig,
    ContextConfig,
    augment_qr_realistic,
    augment_qr_contextual,
    scan_backgrounds,
)

from .decode_qr import decode_qr_array

# tqdm optional
try:
    from tqdm.auto import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _load_bgr(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _resize_min(img: np.ndarray, min_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) >= min_side:
        return img
    s = float(min_side) / float(max(1, min(h, w)))
    nh, nw = int(round(h * s)), int(round(w * s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Folder with QR images (may contain subfolders)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_per_image", type=int, default=1, help="How many augmented copies to create per input")
    ap.add_argument("--seed", type=int, default=42)

    # strength controls
    ap.add_argument("--strength", type=str, default="default", choices=["light", "default", "strong"])

    # context-aware backgrounds
    ap.add_argument("--background_dir", type=str, default="", help="assets/backgrounds root (recursive scan)")
    ap.add_argument("--context_mode", type=str, default="mix", choices=["none", "mix", "always"])
    ap.add_argument("--context_prob", type=float, default=0.75, help="Used when context_mode=mix")

    # output control
    ap.add_argument("--output_size", type=int, default=0, help="If >0, force square output (e.g., 512)")
    ap.add_argument("--min_side", type=int, default=256, help="Ensure image min side >= this before processing")

    # decodability constraint
    ap.add_argument("--decode_filter", action="store_true")
    ap.add_argument("--max_tries", type=int, default=6, help="Max attempts per sample to pass decode_filter")

    # meta output
    ap.add_argument("--save_meta_csv", type=str, default="")

    # logging
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--max_skip_logs", type=int, default=20)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cam_cfg = AugmentConfig()
    if args.strength == "light":
        cam_cfg.p_perspective = 0.6
        cam_cfg.p_blur = 0.5
        cam_cfg.p_noise = 0.45
        cam_cfg.noise_sigma = (1.5, 10.0)
        cam_cfg.perspective_strength = (0.03, 0.08)
        cam_cfg.occlusion_frac = (0.04, 0.14)
        cam_cfg.jpeg_q = (35, 97)
    elif args.strength == "strong":
        cam_cfg.p_perspective = 0.95
        cam_cfg.p_blur = 0.85
        cam_cfg.p_noise = 0.8
        cam_cfg.noise_sigma = (4.0, 26.0)
        cam_cfg.perspective_strength = (0.06, 0.16)
        cam_cfg.occlusion_frac = (0.08, 0.28)
        cam_cfg.jpeg_q = (18, 85)

    ctx_cfg = ContextConfig()
    ctx_cfg.output_size = int(args.output_size) if args.output_size and args.output_size > 0 else 0
    if args.context_mode == "mix":
        ctx_cfg.p_context = float(args.context_prob)
    elif args.context_mode == "always":
        ctx_cfg.p_context = 1.0
    else:
        ctx_cfg.p_context = 0.0

    bg_paths: List[Path] = []
    if args.context_mode != "none":
        if not args.background_dir:
            raise SystemExit(
                "context_mode != none requires --background_dir. Create assets/backgrounds/{document,poster,receipt,screen}/ and add images."
            )
        bg_paths = scan_backgrounds(args.background_dir)
        if not bg_paths:
            raise SystemExit(f"No background images found under {args.background_dir}.")
        print(f"[BG] Found {len(bg_paths)} backgrounds under {args.background_dir}")

    inp = Path(args.input_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = [p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not paths:
        raise SystemExit(f"No images found under {inp}")

    it = paths
    if _HAS_TQDM:
        it = tqdm(paths, desc="Augment QR", unit="img")

    meta_rows: List[Dict[str, object]] = []
    written = 0
    skipped = 0

    for p in it:
        try:
            img = _load_bgr(p)
            if img is None:
                continue
            img = _resize_min(img, int(args.min_side))

            orig_dec = decode_qr_array(img) if args.decode_filter else None

            rel = p.relative_to(inp)
            stem = rel.stem
            parent = rel.parent
            (out / parent).mkdir(parents=True, exist_ok=True)

            for i in range(int(args.n_per_image)):
                ok = False
                last_err = None

                for attempt in range(max(1, int(args.max_tries))):
                    use_context = (args.context_mode != "none") and (random.random() < float(ctx_cfg.p_context))

                    if use_context:
                        bg_path = random.choice(bg_paths)
                        bg = _load_bgr(bg_path)
                        if bg is None:
                            last_err = f"bad background: {bg_path}"
                            continue
                        bg = _resize_min(bg, max(int(args.min_side), 256))
                        aug, feats, _bbox = augment_qr_contextual(img, bg, cam_cfg, ctx_cfg, return_features=True)
                        mode = "context"
                        bg_used = str(bg_path)
                    else:
                        aug = augment_qr_realistic(img, cam_cfg)
                        feats = None
                        mode = "camera_only"
                        bg_used = ""

                    if args.output_size and args.output_size > 0:
                        aug = cv2.resize(aug, (int(args.output_size), int(args.output_size)), interpolation=cv2.INTER_CUBIC)

                    if args.decode_filter:
                        dec = decode_qr_array(aug)
                        if not dec:
                            last_err = "decode_failed"
                            continue
                        if orig_dec is not None and orig_dec and (dec != orig_dec):
                            last_err = "decode_mismatch"
                            continue

                    op = out / parent / f"{stem}_aug{i}.jpg"
                    cv2.imwrite(str(op), aug)
                    written += 1

                    row: Dict[str, object] = {
                        "input_path": str(p),
                        "output_path": str(op),
                        "mode": mode,
                        "bg_path": bg_used,
                        "attempts": attempt + 1,
                    }
                    if feats:
                        row.update(feats)
                    meta_rows.append(row)

                    ok = True
                    break

                if not ok:
                    skipped += 1
                    if skipped <= int(args.max_skip_logs):
                        print(f"[WARN] skipped {p.name} aug{i}: {last_err}")
                    elif skipped == int(args.max_skip_logs) + 1:
                        print(f"[WARN] too many skips... suppressing further skip logs (max_skip_logs={args.max_skip_logs})")

            if _HAS_TQDM and hasattr(it, "set_postfix"):
                it.set_postfix(written=written, skipped=skipped)

        except Exception as e:
            skipped += 1
            if not args.continue_on_error:
                raise
            if skipped <= int(args.max_skip_logs):
                print(f"[WARN] error on {p}: {e}")
            elif skipped == int(args.max_skip_logs) + 1:
                print(f"[WARN] too many errors... suppressing further error logs (max_skip_logs={args.max_skip_logs})")

    print(f"Wrote augmented images to: {out}")
    print(f"Total augmented files: {written}")
    if skipped:
        print(f"Skipped (decode/error): {skipped}")

    if args.save_meta_csv:
        mpath = Path(args.save_meta_csv)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(meta_rows).to_csv(mpath, index=False)
        print(f"Wrote augmentation metadata: {mpath}")


if __name__ == "__main__":
    main()
