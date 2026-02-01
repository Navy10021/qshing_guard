#!/usr/bin/env python
"""Audit background library to ensure URL-agnostic images.

Strategy A requires backgrounds to be unrelated to URLs and ideally contain no QR codes at all.
This script scans images under assets/backgrounds and tries to decode any QR/barcode.

Usage:
  python scripts/audit_backgrounds_no_qr.py --background_dir assets/backgrounds --out_csv artifacts/reports/bg_qr_audit.csv
"""

import argparse
from pathlib import Path
from typing import List, Dict

import cv2
import pandas as pd
from pyzbar.pyzbar import decode

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def decode_any(img_bgr) -> List[str]:
    codes = decode(img_bgr)
    out = []
    for c in codes:
        try:
            out.append(c.data.decode("utf-8", errors="ignore"))
        except Exception:
            out.append("<binary>")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    bg_dir = Path(args.background_dir)
    paths = [p for p in bg_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    rows: List[Dict[str, object]] = []
    flagged = 0
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        decs = decode_any(img)
        has_code = len(decs) > 0
        if has_code:
            flagged += 1
        rows.append(
            {
                "path": str(p),
                "has_code": bool(has_code),
                "decoded": " | ".join(decs) if decs else "",
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Scanned: {len(rows)} backgrounds")
    print(f"Flagged (contains QR/barcode): {flagged}")
    print(f"Wrote audit report: {out_csv}")


if __name__ == "__main__":
    main()
