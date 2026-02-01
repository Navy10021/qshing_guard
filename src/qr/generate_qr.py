from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import qrcode

from ..data.utils import url_hash

# tqdm optional (progress visualization)
try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False


def make_qr(url: str, ecc: str = 'H', box_size: int = 10, border: int = 4):
    ecc_map = {
        'L': qrcode.constants.ERROR_CORRECT_L,
        'M': qrcode.constants.ERROR_CORRECT_M,
        'Q': qrcode.constants.ERROR_CORRECT_Q,
        'H': qrcode.constants.ERROR_CORRECT_H,
    }
    qr = qrcode.QRCode(
        version=None,
        error_correction=ecc_map.get(ecc.upper(), qrcode.constants.ERROR_CORRECT_H),
        box_size=box_size,
        border=border,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white')
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest_path', type=str, required=True, help='CSV with at least url_norm and label')
    ap.add_argument('--out_dir', type=str, required=True, help='Root folder for QR images')
    ap.add_argument('--update_manifest_out', type=str, required=True, help='Output CSV with qr_path added')
    ap.add_argument('--ecc', type=str, default='H', choices=['L','M','Q','H'])
    ap.add_argument('--box_size', type=int, default=10)
    ap.add_argument('--border', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='If >0, only generate first N rows')
    args = ap.parse_args()

    df = pd.read_csv(args.manifest_path)
    if 'url_norm' not in df.columns:
        raise ValueError('manifest must include url_norm')
    if 'label' not in df.columns:
        raise ValueError('manifest must include label')

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = df if args.limit <= 0 else df.iloc[: args.limit].copy()
    qr_paths = []

    iterator = rows.iterrows()
    if _HAS_TQDM:
        iterator = tqdm(iterator, total=len(rows), desc='Generate QR', unit='qr')

    for _, r in iterator:
        url = str(r['url_norm'])
        label = int(r['label'])
        subdir = out_root / str(label)
        subdir.mkdir(parents=True, exist_ok=True)
        fname = f"{url_hash(url)}.png"
        fpath = subdir / fname
        if not fpath.exists():
            img = make_qr(url, ecc=args.ecc, box_size=args.box_size, border=args.border)
            img.save(fpath)
        qr_paths.append(str(fpath))

        if _HAS_TQDM and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(saved=len(qr_paths))

    # write updated manifest (align back to original df)
    if args.limit <= 0:
        df['qr_path'] = qr_paths
        df.to_csv(args.update_manifest_out, index=False)
    else:
        rows['qr_path'] = qr_paths
        rows.to_csv(args.update_manifest_out, index=False)

    print(f"Saved QR images under: {out_root}")
    print(f"Wrote updated manifest: {args.update_manifest_out}")


if __name__ == '__main__':
    main()
