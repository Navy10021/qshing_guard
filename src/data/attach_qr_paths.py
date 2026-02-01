from __future__ import annotations

"""Attach qr_path to split CSVs.

You have manifest_with_qr.csv (full dataset).
This script merges qr_path into train/val/test splits using url_norm as key.

"""

import argparse
from pathlib import Path

import pandas as pd


def _attach(split_csv: Path, map_df: pd.DataFrame, out_path: Path) -> None:
    df = pd.read_csv(split_csv)
    if "url_norm" not in df.columns:
        raise ValueError(f"split CSV must include url_norm: {split_csv}")

    merged = df.merge(map_df[["url_norm", "qr_path"]], on="url_norm", how="left", validate="m:1")
    # if some rows missing (rare), keep but warn
    missing = int(merged["qr_path"].isna().sum())
    if missing:
        print(f"[WARN] {split_csv.name}: qr_path missing for {missing} rows")
    merged.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_with_qr", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True, help="Directory containing train.csv/val.csv/test.csv")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    m = pd.read_csv(args.manifest_with_qr)
    if "url_norm" not in m.columns or "qr_path" not in m.columns:
        raise ValueError("manifest_with_qr must include url_norm and qr_path")
    # de-dup just in case
    m = m.drop_duplicates(subset=["url_norm"], keep="first")

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train", "val", "test"):
        p = splits_dir / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        out_p = out_dir / f"{name}_with_qr.csv"
        _attach(p, m, out_p)
        print("Wrote:", out_p)


if __name__ == "__main__":
    main()
