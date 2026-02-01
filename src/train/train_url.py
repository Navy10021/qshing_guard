from __future__ import annotations

"""Baseline #1: URL string model.

Char-level TF-IDF + LogisticRegression.

Fast, strong baseline when QR decoding is possible (or URLs are available).
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..eval.metrics import binary_report, save_report, threshold_at_fpr


def _load(path: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    if "url_norm" not in df.columns:
        raise ValueError("CSV must include url_norm")
    if "label" not in df.columns:
        raise ValueError("CSV must include label")
    X = df["url_norm"].astype(str).tolist()
    y = df["label"].astype(int).values
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts/models/url")
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--ngram_max", type=int, default=5)
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument(
        "--class_weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Use class_weight='balanced' to mitigate imbalance",
    )
    ap.add_argument("--warn_fpr", type=float, default=0.01, help="Target FPR for WARN threshold")
    ap.add_argument("--block_fpr", type=float, default=0.001, help="Target FPR for BLOCK threshold")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = _load(args.train_csv)
    Xva, yva = _load(args.val_csv)
    Xte, yte = _load(args.test_csv)

    clf = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, args.ngram_max),
                    min_df=2,
                    max_features=args.max_features,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    C=args.C,
                    max_iter=2000,
                    n_jobs=-1,
                    random_state=args.seed,
                    class_weight=None if args.class_weight == "none" else "balanced",
                ),
            ),
        ]
    )

    clf.fit(Xtr, ytr)

    # Choose operating thresholds on validation (FPR-capped)
    proba_val = clf.predict_proba(Xva)[:, 1]
    thr_warn = threshold_at_fpr(yva, proba_val, args.warn_fpr)
    thr_block = threshold_at_fpr(yva, proba_val, args.block_fpr)

    (out_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "warn_fpr": args.warn_fpr,
                "block_fpr": args.block_fpr,
                "thresholds": {"warn": float(thr_warn), "block": float(thr_block)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def _eval(name: str, X, y, thr: float):
        proba = clf.predict_proba(X)[:, 1]
        pred = (proba >= thr).astype(int)
        rep = binary_report(y, pred, proba)
        save_report(str(out_dir / f"report_{name}.json"), rep, extra={"threshold": float(thr)})
        print(name, rep)

    _eval("val", Xva, yva, thr_warn)
    _eval("test", Xte, yte, thr_warn)

    joblib.dump(clf, out_dir / "url_model.joblib")
    print("Saved:", out_dir / "url_model.joblib")


if __name__ == "__main__":
    main()
