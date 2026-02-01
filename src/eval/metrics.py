from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        roc_curve,
        precision_recall_curve,
        confusion_matrix,
    )
    _HAS_SK = True
except Exception:
    _HAS_SK = False
    roc_auc_score = None
    average_precision_score = None
    roc_curve = None
    precision_recall_curve = None
    confusion_matrix = None


def threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """
    Return threshold such that FPR <= target_fpr (best-effort).
    Named to match the API expected by train_url.py.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if not _HAS_SK:
        # fallback: naive threshold
        return 0.5

    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return float(thr[-1])
    return float(thr[idx[-1]])


def find_threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Backward-compatible alias for threshold_at_fpr."""
    return threshold_at_fpr(y_true, y_score, target_fpr)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if not _HAS_SK:
        return 0.0

    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return float(tpr[0])
    return float(tpr[idx[-1]])


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (cnt / max(1, n)) * abs(acc - conf)
    return float(ece)


def summarize_binary(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, Any]:
    """
    Summarize confusion/precision/recall/f1/fpr/tpr at a threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(thr)).astype(int)

    tn = fp = fn = tp = 0
    if _HAS_SK:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    else:
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    fpr = fp / (fp + tn + eps)
    tpr = recall

    return {
        "threshold": float(thr),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": float(precision),
        "recall_tpr": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tpr": float(tpr),
    }


def global_scores(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Compute global scores such as ROC-AUC and PR-AUC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    out: Dict[str, Any] = {}
    if _HAS_SK:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


# ----------------------------
# train_url.py-compatible API.
# ----------------------------
def binary_report(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Provide the report shape expected by train_url.py.
    - y_pred: 0/1 predictions
    - y_prob: probabilities (optional, for auc/ece)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn = fp = fn = tp = 0
    if _HAS_SK:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    else:
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    fpr = fp / (fp + tn + eps)
    tpr = recall

    rep: Dict[str, Any] = {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": float(precision),
        "recall_tpr": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "n": int(len(y_true)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob).astype(float)
        rep.update(global_scores(y_true, y_prob))
        rep["ece"] = ece_score(y_true, y_prob, n_bins=15)

    return rep


def save_report(path: str, report: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a JSON report in the format used by train_url.py.
    """
    out = dict(report)
    if extra:
        out.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
