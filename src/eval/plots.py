from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: str) -> None:
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception:
        # fallback: no sklearn
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: str) -> None:
    try:
        from sklearn.metrics import precision_recall_curve
        p, r, _ = precision_recall_curve(y_true, y_prob)
    except Exception:
        p = np.array([1.0, 0.0])
        r = np.array([0.0, 1.0])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion(cm: np.ndarray, out_path: str, title: str = "Confusion Matrix") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, n_bins: int = 15) -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        xs.append(float(y_prob[m].mean()))
        ys.append(float(y_true[m].mean()))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title("Calibration Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_operating_points_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: str) -> None:
    """TPR vs FPR curve (same as ROC, kept for legacy naming)."""
    save_roc(y_true, y_prob, out_path)
