from __future__ import annotations

"""Lightweight probability calibration utilities.

We use temperature scaling (Guo et al.) which is simple, fast, and
works well for deep models whose probabilities are miscalibrated.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TemperatureScalingResult:
    temperature: float
    nll_before: float
    nll_after: float


def _nll_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss()(logits, y)


@torch.no_grad()
def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature and return calibrated probabilities (class 1)."""
    t = float(max(1e-6, temperature))
    lg = torch.tensor(logits, dtype=torch.float32)
    prob = torch.softmax(lg / t, dim=1)[:, 1].cpu().numpy()
    return prob


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 50) -> TemperatureScalingResult:
    """Fit temperature on a validation set by minimizing NLL."""
    lg = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(y_true.astype(int), dtype=torch.long)

    # single scalar parameter T > 0
    t = torch.ones(1, requires_grad=True)

    optimizer = torch.optim.LBFGS([t], lr=0.1, max_iter=max_iter)

    nll_before = float(_nll_from_logits(lg, y).item())

    def closure():
        optimizer.zero_grad()
        # enforce positivity
        temp = torch.clamp(t, min=1e-6)
        loss = _nll_from_logits(lg / temp, y)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        temp = float(torch.clamp(t, min=1e-6).item())
        nll_after = float(_nll_from_logits(lg / temp, y).item())

    return TemperatureScalingResult(temperature=temp, nll_before=nll_before, nll_after=nll_after)
