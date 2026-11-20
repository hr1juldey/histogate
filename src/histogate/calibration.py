"""PR-6: temperature scaling (Guo et al. 2017) + energy-based OOD (Liu et al. 2020).

Oracles in tests/test_calib.py are analytic; nothing here is fit against the
dataset — calibration correctness is proven BEFORE any model ECE is believed
(TESTING §3 'calibration correctness' rule).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_ece(z: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-width confidence bins."""
    p = _softmax(np.asarray(z, dtype=np.float64))
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == np.asarray(y)).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.sum() == 0:
            continue
        ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def temperature_scale(z: np.ndarray, T: float) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) / float(T)


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter NLL minimization on held-out logits (bounded scalar search)."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y)
    n = len(y)

    def nll(T: float) -> float:
        p = _softmax(z / max(T, 1e-3))
        return -float(np.log(np.clip(p[np.arange(n), y], 1e-12, 1.0)).mean())

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def energy_score(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    """E(x) = -T * logsumexp(logits / T) — higher (less negative) = more OOD."""
    z = np.asarray(z, dtype=np.float64) / float(T)
    m = z.max(axis=1, keepdims=True)
    logsumexp = (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True))).squeeze(1)
    return -float(T) * logsumexp


def fit_energy_threshold(val_energy: np.ndarray, target_fpr: float = 0.05) -> float:
    """Threshold at the (1 - target_fpr) quantile of clean val energies."""
    e = np.asarray(val_energy, dtype=np.float64)
    return float(np.quantile(e, 1.0 - target_fpr))
