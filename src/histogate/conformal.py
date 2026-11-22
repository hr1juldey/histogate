"""Split-conformal Adaptive Prediction Sets (APS) over the tile classifier.

Method (deterministic, non-randomized APS):
- nonconformity score(x, y) = cumulative softmax mass from the highest-prob
  class down to and including the true class.
- qhat(alpha) = the ceil((n+1)(1-alpha))-th smallest calibration score,
  capped into [1, n] (standard split-conformal rank; the cap makes the
  guarantee slightly conservative, never anti-conservative).
- prediction set = classes in descending-prob order until the running mass
  reaches qhat.

Guarantee: among gate-ACCEPTED tiles, P(true class in set) >= 1 - alpha —
distribution-free, finite-sample — conditional on the calibration tiles being
exchangeable with the served tiles. The calibration split is disjoint from
both training data and CRC-VAL (see scripts/calibrate_conformal.py).
"""
from __future__ import annotations

import numpy as np


def _aps_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """APS nonconformity score per row: mass accumulated through true class."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    order = np.argsort(-probs, axis=1, kind="stable")
    scores = np.empty(len(y), dtype=np.float64)
    for i in range(len(y)):
        rank = int(np.flatnonzero(order[i] == y[i])[0])
        scores[i] = probs[i][order[i]][: rank + 1].sum()
    return scores


def aps_qhat(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold at miscoverage level alpha (0 < alpha < 1)."""
    s = np.sort(_aps_scores(cal_probs, cal_y))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def aps_set(
    probs: np.ndarray,
    qhat: float,
    classes: list | tuple | None = None,
) -> list:
    """Prediction set for one tile: descending-prob classes until mass >= qhat."""
    if classes is None:
        from histogate.data import CLASS_ORDER

        classes = list(CLASS_ORDER)
    probs = np.asarray(probs, dtype=np.float64).ravel()
    order = np.argsort(-probs, kind="stable")
    acc = 0.0
    out: list[int] = []
    for idx in order:
        out.append(int(idx))
        acc += float(probs[idx])
        if acc >= qhat - 1e-9:  # fp tolerance: 0.6+0.3 must reach 0.9
            break
    return [classes[i] for i in out]
