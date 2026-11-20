"""PR-5: metrics for the 9-class patch task (macro-F1, quadratic kappa, ECE).

Thin wrappers: F1/kappa delegate to sklearn verbatim (TESTING §3 names them
the reference implementations; bit-level equality is tested in
tests/test_metrics.py against sklearn on random fixtures). ECE delegates to
the PR-6 calibration module, which was proven against analytic fixtures
before any model ECE was believed.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from histogate.calibration import compute_ece
from histogate.data import CLASS_ORDER

N_CLASSES = len(CLASS_ORDER)


def _labels() -> list[int]:
    return list(range(N_CLASSES))


def macro_f1(y_true, y_pred) -> float:
    """Macro-averaged F1 over all 9 classes (absent classes count as 0 F1)."""
    return float(
        f1_score(np.asarray(y_true), np.asarray(y_pred), average="macro", labels=_labels())
    )


def quadratic_kappa(y_true, y_pred) -> float:
    """Quadratically weighted Cohen's kappa (sklearn reference implementation)."""
    return float(
        cohen_kappa_score(np.asarray(y_true), np.asarray(y_pred), weights="quadratic")
    )


def ece(logits, y_true, n_bins: int = 15) -> float:
    """Expected Calibration Error (equal-width bins) on raw logits."""
    return compute_ece(np.asarray(logits, dtype=np.float64), np.asarray(y_true), n_bins=n_bins)
