"""PR-6 RED-first tests: temperature scaling + energy OOD.

Oracles:
- HAND-COMPUTED ECE: identical-confidence fixture, ECE = |conf - acc| exactly.
- CALIBRATED-BY-CONSTRUCTION: logits = log(true posterior), labels sampled from
  the posterior -> ECE ~ 0.
- TEMPERATURE RECOVERY: logits = T_inject * log(p), y ~ p -> fitted T ~ T_inject
  (makes softmax(z/T) the true posterior = NLL optimum).
- ENERGY THRESHOLD: quantile policy on known distributions hits target FPR.
"""

import numpy as np

from histogate.calibration import (
    compute_ece,
    fit_energy_threshold,
    fit_temperature,
    temperature_scale,
)


def _posterior_logits(n: int = 20000, k: int = 9, seed: int = 0,
                      conf: float = 0.9, t_inject: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """z = t_inject * log(p); y sampled from p. Correct with prob `conf`."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, k, n)
    correct = rng.uniform(0, 1, n) < conf
    y = np.where(correct, c, (c + rng.integers(1, k, n)) % k)
    p = np.full((n, k), (1.0 - conf) / (k - 1))
    p[np.arange(n), c] = conf
    z = t_inject * np.log(p)
    return z, y


class TestECE:
    def test_hand_computed_fixture(self) -> None:
        # 4 samples, identical confidence 1/(1+e^-5); preds correct 3 of 4
        z = np.array([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
        y = np.array([0, 0, 1, 0])  # last one wrong
        c = 1.0 / (1.0 + np.exp(-5.0))
        expected = abs(c - 0.75)  # single occupied bin: ECE = |conf - acc|
        got = compute_ece(z, y, n_bins=10)
        assert abs(got - expected) < 1e-3

    def test_calibrated_by_construction_is_small(self) -> None:
        z, y = _posterior_logits(conf=0.9)
        ece = compute_ece(z, y, n_bins=20)
        assert ece < 0.02, f"calibrated-by-construction logits show ECE {ece:.3f}"


class TestTemperature:
    def test_recovers_injected_scale(self) -> None:
        z, y = _posterior_logits(conf=0.9, t_inject=4.0)
        T = fit_temperature(z, y)
        assert 3.0 < T < 5.0, f"fitted T={T}, expected ~4.0"

    def test_already_calibrated_returns_one(self) -> None:
        z, y = _posterior_logits(conf=0.9, t_inject=1.0)
        T = fit_temperature(z, y)
        assert 0.9 < T < 1.1, f"calibrated logits got T={T}"

    def test_scaling_restores_true_confidence(self) -> None:
        """z = 4*log(p) is overconfident (~1.0); dividing by T=4 restores ~0.9."""
        z, _ = _posterior_logits(conf=0.9, t_inject=4.0)
        zs = temperature_scale(z, 4.0)
        e = lambda q: np.exp(q - q.max(axis=1, keepdims=True))  # noqa: E731
        conf_before = (e(z) / e(z).sum(1, keepdims=True)).max()
        conf_after = (e(zs) / e(zs).sum(1, keepdims=True)).max()
        assert conf_before > 0.9999
        assert abs(conf_after - 0.9) < 0.02, f"after scaling conf={conf_after}"


class TestEnergyThreshold:
    def test_quantile_policy_hits_target_fpr(self) -> None:
        val_energy = np.random.default_rng(1).normal(-5.0, 1.0, size=100000)
        thr = fit_energy_threshold(val_energy, target_fpr=0.05)
        fpr = float((val_energy > thr).mean())
        assert abs(fpr - 0.05) < 0.002

    def test_threshold_separates_shifted(self) -> None:
        val = np.random.default_rng(2).normal(-5.0, 1.0, 50000)
        ood = np.random.default_rng(3).normal(-2.0, 1.0, 50000)
        thr = fit_energy_threshold(val, target_fpr=0.05)
        tpr = float((ood > thr).mean())
        assert tpr > 0.9
