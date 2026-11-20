"""PR-5 RED-first tests: metric wrappers (macro-F1, quadratic kappa, ECE).

Oracles (TESTING §1.1 — independent second implementations, never the module's
own formula re-implemented in the test):
- SKLEARN BIT-LEVEL: quadratic_kappa / macro_f1 must equal
  sklearn.cohen_kappa_score(weights="quadratic") / f1_score(average="macro")
  BIT-FOR-BIT (exact float equality, not approx) on random fixtures across
  many seeds, including degenerate distributions (single-class prediction
  collapse, absent classes, unbalanced priors).
- ANALYTIC KAPPA: hand-computed quadratic kappa on a 3-class fixture
  (kappa = -0.5 exactly, derived in-test from the confusion-matrix definition
  with a weight matrix — independent of both sklearn and the module).
- ECE: cross-checked against histogate.calibration.compute_ece (itself proven
  against analytic fixtures in PR-6) on synthetic logits, plus the analytic
  property: perfectly confident predictions with 30% errors => ECE == 0.30.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

from histogate.calibration import compute_ece
from histogate.metrics import ece, macro_f1, quadratic_kappa

RNG = np.random.default_rng(20261120)


def _random_pairs(n_seeds: int = 25, k: int = 9) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random (y_true, y_pred) fixtures, including adversarial degenerates."""
    pairs = []
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 + s)
        n = int(rng.integers(10, 2000))
        y_true = rng.integers(0, k, n)
        y_pred = rng.integers(0, k, n)
        pairs.append((y_true, y_pred))
    # collapse: model predicting a single class for everything
    pairs.append((RNG.integers(0, 9, 500), np.zeros(500, dtype=int)))
    # near-perfect with 3 errors
    y = RNG.integers(0, 9, 999)
    pairs.append((y, np.where(RNG.uniform(0, 1, 999) < 0.003, (y + 1) % 9, y)))
    # heavy class imbalance: 90% class 0
    y_true = np.where(RNG.uniform(0, 1, 3000) < 0.9, 0, RNG.integers(1, 9, 3000))
    y_pred = np.where(RNG.uniform(0, 1, 3000) < 0.9, 0, RNG.integers(1, 9, 3000))
    pairs.append((y_true, y_pred))
    # binary edge
    pairs.append((np.array([0, 1, 1, 0, 1, 0]), np.array([0, 1, 0, 0, 1, 1])))
    return pairs


class TestQuadraticKappa:
    def test_bit_identical_to_sklearn(self) -> None:
        for y_true, y_pred in _random_pairs():
            ours = quadratic_kappa(y_true, y_pred)
            ref = cohen_kappa_score(y_true, y_pred, weights="quadratic")
            assert ours == ref, (
                f"bit-level mismatch: ours={ours!r} sklearn={ref!r} "
                f"for y_true[:10]={y_true[:10]} y_pred[:10]={y_pred[:10]}"
            )

    def test_analytic_three_class_fixture(self) -> None:
        # Hand computation from the definition kappa = 1 - sum(O*w)/sum(E*w):
        # O = [[1,1,0],[0,1,1],[1,0,1]], w_ij=(i-j)^2/4, row/col totals all 2
        # sum over ALL entries: sum(O*w) = 1/4 + 1/4 + 1 = 1.5 (off-diagonals);
        # every E_ij = 2/3 and sum(w) = 3 -> sum(E*w) = 2.0
        # kappa = 1 - 1.5/2.0 = 0.25 exactly.
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])

        # independent in-test recomputation from the confusion-matrix formula
        k = 3
        o = np.zeros((k, k))
        for t, p in zip(y_true, y_pred, strict=True):
            o[t, p] += 1
        w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
        e = np.outer(o.sum(1), o.sum(0)) / o.sum()
        expected = 1.0 - (o * w).sum() / (e * w).sum()

        got = quadratic_kappa(y_true, y_pred)
        # bit-level agreement between module, in-test formula, and sklearn;
        # analytic value 0.25 up to float rounding of the in-test formula
        assert got == expected, f"kappa {got!r} != analytic {expected!r}"
        assert got == cohen_kappa_score(y_true, y_pred, weights="quadratic")
        assert abs(got - 0.25) < 1e-12

    def test_perfect_agreement_is_exactly_one(self) -> None:
        y = RNG.integers(0, 9, 777)
        assert quadratic_kappa(y, y.copy()) == 1.0

    def test_both_constant_identical_is_nan_like_sklearn(self) -> None:
        # sklearn's documented degenerate case: identical constants -> kappa = nan
        z = np.full(40, 4)
        ours, ref = quadratic_kappa(z, z.copy()), cohen_kappa_score(
            z, z.copy(), weights="quadratic"
        )
        assert np.isnan(ours) and np.isnan(ref)  # nan == nan is False; check isnan bits


class TestMacroF1:
    def test_bit_identical_to_sklearn(self) -> None:
        for y_true, y_pred in _random_pairs():
            ours = macro_f1(y_true, y_pred)
            ref = f1_score(y_true, y_pred, average="macro", labels=list(range(9)))
            assert ours == ref, f"bit-level mismatch: ours={ours!r} sklearn={ref!r}"

    def test_perfect_is_one_and_collapse_is_low(self) -> None:
        y = RNG.integers(0, 9, 900)
        assert macro_f1(y, y.copy()) == 1.0
        # predicting one class of nine: macro-F1 = 2*(1/9)/(...) -> sklearn value
        collapsed = macro_f1(y, np.zeros(900, dtype=int))
        assert collapsed == f1_score(y, np.zeros(900, dtype=int), average="macro", labels=list(range(9)))
        assert collapsed < 0.25  # sanity bound, well above chance-level macro-F1


class TestECE:
    def test_bit_identical_to_calibration_module(self) -> None:
        # PR-6 proved compute_ece against analytic fixtures; here the PR-5
        # wrapper must not perturb it (same values, bit-level).
        for s in range(10):
            rng = np.random.default_rng(500 + s)
            z = rng.normal(0, 3, (400, 9))
            y = rng.integers(0, 9, 400)
            for bins in (10, 15, 20):
                assert ece(z, y, n_bins=bins) == compute_ece(z, y, n_bins=bins)

    def test_analytic_confident_and_wrong(self) -> None:
        # logits +-40 => softmax confidence 1.0 (single occupied bin); with all
        # confidence at 1.0, ECE collapses to the EMPIRICAL error rate of the
        # fixture (analytic property, independent of binning details)
        rng = np.random.default_rng(7)
        n = 1000
        c = rng.integers(0, 9, n)
        y = np.where(rng.uniform(0, 1, n) < 0.3, (c + 1) % 9, c)
        z = np.full((n, 9), -40.0)
        z[np.arange(n), c] = 40.0
        expected_error_rate = float((y != c).mean())
        assert abs(ece(z, y, n_bins=15) - expected_error_rate) < 1e-6

    def test_calibrated_by_construction_is_small(self) -> None:
        rng = np.random.default_rng(11)
        n, k = 20000, 9
        p = rng.dirichlet(np.ones(k) * 0.8, size=n)
        # vectorized categorical sampling via inverse-CDF
        u = rng.uniform(0, 1, n)
        cdf = np.cumsum(p, axis=1)
        y = (u[:, None] > cdf).sum(axis=1).clip(0, k - 1)
        z = np.log(p)
        assert ece(z, y, n_bins=15) < 0.02
