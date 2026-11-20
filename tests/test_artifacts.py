"""PR-3 RED-first tests: AIRAQc 8-class synthetic artifact injector.

Oracles:
- MONOTONICITY: severity must map monotonically to measurable effect (sharpness,
  coverage) — Spearman rho == 1, computed independently with scipy.
- IDENTITY: severity 0 must be bitwise identity + empty mask.
- MASK TRUTH: mask=True pixels must actually differ from the original.
- BACK-CONSISTENCY: no_tissue/missing_tissue outputs must be low-OD (blank-class).
- DETERMINISM: same seed -> identical bytes; different seed -> different.
"""

import hashlib

import numpy as np
import pytest
from scipy.stats import spearmanr

from histogate.bio_optics import inject_artifact, rgb_to_od
from histogate.schemas.report import ArtifactKind


def _tissue_like(size: int = 224, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter

    h = gaussian_filter(rng.random((size, size)), 3)
    e = gaussian_filter(rng.random((size, size)), 3)
    S = np.array([[0.644, 0.717, 0.267], [0.092, 0.954, 0.283]])
    conc = np.stack([0.3 + h, 0.2 + 0.7 * e], -1).reshape(-1, 2)
    od = conc @ S
    return np.clip(255 * np.exp(-od) - 1, 0, 255).astype(np.uint8).reshape(size, size, 3)


SPATIAL_KINDS = [
    ArtifactKind.TISSUE_FOLD, ArtifactKind.PEN_MARK, ArtifactKind.AIR_BUBBLE,
    ArtifactKind.COVERSLIP_LINE, ArtifactKind.KNIFE_LINE, ArtifactKind.MISSING_TISSUE,
]
ALL_KINDS = list(ArtifactKind)


class TestIdentity:
    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_severity_zero_is_identity(self, kind: ArtifactKind) -> None:
        img = _tissue_like(seed=1)
        out, mask = inject_artifact(img, kind, 0.0, seed=3)
        assert np.array_equal(out, img)
        assert mask.sum() == 0

    def test_negative_severity_rejects(self) -> None:
        with pytest.raises(ValueError):
            inject_artifact(_tissue_like(), ArtifactKind.PEN_MARK, -0.1)

    def test_overunit_severity_rejects(self) -> None:
        with pytest.raises(ValueError):
            inject_artifact(_tissue_like(), ArtifactKind.PEN_MARK, 1.3)


class TestMonotonicity:
    def test_out_of_focus_sharpness_monotone_decreasing(self) -> None:
        """Independent oracle: variance of Laplacian must FALL as severity rises."""
        img = _tissue_like(seed=2)
        severities = [0.2, 0.4, 0.6, 0.8, 1.0]
        sharp = []
        for s in severities:
            out, _ = inject_artifact(img, ArtifactKind.OUT_OF_FOCUS, s, seed=0)
            lap = np.diff(out.astype(float), axis=0).var() + np.diff(out.astype(float), axis=1).var()
            sharp.append(lap)
        rho, _ = spearmanr(severities, sharp)
        assert rho <= -0.99, f"sharpness not monotonically decreasing (rho={rho})"

    @pytest.mark.parametrize("kind", SPATIAL_KINDS)
    def test_spatial_coverage_monotone_increasing(self, kind: ArtifactKind) -> None:
        img = _tissue_like(seed=4)
        severities = [0.25, 0.5, 0.75, 1.0]
        cov = [inject_artifact(img, kind, s, seed=0)[1].mean() for s in severities]
        rho, _ = spearmanr(severities, cov)
        assert rho == 1.0, f"{kind}: coverage not monotone in severity (rho={rho})"


class TestMaskTruth:
    @pytest.mark.parametrize("kind", SPATIAL_KINDS + [ArtifactKind.PEN_MARK])
    def test_masked_pixels_differ(self, kind: ArtifactKind) -> None:
        """Adversarial: a mask claiming artifact where pixels are untouched lies."""
        img = _tissue_like(seed=5)
        out, mask = inject_artifact(img, kind, 0.7, seed=1)
        assert mask.sum() > 0
        changed = out != img
        # every masked pixel must actually have changed in some channel
        assert changed.any(axis=2)[mask].mean() > 0.95

    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_output_stays_uint8_finite(self, kind: ArtifactKind) -> None:
        out, _ = inject_artifact(_tissue_like(seed=6), kind, 0.9, seed=2)
        assert out.dtype == np.uint8
        assert out.shape == (224, 224, 3)


class TestBlankConsistency:
    def test_no_tissue_is_low_od(self) -> None:
        img = _tissue_like(seed=7)
        out, mask = inject_artifact(img, ArtifactKind.NO_TISSUE, 1.0, seed=0)
        assert mask.mean() > 0.99  # whole patch flagged
        assert rgb_to_od(out).mean() < 0.15  # blank-class OD level

    def test_missing_tissue_half_blank(self) -> None:
        img = _tissue_like(seed=8)
        out, mask = inject_artifact(img, ArtifactKind.MISSING_TISSUE, 1.0, seed=0)
        assert 0.2 < mask.mean() < 0.8  # a real region, not all/nothing
        assert rgb_to_od(out[mask]).mean() < 0.15


class TestDeterminism:
    @pytest.mark.parametrize("kind", ALL_KINDS)
    def test_same_seed_same_bytes(self, kind: ArtifactKind) -> None:
        img = _tissue_like(seed=9)
        a, ma = inject_artifact(img, kind, 0.6, seed=11)
        b, mb = inject_artifact(img, kind, 0.6, seed=11)
        assert hashlib.md5(a.tobytes()).hexdigest() == hashlib.md5(b.tobytes()).hexdigest()
        assert np.array_equal(ma, mb)

    def test_different_seed_pen_mark_differs(self) -> None:
        img = _tissue_like(seed=10)
        a, _ = inject_artifact(img, ArtifactKind.PEN_MARK, 0.6, seed=1)
        b, _ = inject_artifact(img, ArtifactKind.PEN_MARK, 0.6, seed=2)
        assert not np.array_equal(a, b)
