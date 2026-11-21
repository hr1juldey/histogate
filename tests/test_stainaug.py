"""P16-S1 RED: adversarial tests for histogate.stainaug (module NOT built yet).

These tests attack from outside with real data:
- T1 leave-one-scanner-out: augmentation calibrated on 5 of 6 stain-space
  clusters (Macenko-stat clustering of REAL NONORM images = scanner proxy;
  filenames carry no scanner labels) must cover the held-out cluster's
  measured stain stats. < 80% coverage = ranges overfit to calibration data.
- T2 blind discriminator: a logistic classifier must NOT cleanly separate
  augmented-NCT from real-NONORM (AUC in [0.5, 0.65]). Easy separation =
  physically implausible augmentation. AUC < 0.5 = leakage, also fails.
- T3 error-preservation band: the real ft2 model's flip rate under
  augmentation must be in [0.5%, 15%] — 0% = no-op, > 15% = signal destroyed.
- T4 physics directionality: deterministic c_hematoxylin x1.5 perturbation
  must move measured H concentration by ~1.5x and leave eosin ~unchanged.
- T5 identity: zero jitter returns bitwise-identical uint8.

Skipped (with reason) where artifacts/datasets are absent (CI runners).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO = Path(__file__).resolve().parents[1]
NONORM = REPO / "data" / "raw" / "NCT-CRC-HE-100K-NONORM"
NCT = REPO / "data" / "raw" / "NCT-CRC-HE-100K"
CKPT = REPO / "runs" / "nonorm-ft2" / "best.ckpt.pt"

needs_data = pytest.mark.skipif(
    not NONORM.exists() or not NCT.exists(), reason="datasets not on this machine")


def _paths(root: Path, n: int) -> list[Path]:
    out: list[Path] = []
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        tifs = sorted(cls.glob("*.tif"))
        take = tifs[: max(1, n // 9)]
        out += take
        if len(out) >= n:
            break
    return out[:n]


def _img(p: Path) -> np.ndarray:
    a = tifffile.imread(p)
    return a[..., :3] if a.ndim == 3 else np.stack([a] * 3, -1)


def _od(img: np.ndarray) -> np.ndarray:
    return -np.log10((img.astype(np.float64) + 1.0) / 256.0).reshape(-1, 3)


@needs_data
def test_t1_leave_one_scanner_out():
    from sklearn.cluster import KMeans

    from histogate.bio_optics import macenko_stain_matrix
    from histogate.stainaug import StainAugmenter, calibrate_from_scanners

    paths = _paths(NONORM, 240)
    feats, valid = [], []
    for p in paths:  # bounded: 240 Macenko estimates
        w = macenko_stain_matrix(_od(_img(p)))
        if np.isfinite(w).all():
            feats.append(w.flatten())
            valid.append(p)
    feats = np.array(feats)
    km = KMeans(n_clusters=6, n_init=4, random_state=0).fit(feats)
    held = int(np.bincount(km.labels_).argmin())  # rarest cluster = held out
    calib_paths = [p for p, c in zip(valid, km.labels_, strict=True) if c != held]
    test_paths = [p for p, c in zip(valid, km.labels_, strict=True) if c == held]

    params = calibrate_from_scanners(calib_paths)
    aug = StainAugmenter(params, seed=0)

    def stat(ps: list[Path]) -> np.ndarray:
        return np.array([macenko_stain_matrix(_od(_img(p))).flatten()
                         for p in ps])

    calib_stats = stat(calib_paths)
    lo, hi = calib_stats.min(0), calib_stats.max(0)
    held_real = stat(test_paths)
    auged = np.array([macenko_stain_matrix(
        _od(aug(_img(p)))).flatten() for p in test_paths])
    # per-dimension coverage (a 6-D min/max box fails on ANY dim excursion;
    # Macenko re-estimation noise makes all-dims-inside impossible)
    m = np.maximum(hi - lo, 1e-3) * 0.5
    reach = float(((auged >= lo - m) & (auged <= hi + m)).mean())
    q05 = np.quantile(auged, 0.05, axis=0)
    q95 = np.quantile(auged, 0.95, axis=0)
    touch = float(((held_real >= q05 - m) & (held_real <= q95 + m)).mean())
    assert reach >= 0.8, (
        f"augmented stain stats exceed real calibration range on >20% of "
        f"per-dim samples ({reach:.2f}): augmentation implausible")
    assert touch >= 0.8, (
        f"augmentation fails to reach held-out scanner-cluster stain stats "
        f"({touch:.2f} < 0.80): ranges overfit to 5 clusters")
    assert held_real.shape[0] >= 5


@needs_data
def test_t2_blind_discriminator():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    from histogate.stainaug import StainAugmenter, calibrate_from_scanners

    calib = calibrate_from_scanners(_paths(NONORM, 200))
    aug = StainAugmenter(calib, seed=1)

    def feat(img: np.ndarray) -> np.ndarray:
        od = _od(img)
        od = od[od.sum(1) > 0.3]
        if len(od) == 0:                       # fully washed image guard
            return np.zeros(6)
        return np.concatenate([od.mean(0), od.std(0)])

    x_base = _paths(NCT, 500)                 # PAIRED design: same base
    x_real = np.array([feat(_img(p)) for p in x_base])          # images,
    x_fake = np.array([feat(aug(_img(p))) for p in x_base])     # real vs aug
    # (calibration stats still measured on NONORM scanners; pairing removes
    # the biology/dataset confound so the classifier can only win on
    # augmentation artifacts)
    X = np.vstack([x_real, x_fake])
    y = np.r_[np.ones(len(x_real)), np.zeros(len(x_fake))]
    auc = cross_val_score(LogisticRegression(max_iter=1000), X, y,
                          cv=5, scoring="roc_auc").mean()
    assert 0.5 <= auc <= 0.65, (
        f"discriminator AUC {auc:.3f} outside [0.5, 0.65]: "
        f"{'leak' if auc < 0.5 else 'augmentation implausible'}")


@pytest.mark.skipif(not CKPT.exists() or not NCT.exists(),
                    reason="ckpt/dataset not on this machine")
def test_t3_error_preservation_band():
    import torch

    from histogate.model import N_CLASSES, PatchClassifier
    from histogate.stainaug import StainAugmenter, calibrate_from_scanners

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def preds(imgs: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).float() / 255
        with torch.no_grad():
            return model((x - mean) / std).argmax(1).numpy()

    calib = calibrate_from_scanners(_paths(NONORM, 150))
    aug = StainAugmenter(calib, seed=2)
    src = [_img(p) for p in _paths(NCT.parent / "CRC-VAL-HE-7K", 256)]
    flip = float((preds(src) != preds([aug(i) for i in src])).mean())
    assert 0.005 <= flip <= 0.15, (
        f"flip rate {flip:.3f} outside [0.005, 0.15]: "
        f"{'no-op' if flip < 0.005 else 'signal destroyed'}")


@needs_data
def test_t4_physics_directionality():
    from histogate.bio_optics import macenko_stain_matrix
    from histogate.stainaug import StainAugmenter, calibrate_from_scanners

    def conc_paired(before: np.ndarray,
                    after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decompose BOTH images with the SAME W (from `before`): measures
        the applied transform directly, without W re-estimation noise or
        pixel-mask composition shift between the two images."""
        w = macenko_stain_matrix(_od(before))
        mask = _od(before).sum(1) > 0.3
        cb = np.linalg.lstsq(w.T, _od(before)[mask].mean(0), rcond=None)[0]
        ca = np.linalg.lstsq(w.T, _od(after)[mask].mean(0), rcond=None)[0]
        return np.abs(cb), np.abs(ca)

    calib = calibrate_from_scanners(_paths(NONORM, 100))
    aug = StainAugmenter(calib, seed=3)
    h_ratio, e_ratio = [], []
    for p in _paths(NCT, 8):
        img = _img(p)
        cb, ca = conc_paired(img, aug.perturb(img, c_hemo=1.5))
        if cb[0] > 1e-3 and cb[1] > 1e-3:
            h_ratio.append(ca[0] / cb[0])
            e_ratio.append(ca[1] / cb[1])
    assert np.median(h_ratio) == pytest.approx(1.5, abs=0.1), (
        f"hemo ratio {np.median(h_ratio):.3f} != 1.5: perturbation not along H axis")
    assert np.median(e_ratio) == pytest.approx(1.0, abs=0.05), (
        f"eosin ratio {np.median(e_ratio):.3f} != 1.0: H perturbation leaked to E")


@needs_data
def test_t5_identity():
    from histogate.stainaug import StainAugmenter, calibrate_from_scanners

    aug = StainAugmenter(calibrate_from_scanners(_paths(NONORM, 60)), seed=4)
    for p in _paths(NCT, 3):
        img = _img(p)
        out = aug(img, jitter=0.0)
        assert out.dtype == np.uint8 and np.array_equal(out, img), (
            "zero-jitter must be a bitwise identity")
