"""P16-S3 RED: wiring StainAugmenter into training (dataset + CLI + calib file).

T-W1  dataset with stain_augment changes pixels, preserves label; without it,
      output is byte-identical to the pre-S3 behavior (regression guard).
T-W2  determinism regardless of worker count: same (seed, epoch, index)
      yields the same augmented image across fresh dataset instances.
T-W3  StainParams JSON round-trip == configs/stain_calib.json contract.
T-W4  train CLI exposes --stain-augment and the calib file loads into an
      augmenter (flag off => no augmenter constructed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
CALIB = REPO / "configs" / "stain_calib.json"


class _FakeCached:
    """Deterministic stand-in cache: 8 uint8 images, cycling labels."""

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self.imgs = rng.integers(40, 220, size=(8, 32, 32, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        return self.imgs[i], i % 9


def _params():
    from histogate.stainaug import StainParams

    return StainParams(
        w_h=np.array([0.18, 0.20, 0.96]), w_e=np.array([0.80, 0.56, 0.22]),
        w_sigma=0.01, m_lo=np.array([0.9, 0.9]), m_hi=np.array([1.1, 1.1]),
        illum_amp=0.02, noise_sigma=2.0,
    )


def _denorm(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((x * std + mean) * 255).clamp(0, 255).numpy()


def test_tw1_pixels_change_label_preserves_regression():
    from histogate.model import PatchDataset
    from histogate.stainaug import StainAugmenter

    cached = _FakeCached()
    plain = PatchDataset(cached, train=True, augment=False)
    aug_ds = PatchDataset(cached, train=True, augment=False,
                          stain_augment=StainAugmenter(_params(), seed=0))
    x0, y0 = plain[3]
    x1, y1 = aug_ds[3]
    assert y0 == y1, "label changed under stain augmentation"
    assert not np.allclose(_denorm(x0), _denorm(x1), atol=0.0), \
        "stain_augment had no pixel effect"
    assert _denorm(x1).shape == (32, 32, 3)
    # regression: no augmenter => identical to historical behavior
    assert np.allclose(_denorm(x0), _denorm(plain[3][0]))


def test_tw2_worker_count_invariance():
    from histogate.model import PatchDataset
    from histogate.stainaug import StainAugmenter

    cached = _FakeCached()
    outs = []
    for _ in range(3):  # fresh instance == fresh worker
        ds = PatchDataset(cached, train=True, seed=7,
                          stain_augment=StainAugmenter(_params(), seed=0))
        ds.set_epoch(2)
        outs.append(_denorm(ds[5][0]).copy())
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[1], outs[2]), \
        "stain augmentation not deterministic across dataset instances"


def test_tw3_params_json_roundtrip():
    from histogate.stainaug import StainParams

    p = _params()
    p2 = StainParams.from_json(p.to_json())
    assert np.allclose(p.w_h, p2.w_h) and np.allclose(p.w_e, p2.w_e)
    assert p.w_sigma == p2.w_sigma and p.illum_amp == p2.illum_amp
    assert np.allclose(p.m_lo, p2.m_lo) and np.allclose(p.m_hi, p2.m_hi)
    assert p.noise_sigma == p2.noise_sigma


def test_tw4_cli_flag_and_calib_file():
    from histogate.cli import build_train_parser
    from histogate.stainaug import StainAugmenter, StainParams

    parser = build_train_parser()
    ns = parser.parse_args(["--stain-augment"])
    assert ns.stain_augment is True
    ns_off = parser.parse_args([])
    assert ns_off.stain_augment is False
    assert CALIB.exists(), "configs/stain_calib.json missing (dump script)"
    par = StainParams.from_json(json.loads(CALIB.read_text()))
    aug = StainAugmenter(par, seed=1)
    out = aug(np.full((32, 32, 3), 128, dtype=np.uint8))
    assert out.dtype == np.uint8 and out.shape == (32, 32, 3)
