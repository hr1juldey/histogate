"""P17-S1 RED: adversarial ensemble tests (EnsembleEngine NOT built yet).

T1  error-overlap: the two shipping checkpoints' error sets on a CRC-VAL
    subset must NOT overlap almost completely — an ensemble of two models
    that misclassify the same patches is inert. Overlap >= 0.9 fails.
T2  gain-localization: EnsembleEngine's improvement over the best single
    model must be at least twice as large on the disagreement subset
    (where members differ) as on the agreement subset. A uniform small
    gain everywhere = the ensemble did nothing member-specific (or one
    member dominates); gain localized where members disagree = genuine
    voting.
T3  TTA-label-safety: each candidate transform {identity, hflip, vflip,
    rot180} must keep single-model accuracy within 1% of identity on real
    labeled patches; a transform that loses more is BANNED from TTA.

Skipped where artifacts/datasets absent (CI runners).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

REPO = Path(__file__).resolve().parents[1]
CKPTS = [REPO / "runs" / "pr16-mixed" / "best.ckpt.pt",
         REPO / "runs" / "overnight-30ep" / "best.ckpt.pt"]
CRCVAL = REPO / "data" / "raw" / "CRC-VAL-HE-7K"

needs_all = pytest.mark.skipif(
    not all(p.exists() for p in CKPTS) or not CRCVAL.exists(),
    reason="ckpts/dataset not on this machine")

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _paths(n: int) -> list[Path]:
    out: list[Path] = []
    for cls in sorted(p for p in CRCVAL.iterdir() if p.is_dir()):
        out += sorted(cls.glob("*.tif"))[: max(1, n // 9)]
    return out[:n]


def _batch(paths: list[Path], tf: str = "identity") -> torch.Tensor:
    x = np.stack([tifffile.imread(p)[..., :3] for p in paths])
    if tf == "hflip":
        x = x[:, :, ::-1]
    elif tf == "vflip":
        x = x[:, ::-1]
    elif tf == "rot180":
        x = x[:, ::-1, ::-1]
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float() / 255
    return (t - _MEAN) / _STD


def _model(path: Path):
    from histogate.model import N_CLASSES, PatchClassifier

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


@needs_all
def test_t1_error_overlap():
    from histogate.data import CLASS_TO_IDX

    paths = _paths(2000)
    labels = np.array([CLASS_TO_IDX[p.parent.name] for p in paths])
    errors = []
    for ck in CKPTS:
        m = _model(ck)
        preds = []
        with torch.no_grad():
            for i in range(0, len(paths), 128):
                preds.append(m(_batch(paths[i:i + 128])).argmax(1))
        p = torch.cat(preds).numpy()
        errors.append(set(np.where(p != labels)[0].tolist()))
    both = len(errors[0] & errors[1])
    overlap = both / max(min(len(errors[0]), len(errors[1])), 1)
    assert overlap < 0.9, (
        f"error-set overlap {overlap:.2f} >= 0.90: second member adds "
        f"nothing — ensemble inert (|E1|={len(errors[0])}, |E2|={len(errors[1])})")


@needs_all
def test_t2_gain_localization():
    from histogate.data import CLASS_TO_IDX
    from histogate.serve.ensemble import EnsembleEngine

    paths = _paths(2000)
    labels = np.array([CLASS_TO_IDX[p.parent.name] for p in paths])
    engine = EnsembleEngine([str(p) for p in CKPTS], tta=False)

    def acc(preds: np.ndarray, idx: list[int]) -> float:
        return float((preds[idx] == labels[idx]).mean())

    ens = engine.predict_labels(_batch(paths)).numpy()
    single_preds = []
    for ck in CKPTS:
        m = _model(ck)
        preds = []
        with torch.no_grad():
            for i in range(0, len(paths), 128):
                preds.append(m(_batch(paths[i:i + 128])).argmax(1))
        single_preds.append(torch.cat(preds).numpy())
    dis = np.where(single_preds[0] != single_preds[1])[0].tolist()
    agr = np.where(single_preds[0] == single_preds[1])[0].tolist()
    gain_dis = acc(ens, dis) - max(acc(single_preds[b], dis) for b in range(2))
    gain_agr = acc(ens, agr) - max(acc(single_preds[b], agr) for b in range(2))
    assert gain_dis >= 2 * gain_agr, (
        f"ensemble gain not localized on disagreement: dis {gain_dis:.4f} "
        f"vs agr {gain_agr:.4f} — voting adds nothing member-specific")


@needs_all
def test_t3_tta_label_safety():
    from histogate.data import CLASS_TO_IDX

    paths = _paths(500)
    labels = np.array([CLASS_TO_IDX[p.parent.name] for p in paths])
    m = _model(CKPTS[0])
    base = None
    for tf in ("identity", "hflip", "vflip", "rot180"):
        preds = []
        with torch.no_grad():
            for i in range(0, len(paths), 128):
                preds.append(m(_batch(paths[i:i + 128], tf)).argmax(1))
        acc = float((torch.cat(preds).numpy() == labels).mean())
        if tf == "identity":
            base = acc
            continue
        assert acc >= base - 0.01, (
            f"TTA transform {tf} loses {base - acc:.4f} (> 1%): banned")
