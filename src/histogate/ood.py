"""Feature-space KNN OOD detector — the upgrade path for the energy gate.

Energy-score thresholds are the weakest detector class (measured AUROC 0.797
on CRC-VAL vs Camelyon16). Distance to the training distribution in the
penultimate feature space (cosine KNN, following Sun et al. 2022, "Out-of-
Distribution Detection with Deep Nearest Neighbors") separates far better:
the score of a patch is the negative mean cosine similarity to its k nearest
training features. Familiar tissue sits close; a foreign organ sits far.

No retraining: it reuses the shipped checkpoint's frozen features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from histogate.model import N_CLASSES, PatchClassifier

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_model(ckpt: Path) -> PatchClassifier:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def _pre(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((224, 224))
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255
    return (x - _MEAN) / _STD


@torch.no_grad()
def _embed(model: PatchClassifier, batch: torch.Tensor, device: str) -> torch.Tensor:
    feats = model.backbone(batch.to(device))  # headless trunk -> (B, C) pooled
    return F.normalize(feats, dim=1).cpu()


def features_for_dir(root: Path, ckpt: Path, max_images: int = 500,
                     batch: int = 64) -> torch.Tensor:
    """Pooled L2-normalized penultimate features for up to max_images tiles,
    sampled evenly across the class subdirectories."""
    files: list[Path] = []
    if any(d.is_dir() for d in root.iterdir()):
        dirs = sorted(d for d in root.iterdir() if d.is_dir())
        per = max(1, max_images // len(dirs))
        for d in dirs:
            fs = sorted(d.glob("*.tif")) + sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
            files.extend(fs[:per])
    else:
        files = sorted(p for p in root.iterdir()
                       if p.suffix.lower() in {".tif", ".png", ".jpg"})[:max_images]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(ckpt).to(device)
    out = []
    for i in range(0, len(files), batch):
        xs = torch.stack([
            _pre(Image.open(f)) for f in files[i:i + batch]])
        out.append(_embed(model, xs, device))
    return torch.cat(out)


class KNNDetector:
    """Negative mean cosine similarity to the k nearest training features."""

    def __init__(self, bank: torch.Tensor, k: int = 50, ckpt: Path | None = None) -> None:
        self.bank = F.normalize(bank, dim=1).float()
        self.k = min(k, self.bank.shape[0])
        self.ckpt = ckpt

    @classmethod
    def fit_from_dir(cls, train_root: Path, ckpt: Path, calib: Path,
                     max_images: int = 1500, k: int = 50) -> KNNDetector:
        feats = features_for_dir(train_root, ckpt, max_images=max_images)
        return cls(feats, k=k, ckpt=ckpt)

    @torch.no_grad()
    def score(self, feats: torch.Tensor) -> np.ndarray:
        x = F.normalize(feats, dim=1).float()
        sims = x @ self.bank.T                       # (N, M) cosine
        topk, _ = torch.topk(sims, self.k, dim=1)    # nearest neighbours
        return (-topk.mean(dim=1)).numpy()

    def features_from_wsi(self, path: Path, level: int = 4,
                          max_tiles: int = 200, patch: int = 448) -> torch.Tensor:
        import tifffile

        from histogate.wsi import read_patch, tile_grid, tissue_mask

        with tifffile.TiffFile(path) as tf:
            frame = tf.series[0].levels[level].asarray()
        frame = frame[..., :3]
        coords = tile_grid(frame.shape[:2], tissue_mask(frame), patch=patch, stride=patch)
        if not coords:
            raise ValueError("no tissue tiles found")
        step = max(1, len(coords) // max_tiles)
        coords = coords[::step][:max_tiles]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.ckpt is None:
            raise ValueError("ckpt path required for WSI features")
        model = _load_model(self.ckpt).to(device)
        out = []
        for i in range(0, len(coords), 16):
            xs = []
            for y, x in coords[i:i + 16]:
                crop = read_patch(frame, y, x, patch)
                im = Image.fromarray(crop)
                xs.append(_pre(im))
            out.append(_embed(model, torch.stack(xs), device))
        return torch.cat(out)


def auroc(ood_scores: np.ndarray, id_scores: np.ndarray) -> float:
    """Rank AUROC: 1.0 when every OOD score exceeds every in-dist score."""
    s = np.concatenate([ood_scores, id_scores])
    labels = np.concatenate([np.ones(len(ood_scores)), np.zeros(len(id_scores))])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = labels.sum(), (1 - labels).sum()
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibrate_fpr05(detector: KNNDetector, val_feats: torch.Tensor) -> float:
    """Threshold at 5% FPR on in-distribution data (same doctrine as the
    energy gate's energy_threshold_fpr05)."""
    id_scores = np.sort(detector.score(val_feats))
    idx = max(0, int(np.ceil(0.95 * len(id_scores))) - 1)
    return float(id_scores[idx])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/raw/NCT-CRC-HE-100K"))
    ap.add_argument("--val", type=Path, default=Path("data/raw/CRC-VAL-HE-7K"))
    ap.add_argument("--wsi", type=Path, default=Path("data/raw/camelyon16/tumor_001.tif"))
    ap.add_argument("--ckpt", type=Path, default=Path("runs/pr16-mixed/best.ckpt.pt"))
    ap.add_argument("--calib", type=Path, default=Path("runs/pr16-mixed/calibration.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/pr16-mixed/ood_knn.json"))
    a = ap.parse_args()

    det = KNNDetector.fit_from_dir(a.train, a.ckpt, a.calib, max_images=2000, k=50)
    val = features_for_dir(a.val, a.ckpt, max_images=500)
    neg = det.features_from_wsi(a.wsi, level=4, max_tiles=300)
    auc = auroc(det.score(neg), det.score(val))
    thr = calibrate_fpr05(det, val)
    a.out.write_text(json.dumps({
        "auroc": round(auc, 4), "threshold_fpr05": round(thr, 4),
        "bank_size": int(det.bank.shape[0]), "k": det.k,
    }, indent=2))
    print(f"KNN OOD AUROC {auc:.4f} (bar 0.90) | fpr05 threshold {thr:.4f} -> {a.out}")
