"""P15-S4: real inference engine — WSI/PNG → SlideReport via the shipped
checkpoint and its measured calibration. No hardcoded defaults: temperature,
energy threshold and model revision come from the run's calibration.json.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from histogate.biomarkers import compute_til, compute_tsr
from histogate.calibration import energy_score
from histogate.model import N_CLASSES, PatchClassifier
from histogate.schemas.report import SlideReport, TissueClass
from histogate.wsi import load_levels, read_patch, tile_grid, tissue_mask

_RUO = "Research Use Only. Not for use in diagnostic procedures."


class _TileDS(Dataset):
    def __init__(self, img: np.ndarray, coords: list[tuple[int, int]]) -> None:
        self.img, self.coords = img, coords

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, i: int) -> torch.Tensor:
        y, x = self.coords[i]
        patch = read_patch(self.img, y, x, 224)
        t = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).float()
        return t.div_(255.0)


def build_analyze(ckpt_path: Path, calib_path: Path,
                  device: str = "cuda") -> Callable[[Path], SlideReport]:
    """Load model + measured calibration once; return analyze(path)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    calib = json.loads(Path(calib_path).read_text())
    temperature = float(calib["temperature"])
    threshold = float(calib["energy_threshold_fpr05"])
    revision = str(calib["git_sha"])

    @torch.no_grad()
    def analyze(path: Path) -> SlideReport:
        img = load_levels(path)[0]
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[..., :3]
        mask = tissue_mask(img)
        coords = tile_grid(img.shape, mask)
        if not coords:
            raise ValueError(f"no tissue tiles found in {path}")

        loader = DataLoader(_TileDS(img, coords), batch_size=32, num_workers=0)
        logits: list[np.ndarray] = []
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        for x in loader:
            xc = x.to(device)
            xc = (xc - mean) / std
            logits.append(model(xc).float().cpu().numpy())
        z = np.concatenate(logits)
        energies = energy_score(z, T=temperature)
        is_ood = energies > threshold

        probs = np.exp(z / temperature)
        probs /= probs.sum(axis=1, keepdims=True)
        preds = z.argmax(axis=1)

        frac = {c: 0.0 for c in TissueClass}
        for p, ood in zip(preds, is_ood, strict=True):
            if not ood:
                frac[list(TissueClass)[int(p)]] += 1.0
        kept = int((~is_ood).sum())
        if kept:
            frac = {k: v / kept for k, v in frac.items()}

        side = int(np.ceil(np.sqrt(len(coords))))
        padded = np.full(side * side, "BACK", dtype=object)
        for slot, (p, ood) in enumerate(zip(preds, is_ood, strict=True)):
            padded[slot] = "BACK" if ood else list(TissueClass)[int(p)].value
        grid = padded.reshape(side, side)
        tsr = compute_tsr(grid)
        til = compute_til(grid)

        return SlideReport(
            slide_id=path.stem, n_patches=len(coords),
            n_gated=int(is_ood.sum()),
            compartment_fractions=frac,
            artifacts=[],
            tsr=tsr, til=til,
            slide_label=None, slide_confidence=None,
            calibration_temperature=temperature,
            model_revision=revision,
            ruo_disclaimer=_RUO,
        )

    return analyze
