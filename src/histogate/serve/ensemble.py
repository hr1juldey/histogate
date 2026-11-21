"""P17-S2: EnsembleEngine — mean softmax over checkpoints × TTA transforms.

Design laws (from the P17-S1 adversarial tests):
- TTA admits ONLY label-safe transforms. {identity, hflip, vflip, rot180}
  passed T3 on real labeled patches (<= 1% accuracy loss); rot90/270 are
  NOT in the default set — the model was trained on axis-aligned patches
  and rot90 is a different geometry, admitted only if a future T3 run
  proves them safe (refine hook P17-S3).
- Mean of softmax (not logits): members are temperature-calibrated
  independently; averaging probabilities preserves each member's
  calibration (T2 gain-localization must still hold).
"""
from __future__ import annotations

import torch
from torch import nn

from histogate.model import N_CLASSES, PatchClassifier

TTA_TRANSFORMS: tuple[str, ...] = ("identity", "hflip", "vflip", "rot180")


def _apply_tf(x: torch.Tensor, tf: str) -> torch.Tensor:
    if tf == "hflip":
        return x.flip(-1)
    if tf == "vflip":
        return x.flip(-2)
    if tf == "rot180":
        return x.flip(-1).flip(-2)
    return x


class EnsembleEngine:
    """Mean-softmax ensemble over checkpoint paths × allowed transforms."""

    def __init__(self, ckpt_paths: list[str], tta: bool = True) -> None:
        if len(ckpt_paths) < 2:
            raise ValueError("ensemble needs >= 2 checkpoints")
        self.models: list[nn.Module] = []
        for p in ckpt_paths:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            m = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.models.append(m)
        self.transforms: tuple[str, ...] = TTA_TRANSFORMS if tta else ("identity",)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, batch: int = 128) -> torch.Tensor:
        """Mean softmax over (model, transform). x: normalized (N,3,H,W)."""
        probs = torch.zeros(x.shape[0], N_CLASSES)
        n_views = len(self.models) * len(self.transforms)
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch]
            acc = torch.zeros(xb.shape[0], N_CLASSES)
            for m in self.models:
                for tf in self.transforms:
                    acc += torch.softmax(m(_apply_tf(xb, tf)), dim=1)
            probs[i:i + batch] = acc / n_views
        return probs

    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).argmax(1)
