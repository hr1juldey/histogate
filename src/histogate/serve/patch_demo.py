"""Patch-level demo analysis: calibrated probabilities, energy OOD verdict,
and a Grad-CAM localization heatmap for the demo UI.

Grad-CAM: features of the last conv stage hooked; gradient of the top
calibrated logit flows back; channel-weighted activation upsampled to 224.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from histogate.calibration import energy_score
from histogate.data import CLASS_ORDER
from histogate.model import N_CLASSES, PatchClassifier

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class PatchAnalyzer:
    def __init__(self, ckpt_path: Path, calib_path: Path) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model: PatchClassifier = PatchClassifier(
            pretrained=False, num_classes=N_CLASSES)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        calib = json.loads(Path(calib_path).read_text())
        self.temperature = float(calib["temperature"])
        self.energy_thr = float(calib["energy_threshold_fpr05"])
        # optional feature-space KNN OOD detector (calibration key
        # "ood_detector": "knn" flips the gate; absent -> energy, unchanged).
        # Measured AUROC 0.9599 vs energy 0.7972 — see runs/pr16-mixed/ood_knn.json
        self._knn = None
        if calib.get("ood_detector") == "knn":
            import torch as _t

            from histogate.ood import KNNDetector

            ck = _t.load(calib_path.parent / "ood_knn_bank.pt",
                         map_location="cpu", weights_only=True)
            self._knn = KNNDetector(ck["bank"], k=int(ck["k"]))
            self.knn_thr = float(calib.get("knn_threshold_fpr05", -0.9163))
        self._feats: list[torch.Tensor] = []
        self.model.backbone.stages[-1].register_forward_hook(
            lambda m, i, o: self._feats.append(o))
        # optional conformal guarantee (P0a): runs/<run>/conformal.json beside
        # the calibration. Absent -> response shape unchanged (no key).
        self.conformal_qhat: float | None = None
        cpath = Path(calib_path).parent / "conformal.json"
        if cpath.exists():
            _c = json.loads(cpath.read_text())
            try:
                self.conformal_qhat = float(_c["alphas"]["0.1"])
            except (KeyError, TypeError, ValueError):
                self.conformal_qhat = None
        # optional per-nucleus counts (P0b): calibration key "cells": "pannuke"
        # flips the second-model stage ON — only after
        # runs/pr16-mixed/cells_validation.json passed (flag law). Lazy.
        self.cells_mode = calib.get("cells") == "pannuke"
        self._cells = None

    def _pre(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((224, 224))
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255
        return ((x - _MEAN) / _STD).unsqueeze(0)

    @torch.enable_grad()
    def analyze(self, img: Image.Image) -> dict:
        x = self._pre(img)
        self._feats.clear()
        logits = self.model(x)
        z = logits[0] / self.temperature
        probs = F.softmax(z.detach(), dim=0)
        energy = float(energy_score(logits.detach(), T=self.temperature)[0])
        feats = self._feats[-1]            # grad-enabled, for Grad-CAM below
        if self._knn is not None:
            # SAME trunk path the calibration bank was built from
            # (features_for_dir -> model.backbone). Never the stages[-1]
            # hook output: it is pre-final-norm and scores differently —
            # a naive flip on it rejected 100% of CRC-VAL.
            with torch.no_grad():
                emb = self.model.backbone(x)
            is_ood = bool(self._knn.score(emb)[0] > self.knn_thr)
        else:
            is_ood = energy > self.energy_thr
        top = int(probs.argmax())

        cam = np.zeros((224, 224), dtype=np.float32)
        if feats.dim() == 4:
            grads, = torch.autograd.grad(z[top], feats)
            w = grads.mean(dim=(2, 3), keepdim=True)
            a = torch.relu((w * feats).sum(1))[0].detach().numpy()
            a -= a.min()
            if a.max() > 0:
                a /= a.max()
            cam = np.asarray(
                Image.fromarray((a * 255).astype(np.uint8))
                .resize((224, 224))) / 255
        result = {
            "probabilities": [
                {"class": CLASS_ORDER[i], "p": round(float(probs[i]), 4)}
                for i in range(N_CLASSES)],
            "top_class": CLASS_ORDER[top],
            "confidence": round(float(probs[top]), 4),
            "energy": round(energy, 4),
            "energy_threshold": round(self.energy_thr, 4),
            "ood": bool(is_ood),
            "verdict": ("OUT-OF-DISTRIBUTION — not colorectal H&E "
                        "at the trained scale" if is_ood else
                        f"{CLASS_ORDER[top]} (calibrated confidence)"),
            "gradcam": cam.round(3).tolist(),
        }
        if not is_ood and self.conformal_qhat is not None:
            from histogate.conformal import aps_set

            result["conformal_set"] = aps_set(
                probs.detach().numpy(), self.conformal_qhat)
        # P0b: second-model nucleus counts — gate-passed, non-BACK tiles only.
        # Declined or BACK tiles: no `nuclei` key at all (fail-closed law).
        if (not is_ood and self.cells_mode
                and result["top_class"] != "BACK"):
            if self._cells is None:
                from histogate.cells import NucleusCounter

                self._cells = NucleusCounter()
            result["nuclei"] = self._cells.count(img)
        return result
