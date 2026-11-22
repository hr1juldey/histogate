"""Per-nucleus counts, PanNuke taxonomy, via the vendored HoVer-Net (plan B3).

HONESTY LAW: this is a SECOND model — counts, not diagnoses. Only ever invoked
on gate-PASSED tiles. 'neoplastic' is the PanNuke organ-agnostic label
(segmentation-validated on CoNSeP/PanNuke), not a colon-neoplasia diagnosis.

Resolution caveat (measured & documented, docs/BENCHMARKS.md P0b): the
fast-pannuke checkpoint was trained on 256x256 patches at 0.25 mpp; CRC tiles
are 224x224 at 0.5 mpp. We infer single-pass on the resized tile (native
scale) — absolute counts are plausibility-grade, not calibrated.

Fail-closed law: weights missing / any inference error → ``{}`` (no counts,
no crash). Weights: TIACentre/TIAToolbox_pretrained_weights
hovernet_fast-pannuke.pth (144.1 MiB, HF Hub, NOT redistributed).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

TAXONOMY = ["neoplastic", "inflammatory", "connective", "dead", "nonneop_epithelial"]
# PanNuke ids 1..5 -> our keys (nuc_type_dict order: 1 Neoplastic, 2
# Inflammatory, 3 Connective, 4 Dead, 5 Non-Neoplastic Epithelial)

WEIGHT_REPO = "TIACentre/TIAToolbox_pretrained_weights"
WEIGHT_FILE = "hovernet_fast-pannuke.pth"
INPUT = 256


class NucleusCounter:
    """Lazy per-tile nucleus classifier/counter (vendored HoVer-Net fast)."""

    def __init__(self, device: str | None = None) -> None:
        self._m = None
        self._dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _model(self):
        if self._m is None:
            from histogate.vendor.hovernet import HoVerNet

            weights = self._weights()
            m = HoVerNet(
                num_types=6,
                mode="fast",
                nuc_type_dict={
                    "0": "Background",
                    "1": "Neoplastic",
                    "2": "Inflammatory",
                    "3": "Connective",
                    "4": "Dead",
                    "5": "Non-Neoplastic Epithelial",
                },
            )
            if weights is not None:
                m.load_state_dict(weights, strict=True)
            m.to(self._dev).eval()
            self._m = m
        return self._m

    @staticmethod
    def _weights():
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(WEIGHT_REPO, WEIGHT_FILE)
            return torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            return None  # fail-closed: untrained weights are never used silently

    def count(self, img: Image.Image) -> dict[str, int]:
        out: dict[str, int] = {k: 0 for k in TAXONOMY}
        try:
            arr = np.asarray(
                img.convert("RGB").resize((INPUT, INPUT), Image.Resampling.LANCZOS)
            )
            # fast-pannuke quirk (measured): the checkpoint expects RAW 0-255
            # floats — /255 normalization kills all detection (np frac>0.5:
            # 0.168 raw vs 0.000 normalized)
            x = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)[None].to(
                self._dev
            )
            with torch.no_grad():
                pred = self._model()(x)
            np_map = torch.softmax(pred["np"], dim=1)[0, 1].cpu().numpy()[..., None]
            hv_map = pred["hv"][0].cpu().numpy().transpose(1, 2, 0)
            tp_map = (
                torch.softmax(pred["tp"], dim=1).argmax(dim=1)[0].cpu().numpy()[
                    ..., None
                ]
            )
            seg = self._model().postproc([np_map, hv_map, tp_map])[0]
            types = np.asarray(seg["info_dict"]["type"])
            for t in types:
                t = int(t)
                if 0 < t <= len(TAXONOMY):
                    out[TAXONOMY[t - 1]] += 1
        except Exception:
            return {}  # fail-closed: no counts, no crash
        return {k: v for k, v in out.items() if v or k != "dead"}
