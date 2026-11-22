"""Per-nucleus counts, PanNuke taxonomy, via the vendored HoVer-Net (plan B3).

HONESTY LAW: this is a SECOND model — counts, not diagnoses. Only ever invoked
on gate-PASSED tiles. 'neoplastic' is the PanNuke organ-agnostic label
(segmentation-validated on CoNSeP/PanNuke), not a colon-neoplasia diagnosis.

Resolution law (measured & documented, docs/BENCHMARKS.md P0b): the
fast-pannuke checkpoint was trained on 256x256 patches at 0.25 mpp; CRC tiles
are 224x224 at 0.5 mpp. count() upscales 2x to 448 (checkpoint scale) and
runs a 3x3 crop grid with Voronoi dedup — counts are plausibility-grade.

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
CROP = 256          # checkpoint input size
SCALE = 2           # CRC tiles are 0.5 mpp; checkpoint expects 0.25 mpp
BIG = 224 * SCALE   # 448 -> pad to 512 for the 3x3 crop grid
PAD = 512
STRIDE = 128        # crop origins 0/128/256 -> cores overlap 32px
CORE_OFF = (CROP - 164) // 2  # forward crops 256->164, border is unreliable
CENTERS = (128, 256, 384)     # crop core centers; Voronoi boundaries 192/320


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
        """Counts over the 224px tile: 2x upscale to the checkpoint's 0.25 mpp
        scale, 3x3 overlapping 256px crops, per-instance Voronoi dedup.

        Known under-count (measured, documented): a ~7px-equivalent ring at
        the tile edge falls outside all crop cores."""
        out: dict[str, int] = {k: 0 for k in TAXONOMY}
        try:
            big = img.convert("RGB").resize((BIG, BIG), Image.Resampling.LANCZOS)
            a = np.pad(
                np.asarray(big, dtype=np.float32),
                ((32, 32), (32, 32), (0, 0)),
                mode="reflect",
            )
            crops, origins = [], []
            for oy in (0, STRIDE, 2 * STRIDE):
                for ox in (0, STRIDE, 2 * STRIDE):
                    crops.append(a[oy : oy + CROP, ox : ox + CROP])
                    origins.append((ox, oy))
            x = (
                torch.from_numpy(np.stack(crops))
                .permute(0, 3, 1, 2)
                .to(self._dev)
            )
            # fast-pannuke quirk (measured): the checkpoint expects RAW 0-255
            # floats — /255 normalization kills all detection (np frac>0.5:
            # 0.168 raw vs 0.000 normalized)
            with torch.no_grad():
                pred = self._model()(x)
            np_maps = torch.softmax(pred["np"], dim=1)[..., 1].cpu().numpy()
            hv_maps = pred["hv"].cpu().numpy().transpose(0, 2, 3, 1)
            tp_maps = (
                torch.softmax(pred["tp"], dim=1).argmax(dim=1).cpu().numpy()
            )
            for i, (ox, oy) in enumerate(origins):
                seg = self._model().postproc(
                    [np_maps[i][..., None], hv_maps[i], tp_maps[i][..., None]]
                )[0]
                cents = np.asarray(seg["info_dict"]["centroid"]).reshape(-1, 2)
                types = np.asarray(seg["info_dict"]["type"])
                for c, t in zip(cents, types, strict=False):
                    gx = ox + CORE_OFF + c[0]
                    gy = oy + CORE_OFF + c[1]
                    # Voronoi dedup: this crop owns the centroid iff its core
                    # center is the nearest in BOTH axes
                    cx = min(CENTERS, key=lambda v: abs(gx - v))
                    cy = min(CENTERS, key=lambda v: abs(gy - v))
                    if (cx, cy) != (ox + CROP // 2, oy + CROP // 2):
                        continue
                    t = int(t)
                    if 0 < t <= len(TAXONOMY):
                        out[TAXONOMY[t - 1]] += 1
        except Exception:
            return {}  # fail-closed: no counts, no crash
        return {k: v for k, v in out.items() if v or k != "dead"}
