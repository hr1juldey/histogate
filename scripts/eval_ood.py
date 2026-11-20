"""OOD AUROC: energy score separates clean tissue from artifact-corrupted input.

ID = clean CRC-VAL patches (external cohort). OOD = the same patches through
the AIRAQc-style artifact injector (8 kinds, severity 0.6). AUROC over energy
scores feeds the gate (ood_auroc_min 0.90). Also reports the calibrated
threshold's realized FPR on clean data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from histogate.bio_optics import ArtifactKind, inject_artifact
from histogate.calibration import energy_score
from histogate.cli import _balanced_subset, _cache_for, build_index, load_settings
from histogate.data import CachedPatches
from histogate.model import N_CLASSES, PatchClassifier, PatchDataset

RUN = Path(os.environ.get("HISTOGATE_RUN", "runs/overnight-30ep"))
CKPT = RUN / "best.ckpt.pt"
N = 1024
SEV = float(os.environ.get("SEV", "0.6"))

settings = load_settings(None)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
model.load_state_dict(ckpt["state_dict"])
model.to("cuda").eval()
dev = next(model.parameters()).device

idx = _balanced_subset(build_index(settings.paths.data_dir / "raw" / "CRC-VAL-HE-7K"), N)
cache = _cache_for(idx, settings.paths.data_dir / "cache" / f"crcval_{len(idx)}.bin")
ds = PatchDataset(CachedPatches(cache, idx, 224), train=False)


def logits_of(x: torch.Tensor) -> np.ndarray:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(x.to(dev).to(memory_format=torch.channels_last)).float().cpu().numpy()


def energy_of(ds_patches) -> np.ndarray:
    out = []
    for x, _ in DataLoader(ds_patches, batch_size=256, num_workers=0):
        out.append(logits_of(x))
    z = np.concatenate(out)
    return energy_score(z), z


e_clean, z_clean = energy_of(ds)
cal = json.loads((RUN / "calibration.json").read_text())
T_FIT = cal["temperature"]

# flag convention: energy > threshold => OOD (higher energy = less
# in-distribution); fit_energy_threshold targets 5% FPR on clean val
def flag_rate(e: np.ndarray, thr: float) -> float:
    return float((e > thr).mean())

# artifact corruption as OOD — cycle the AIRAQc-class artifact kinds
_KINDS = list(ArtifactKind)


class _Corrupted:
    def __init__(self, base) -> None:
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        kind = _KINDS[i % len(_KINDS)]
        # PatchDataset hands float in [0,1] (uint8/255); the injector wants
        # uint8 pixels — exact round-trip, then back to the dataset's scale
        pix = (x * 255.0).round().clamp_(0, 255).to(torch.uint8)
        bad, _mask = inject_artifact(
            pix.numpy().transpose(1, 2, 0), kind, SEV, seed=i)
        return torch.from_numpy(bad).permute(2, 0, 1).float().div_(255.0), y


e_bad, _ = energy_of(_Corrupted(ds))

# per-kind energies for the diagnostic breakdown
per_kind: dict[str, list[float]] = {k.value: [] for k in _KINDS}
dl = DataLoader(ds, batch_size=256, num_workers=0)
i = 0
for x, _y in dl:
    pix = (x * 255.0).round().clamp_(0, 255).to(torch.uint8)
    for j in range(pix.shape[0]):
        kind = _KINDS[i % len(_KINDS)]
        bad, _m = inject_artifact(
            pix[j].numpy().transpose(1, 2, 0), kind, SEV, seed=i)
        xt = torch.from_numpy(bad).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = model(xt.to(dev).to(memory_format=torch.channels_last))
        per_kind[kind.value].append(float(energy_score(z.float().cpu().numpy())[0]))
        i += 1

# AUROC: higher energy = more OOD, so artifact is the positive class
labels = np.concatenate([np.zeros_like(e_clean), np.ones_like(e_bad)])
order = np.argsort(np.concatenate([e_clean, e_bad]))
ranks = np.empty_like(order)
ranks[order] = np.arange(len(labels))
n_pos = int(labels.sum())
n_neg = len(labels) - n_pos
auroc = (ranks[labels == 1].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)

thr = cal["energy_threshold_fpr05"]
# global-corruption kinds: artifact affects the WHOLE patch — the design
# scope of an energy OOD detector. Local blemishes (fold/line/pen/bubble)
# are a QC-segmentation problem, deliberately out of scope.
GLOBAL_KINDS = ("no_tissue", "out_of_focus", "missing_tissue")
e_global = np.concatenate([np.array(per_kind[k]) for k in GLOBAL_KINDS])
lab_g = np.concatenate([np.zeros_like(e_clean), np.ones_like(e_global)])
order_g = np.argsort(np.concatenate([e_clean, e_global]))
ranks_g = np.empty_like(order_g)
ranks_g[order_g] = np.arange(len(lab_g))
np_g = int(lab_g.sum())
auroc_global = (ranks_g[lab_g == 1].sum() - np_g * (np_g - 1) / 2) / (np_g * (len(lab_g) - np_g))

out = {
    "n_id": int(len(e_clean)), "n_ood": int(len(e_bad)),
    "artifact_severity": SEV,
    "energy_auroc": round(float(auroc), 4),
    "energy_auroc_global_kinds": round(float(auroc_global), 4),
    "global_kinds": list(GLOBAL_KINDS),
    "threshold_fpr05": thr,
    "clean_fpr_at_threshold": round(flag_rate(e_clean, thr), 4),
    "artifact_flag_rate_at_threshold": round(flag_rate(e_bad, thr), 4),
    "energy_mean_clean": round(float(e_clean.mean()), 3),
    "energy_mean_artifact": round(float(e_bad.mean()), 3),
    "per_kind_flag_rate": {
        k: round(flag_rate(np.array(v), thr), 3) for k, v in per_kind.items()},
    "per_kind_energy_mean": {
        k: round(float(np.mean(v)), 3) for k, v in per_kind.items()},
}
(RUN / f"ood_sev{SEV:g}.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
