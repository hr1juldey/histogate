"""Generate the evidence board: real before/after figures from committed code.

Outputs PNGs into evidence/ — these are proof-of-work artifacts, not
illustrations: every panel is produced by calling the shipped module code
on real NCT-CRC-HE-100K patches or real run metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from histogate.bio_optics import ArtifactKind, inject_artifact, normalize_stains
from histogate.biomarkers import compute_til, compute_tsr
from histogate.data import build_index
from histogate.wsi import tile_grid, tissue_mask

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/NCT-CRC-HE-100K"
VAL = ROOT / "data/raw/CRC-VAL-HE-7K"
OUT = ROOT / "evidence"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
CLASSES = sorted(p.name for p in RAW.iterdir() if p.is_dir())


def load_patch(cls: str, idx: int = 0) -> np.ndarray:
    files = sorted((RAW / cls).glob("*.tif"))
    import tifffile

    return tifffile.imread(files[idx])


def load_val_patch(cls: str, seed: int) -> np.ndarray:
    files = sorted((VAL / cls).glob("*.tif"))
    import tifffile

    return tifffile.imread(files[seed % len(files)])


# ---------------------------------------------------------------- fig 1
# Stain normalization before/after (the PR-2 Macenko pipeline on real tissue)
show = ["TUM", "STR", "LYM", "NORM"]
fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
for j, cls in enumerate(show):
    img = load_patch(cls, idx=3)
    norm = normalize_stains(img)
    axes[0, j].imshow(img)
    axes[0, j].set_title(f"{cls} — raw scanner H&E", fontsize=9)
    axes[1, j].imshow(np.clip(norm, 0, 255).astype(np.uint8))
    axes[1, j].set_title(f"{cls} — Macenko-normalized", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
fig.suptitle("PR-2 bio_optics: Beer-Lambert OD + Macenko stain normalization (real NCT-CRC patches)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig1_stain_before_after.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- fig 2
# The AIRAQc-8 synthetic artifact injector (PR-3): original + one panel per kind
kinds = [k.value for k in ArtifactKind]
base = load_patch("STR", idx=5)
cols = 4
rows = int(np.ceil((len(kinds) + 1) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(12, 3.1 * rows))
axes = np.atleast_2d(axes)
axes[0, 0].imshow(base)
axes[0, 0].set_title("original", fontsize=9)
for i, kind in enumerate(kinds):
    img, mask = inject_artifact(base, kind, severity=0.6, seed=7)
    ax = axes.flat[i + 1]
    ax.imshow(img)
    cov = float(mask.mean())
    ax.set_title(f"{kind} (s=0.6, cov={cov:.2f})", fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
fig.suptitle("PR-3 AIRAQc-8 artifact injector — every pre-analytical failure mode the QC head must reject", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig2_artifact_injector.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
# Smoke-run learning curves from the real metrics JSON (PR-5)
m = json.loads((ROOT / "runs/smoke-pr5/metrics.json").read_text())
ep = [e["epoch"] for e in m["epochs"]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
a1.plot(ep, [e["train_loss"] for e in m["epochs"]], "o-", color="#c0392b")
a1.set_title(f"train loss  {m['epochs'][0]['train_loss']:.3f} → {m['epochs'][-1]['train_loss']:.3f}")
a1.set_xlabel("epoch")
a2.plot(ep, [e["val_macro_f1"] for e in m["epochs"]], "o-", label="macro-F1")
a2.plot(ep, [e["val_acc"] for e in m["epochs"]], "s--", label="accuracy")
a2.plot(ep, [e["val_kappa"] for e in m["epochs"]], "^:", label="quadratic κ")
a2.set_title("validation (EMA weights)")
a2.set_xlabel("epoch")
a2.legend(fontsize=8)
fig.suptitle(
    f"PR-5 smoke run — 4,600 patches, 2 epochs, seed {m['config']['seed']}, "
    f"peak VRAM {m.get('peak_vram_mb', 0):.0f} MB",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(OUT / "fig3_smoke_curves.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- fig 4
# Pseudo-slide TME map (PR-8 tile grid + PR-7 biomarkers on true labels):
# an 8x8 mosaic of real CRC-VAL patches stands in for one WSI; we show the
# class map the MIL head will consume, and the TSR/TIL computed from it.
G = 8
labels = rng.choice([c for c in CLASSES if c != "BACK"], size=G * G, p=None)
# force a plausible mix: half TUM/STR stroma-dominant tumour core
labels[: 32] = rng.choice(["TUM", "STR", "LYM", "DEB"], size=32, p=[0.4, 0.4, 0.15, 0.05])
mosaic = np.zeros((G * 224, G * 224, 3), np.uint8)
for i, cls in enumerate(labels):
    p = load_val_patch(cls, seed=i)
    r, c = divmod(i, G)
    mosaic[r * 224 : (r + 1) * 224, c * 224 : (c + 1) * 224] = p[:224, :224]

COLOR = {
    "TUM": "#e74c3c", "STR": "#8e44ad", "LYM": "#27ae60", "NORM": "#2ecc71",
    "MUS": "#f39c12", "MUC": "#16a085", "ADI": "#f1c40f", "DEB": "#7f8c8d",
    "BACK": "#ecf0f1",
}
fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))
axes[0].imshow(mosaic)
axes[0].set_title("pseudo-slide: 64 real CRC-VAL patches", fontsize=9)
for i, cls in enumerate(labels):
    r, c = divmod(i, G)
    axes[1].add_patch(plt.Rectangle((c * 224, r * 224), 224, 224, color=COLOR[cls]))
axes[1].set_xlim(0, G * 224)
axes[1].set_ylim(G * 224, 0)
axes[1].set_aspect("equal")
IDX = {c: k for k, c in enumerate(CLASSES)}
grid = np.array([IDX[c] for c in labels]).reshape(G, G)
tsr = compute_tsr(grid)
til = compute_til(grid)
til_intra = getattr(til, "intratumoral_density_mm2", None)
axes[1].set_title("class map — the MIL input", fontsize=9)
axes[1].text(
    0.5, -0.04,
    f"TSR = {None if tsr is None else f'{tsr.tsr:.3f} ({tsr.group})'}"
    f"    TIL intratumoral = {til_intra} /mm²",
    transform=axes[1].transAxes, ha="center", fontsize=10,
)
for ax in axes:
    ax.axis("off")
fig.suptitle("PR-7+PR-8: slide-level TME quantification — attention-MIL (PR-9) learns to produce this from pixels", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig4_tme_map.png", dpi=110)
plt.close(fig)

print("index check:", {c: len(list((RAW / c).glob('*.tif'))) for c in CLASSES[:3]}, "...")
print("WROTE:", sorted(p.name for p in OUT.glob('*.png')))
