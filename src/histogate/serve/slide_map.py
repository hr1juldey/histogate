"""One-slide verdict map for the /slide page (PR: whole-slide view).

Tiles a whole-slide image on its tissue mask, runs the shipped patch
analyzer per tile, and memoizes the result per process — the first
request computes, every later request is a dict lookup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from histogate.serve.patch_demo import PatchAnalyzer
from histogate.wsi import read_patch, tile_grid, tissue_mask

ELO, EHI = -4.6, -1.6  # shared display scale: -4.6 familiar … -1.6 unfamiliar
_MAX_TILES = 4000

_cache: dict[str, dict] = {}
_frames: dict[str, tuple] = {}  # key -> (img ndarray, patch, stride) for tile crops


def _familiarity(energy: float) -> int:
    """Familiarity 0-100 = 100 - needle position on the display scale."""
    f = 100 * (1 - (energy - ELO) / (EHI - ELO))
    return max(0, min(100, round(f)))


def build_slide_map(slide_path: Path, analyzer: PatchAnalyzer,
                    patch: int = 448, stride: int | None = None,
                    level: int = 4) -> dict:
    """level: pyramid level to tile. Level 0 of a Camelyon16 slide decodes
    to ~65 GB of RAM; level 4 (~13824x6112) covers the whole slide in a
    few hundred tiles — the right demo granularity."""
    key = f"v2|{slide_path}|{patch}|{analyzer.energy_thr}|{level}"
    if key in _cache:
        return _cache[key]
    stride = stride or patch
    # decode ONLY the requested level — wsi.load_levels() materializes every
    # level including a ~65 GB level-0, which OOMs this machine
    with tifffile.TiffFile(str(slide_path)) as tf:
        levels = tf.series[0].levels
        if not levels:
            raise ValueError(f"cannot read {slide_path}")
        img = levels[min(level, len(levels) - 1)].asarray()
    mask = tissue_mask(img)
    coords = tile_grid(img.shape, mask, patch=patch, stride=stride)
    ys = [c[0] for c in coords]
    xs = [c[1] for c in coords]
    if len(ys) > _MAX_TILES:  # thin out deterministically, keep coverage
        step = len(ys) / _MAX_TILES
        keep = [round(i * step) for i in range(_MAX_TILES)]
        ys, xs = [ys[i] for i in keep], [xs[i] for i in keep]
    if not ys:
        raise ValueError("no tissue tiles in slide")
    cols = (max(xs) // stride) + 1
    rows = (max(ys) // stride) + 1
    tiles = []
    for i, (y, x) in enumerate(zip(ys, xs, strict=True)):
        pil = Image.fromarray(np.asarray(read_patch(img, y, x, patch)))
        r = analyzer.analyze(pil)
        tiles.append({
            "id": f"t{i}", "col": x // stride, "row": y // stride,
            "cls": r["top_class"], "conf": r["confidence"],
            "energy": r["energy"], "familiarity": _familiarity(r["energy"]),
            "declined": bool(r["ood"]),
        })
    out = {"slide": slide_path.name,
           "grid": {"cols": int(cols), "rows": int(rows)},
           "dims": {"w": int(img.shape[1]), "h": int(img.shape[0])},
           "patch": int(patch), "stride": int(stride),
           "gate": analyzer.energy_thr, "tiles": tiles}
    _cache[key] = out
    _frames[key] = (img, patch, stride)
    return out


def slide_frame(slide_path: Path, analyzer: PatchAnalyzer,
                level: int = 4) -> tuple | None:
    """The decoded level array for a built map (None if not built yet)."""
    key = f"v2|{slide_path}|448|{analyzer.energy_thr}|{level}"
    return _frames.get(key)
