"""P18-S1b: tile CAMELYON16 WSIs and extract HistoQC-flagged degraded tiles.

Per slide:
  1. Load HistoQC blemish masks (blurry/dark/bright/fatlike/spur — NOT fuse,
     which includes plain background) from the pipeline output dir.
  2. Union the blemish masks; for each 224-px tile cell (at the level whose
     physical scale matches NCT patches, ~0.5 um/px) compute the flagged
     pixel fraction.
  3. Tiles with flagged fraction >= MIN_FRAC become candidate degraded tiles;
     extracted via openslide at that level and saved as PNG (cap per slide).
  4. Contact sheets (3 rows x 20 cols) rendered to ~/Documents for the HUMAN
     spot-check gate (L3) — flagged tiles only become training truth after
     the user confirms they really show artifacts.

Usage (from repo root, .venv-qc):
  LD_LIBRARY_PATH=.venv-qc/.../openslide_bin python scripts/flag_camelyon_tiles.py \
      <hqc_output_dir> <slides_dir> <out_tiles_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openslide
from PIL import Image

TILE = 224
MIN_FRAC = 0.25
CAP_PER_SLIDE = 60
BLEMISH = ("blurry", "dark", "bright", "fatlike", "spur")
TARGET_UM_PER_PX = 0.5
HOME = Path.home() / "Documents"


def load_masks(slide_out: Path, stem: str) -> np.ndarray | None:
    acc = None
    for b in BLEMISH:
        p = slide_out / f"{stem}_{b}.png"
        if not p.exists():
            continue
        m = np.asarray(Image.open(p).convert("L")) > 0
        acc = m if acc is None else (acc | m)
    return acc


def pick_level(slide: openslide.OpenSlide) -> int:
    """Level whose um/px is closest to the NCT patch scale."""
    best, best_d = 0, 1e9
    for lv in range(slide.level_count):
        try:
            um = float(slide.properties[
                f"openslide.level[{lv}].mpp-x"]) or 0.243 * (2 ** lv)
        except (KeyError, ValueError):
            um = 0.243 * (2 ** lv)
        d = abs(um - TARGET_UM_PER_PX)
        if d < best_d:
            best, best_d = lv, d
    return best


def main(hqc_dir: Path, slides_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_tiles: list[np.ndarray] = []
    total = 0
    for svsp in sorted(slides_dir.glob("*.tif")):
        stem = svsp.name
        slide_out = hqc_dir / stem
        if not slide_out.is_dir():
            print(f"[skip] {stem}: no HistoQC output")
            continue
        flagged = load_masks(slide_out, stem)
        if flagged is None:
            print(f"[skip] {stem}: no blemish masks")
            continue
        slide = openslide.OpenSlide(str(svsp))
        lv = pick_level(slide)
        ds = slide.level_downsamples[lv]
        lw, lh = slide.level_dimensions[lv]
        # tile grid in level-lv coords; mask maps linearly onto level-0
        mask = Image.fromarray((flagged * 255).astype(np.uint8))
        cols = max(1, lw // TILE)
        rows = max(1, lh // TILE)
        fracs = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                box = (int(c * TILE / lw * flagged.shape[1]),
                       int(r * TILE / lh * flagged.shape[0]),
                       int((c + 1) * TILE / lw * flagged.shape[1]),
                       int((r + 1) * TILE / lh * flagged.shape[0]))
                cell = np.asarray(mask.resize(
                    (flagged.shape[1], flagged.shape[0])).crop(box)) > 0
                fracs[r, c] = cell.mean()
        hot = np.argwhere(fracs >= MIN_FRAC)
        np.random.default_rng(0).shuffle(hot)
        kept = 0
        for r, c in hot[:CAP_PER_SLIDE]:
            x0, y0 = int(c * TILE * ds), int(r * TILE * ds)
            region = slide.read_region((x0, y0), lv, (TILE, TILE)).convert("RGB")
            arr = np.asarray(region)
            if arr.mean() < 200:  # skip blank-white
                region.save(out_dir / f"{stem}_r{r}c{c}.png")
                sheet_tiles.append(arr)
                kept += 1
        total += kept
        print(f"[flag] {stem}: grid {rows}x{cols} hot={len(hot)} "
              f"kept={kept} (level {lv}, ds {ds:.1f})")
    # 3x20 contact sheet of flagged tiles
    if sheet_tiles:
        n = min(60, len(sheet_tiles))
        sheet = np.zeros((3 * TILE, 20 * TILE, 3), np.uint8)
        for i in range(n):
            r, c = divmod(i, 20)
            sheet[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = \
                sheet_tiles[i]
        Image.fromarray(sheet).save(HOME / "flagged_contact_sheet.png")
        print(f"contact sheet: {HOME / 'flagged_contact_sheet.png'}")
    print(f"TOTAL flagged tiles: {total}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
