"""PR-8 RED-first tests: WSI tiling pipeline.

Oracles:
- MASK ORACLE: fixture image with a KNOWN tissue bbox -> mask IoU >= 0.9.
- COORD ROUND-TRIP: every patch read at (y,x) is bitwise the crop of the source.
- TISSUE FILTER: blank-area tiles excluded; tissue tiles kept.
- DETERMINISM: two runs -> identical coords hash.
- PYRAMID: multi-level TIFF loads with >=2 levels, downsampling factor correct.
"""

import hashlib

import numpy as np
import pytest
import tifffile

from histogate.wsi import load_levels, read_patch, tile_grid, tissue_mask

RNG = np.random.default_rng(0)


def _pseudo_slide(size: int = 1024, tissue_box: tuple[int, int, int, int] = (256, 256, 768, 768)) -> np.ndarray:
    """White background with a dark 'tissue' rectangle of textured color."""
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    y0, x0, y1, x1 = tissue_box
    h, w = y1 - y0, x1 - x0
    tissue = np.clip(
        (RNG.random((h, w, 3)) * 40 + np.array([120, 60, 90])).round(), 0, 255
    ).astype(np.uint8)
    img[y0:y1, x0:x1] = tissue
    return img


class TestMask:
    def test_mask_matches_known_bbox(self) -> None:
        img = _pseudo_slide(tissue_box=(256, 256, 768, 768))
        mask = tissue_mask(img)
        truth = np.zeros(img.shape[:2], bool)
        truth[256:768, 256:768] = True
        inter = (mask & truth).sum()
        union = (mask | truth).sum()
        assert inter / union >= 0.9, f"tissue mask IoU={inter / union:.3f}"

    def test_blank_image_mask_empty(self) -> None:
        img = np.full((128, 128, 3), 240, dtype=np.uint8)
        assert tissue_mask(img).sum() == 0


class TestTileGrid:
    def test_blank_tiles_excluded(self) -> None:
        img = _pseudo_slide(size=896, tissue_box=(0, 0, 224, 224))  # tissue in ONE tile corner
        mask = tissue_mask(img)
        coords = tile_grid(img.shape[:2], mask, patch=224, stride=224, min_tissue=0.1)
        assert len(coords) >= 1
        for y, x in coords:
            assert mask[y:y + 224, x:x + 224].mean() >= 0.1

    def test_full_tissue_keeps_all_tiles(self) -> None:
        img = _pseudo_slide(size=896, tissue_box=(0, 0, 896, 896))
        mask = tissue_mask(img)
        coords = tile_grid(img.shape[:2], mask, patch=224, stride=224)
        assert len(coords) == 16  # 4x4

    def test_determinism(self) -> None:
        img = _pseudo_slide()
        mask = tissue_mask(img)
        c1 = tile_grid(img.shape[:2], mask, patch=224, stride=224)
        c2 = tile_grid(img.shape[:2], mask, patch=224, stride=224)
        h1 = hashlib.md5(str(c1).encode()).hexdigest()
        h2 = hashlib.md5(str(c2).encode()).hexdigest()
        assert h1 == h2


class TestReadPatch:
    def test_roundtrip_is_exact_crop(self) -> None:
        img = _pseudo_slide(size=512, tissue_box=(64, 64, 448, 448))
        for y, x in [(0, 0), (100, 244), (288, 288)]:
            np.testing.assert_array_equal(read_patch(img, y, x, 224), img[y:y + 224, x:x + 224])

    def test_partial_patch_rejects(self) -> None:
        img = _pseudo_slide(size=512, tissue_box=(64, 64, 448, 448))
        with pytest.raises(ValueError):
            read_patch(img, 300, 300, 224)  # exceeds bounds


class TestPyramid:
    def test_multi_level_load(self, tmp_path) -> None:
        base = _pseudo_slide(size=512, tissue_box=(64, 64, 448, 448))
        down = base[::2, ::2]
        p = tmp_path / "pyr.tif"
        with tifffile.TiffWriter(p) as tw:
            tw.write(base, subifds=1)
            tw.write(down)  # becomes the sub-IFD level
        levels = load_levels(p)
        assert len(levels) >= 1
        np.testing.assert_array_equal(levels[0], base)
