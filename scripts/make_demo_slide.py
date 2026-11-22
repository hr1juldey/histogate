"""Build data/demo/demo_slide.tif — a small standalone demo WSI crop.

Center-of-mass-of-tissue crop from Camelyon16 tumor_001 level 3, resized to
5500x4500, single-level tiled TIFF (deflate). The slide endpoints min-clamp
the pyramid level, so a one-level file is drop-in for the demo deployment.
"""
from __future__ import annotations

import os

import numpy as np
import tifffile
from PIL import Image

f = tifffile.TiffFile("data/raw/camelyon16/tumor_001.tif")
lvl3 = f.series[0].levels[3]          # 27648 x 12224
h, w = lvl3.shape[:2]
crop = lvl3.asarray()                 # ~1 GB peak
gray = crop.mean(2)
tissue = (gray < 220) & (gray > 20)
col_mass = tissue.sum(0)
com = int((col_mass * np.arange(w)).sum() / max(col_mass.sum(), 1))
cw = 11000
x0 = int(np.clip(com - cw // 2, 0, w - cw))
y0 = (h - 9000) // 2
region = crop[y0:y0 + 9000, x0:x0 + cw]
img = Image.fromarray(region).resize((5500, 4500), Image.LANCZOS)
tifffile.imwrite("data/demo/demo_slide.tif", np.asarray(img),
                 compression="deflate", tile=(512, 512))
print("demo slide:", os.path.getsize("data/demo/demo_slide.tif") // 1024, "KB",
      "shape", img.size, "tissue%",
      round(tissue[:, x0:x0 + cw].mean() * 100, 1))
