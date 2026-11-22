"""Sweep the /story stain dial through the REAL models and emit a frontend
response curve (story-dial-curve.json). The dial math is deterministic
(Ruifrok deconv + concentration scaling), so the browser can interpolate
the curve instead of needing the live backend.

Usage: cd histogate && uv run python scripts/make_dial_curve.py
"""
from __future__ import annotations

import json
import pathlib
import urllib.request
import uuid

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000/analyze-patch"
SH = np.array([0.644, 0.717, 0.267])
SE = np.array([0.092, 0.954, 0.283])
H = SH / np.linalg.norm(SH)
E = SE / np.linalg.norm(SE)
M = np.stack([H, E])                      # 2x3
PINV = np.linalg.pinv(M)                  # 3x2


def deconvolve(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rgb: (N,3) uint8 -> cH, cE (N,), res (N,3). Mirrors StainDial.tsx."""
    s = rgb.astype(np.float64)
    od = np.where(s > 1, -np.log(np.maximum(s, 1e-9) / 255.0), 0.0)
    he = np.maximum(0.0, od @ PINV)       # (N,2)
    res = od - he @ M
    return he[:, 0].copy(), he[:, 1].copy(), res


def render(src: np.ndarray, t: float) -> Image.Image:
    """Dial at position t -> restained 224x224 PNG (mirrors render())."""
    cH, cE, res = deconvolve(src.reshape(-1, 3))
    aH = 1 + 1.2 * t
    aE = max(0.2, 1 - 0.7 * t)
    out = H[None, :] * (cH * aH)[:, None] + E[None, :] * (cE * aE)[:, None] + res
    px = (255 * np.exp(-np.maximum(0.0, out))).clip(0, 255).astype(np.uint8)
    return Image.fromarray(px.reshape(224, 224, 3))


def analyze(img: Image.Image, model: str) -> dict:
    b = uuid.uuid4().hex
    buf = __import__("io").BytesIO()
    img.save(buf, "PNG")
    body = (f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'
            f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="dial.png"\r\n'
            f'Content-Type: image/png\r\n\r\n').encode() + buf.getvalue() + f'\r\n--{b}--\r\n'.encode()
    req = urllib.request.Request(API, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def main() -> None:
    grid = [round(x, 3) for x in np.linspace(0.0, 1.0, 9)]
    curve: dict[str, list[dict]] = {}
    for tile, fname in [("lab", "tumor.png"), ("raw", "nonorm.png")]:
        src = np.asarray(Image.open(ROOT / "web/public/samples" / fname).convert("RGB").resize((224, 224)))
        pts = []
        for t in grid:
            img = render(src, t)
            row = {"t": t}
            for model in ("v1", "ship"):
                j = analyze(img, model)
                row[model] = {k: j.get(k) for k in
                              ("confidence", "top_class", "energy", "energy_threshold", "ood")}
                if model == "ship":
                    row["gradcam"] = j.get("gradcam")
            pts.append(row)
            print(tile, t, row["v1"]["top_class"], row["v1"]["confidence"],
                  row["ship"]["top_class"], row["ship"]["confidence"], flush=True)
        curve[tile] = pts
    out = ROOT / "web/src/data/story-dial-curve.json"
    out.write_text(json.dumps(curve))
    print("wrote", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
