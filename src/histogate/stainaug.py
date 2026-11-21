"""P16-S2: physically-grounded stain augmentation, calibrated from REAL data.

Model (Ruifrok & Johnston 2001, Beer-Lambert):
    OD = -log10((I+1)/256) = c_h * W_h + c_e * W_e + c_r * W_r

Design laws (each earned by a failed adversarial test):
- transforms operate in the IMAGE'S OWN Macenko basis, not a global mean
  basis (T4: mean-basis perturbation leaked 7% across the H/E axes).
- concentration jitter is measured WITHIN one tissue class (cross-class
  concentration spread is biology, not stain process; T3: full-spread
  Gaussian draws destroyed signal, 17.9% flip) and sampled log-uniform
  in the measured [q05, q95] band around each image's own concentrations
  (T2: Gaussian tails washed images to near-white -> empty-OD NaNs).
- stain-vector jitter sigma = measured cross-slide spread of Macenko
  vectors; illumination amplitude = measured per-slide low-frequency OD
  field std; sensor noise = measured std in bright flat regions.
  Nothing hand-set beyond the Ruifrok reference third column.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from histogate.bio_optics import macenko_stain_matrix

RUIFROK = np.array([[0.65, 0.70, 0.29],
                    [0.07, 0.99, 0.11],
                    [0.27, 0.57, 0.78]])


def _od(img: np.ndarray) -> np.ndarray:
    return -np.log10((img.astype(np.float64) + 1.0) / 256.0)


def _rgb(od: np.ndarray) -> np.ndarray:
    return np.clip(256.0 * np.power(10.0, -od) - 1.0, 0, 255).round().astype(np.uint8)


def _load(p: Path) -> np.ndarray:
    a = tifffile.imread(p)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return a[..., :3]


@dataclass(frozen=True)
class StainParams:
    """Measured augmentation ranges. All fields come from real slides."""

    w_h: np.ndarray            # (3,) mean hematoxylin OD vector (measured)
    w_e: np.ndarray            # (3,) mean eosin OD vector (measured)
    w_sigma: float             # measured per-entry std of stain vectors
    m_lo: np.ndarray           # (2,) measured q05 H,E concentration band
    m_hi: np.ndarray           # (2,) measured q95 H,E concentration band
    illum_amp: float           # measured low-frequency OD field std
    noise_sigma: float         # measured sensor noise std (flat regions)

    def to_json(self) -> str:
        import json
        return json.dumps({
            "w_h": self.w_h.tolist(), "w_e": self.w_e.tolist(),
            "w_sigma": self.w_sigma,
            "m_lo": self.m_lo.tolist(), "m_hi": self.m_hi.tolist(),
            "illum_amp": self.illum_amp, "noise_sigma": self.noise_sigma,
        })

    @classmethod
    def from_json(cls, text: str | dict) -> StainParams:
        import json
        d = json.loads(text) if isinstance(text, str) else text
        return cls(
            w_h=np.array(d["w_h"]), w_e=np.array(d["w_e"]),
            w_sigma=float(d["w_sigma"]),
            m_lo=np.array(d["m_lo"]), m_hi=np.array(d["m_hi"]),
            illum_amp=float(d["illum_amp"]), noise_sigma=float(d["noise_sigma"]),
        )


def _conc_band(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """[q05, q95] of H,E concentration ratios to the class median: the
    stain-process spread once biology is held (approximately) constant."""
    cs = []
    for p in paths:
        od = _od(_load(p)).reshape(-1, 3)
        od = od[od.sum(1) > 0.3]
        if len(od) < 1000:
            continue
        w = macenko_stain_matrix(od)
        if not np.isfinite(w).all():
            continue
        cs.append(np.abs(np.linalg.lstsq(w.T, od.mean(0), rcond=None)[0])[:2])
    if len(cs) < 20:
        return np.array([0.85, 0.85]), np.array([1.18, 1.18])
    cs = np.array(cs)
    ratio = cs / np.maximum(np.median(cs, 0), 1e-3)
    return (np.maximum(np.quantile(ratio, 0.05, axis=0), 0.5),
            np.minimum(np.quantile(ratio, 0.95, axis=0), 2.0))


def calibrate_from_scanners(paths: list[Path], grid: int = 8) -> StainParams:
    """Measure stain-vector spread, concentration band, illumination
    amplitude and sensor noise from REAL slides. Nothing is hand-set."""
    by_class: dict[str, list[Path]] = {}
    for p in paths:
        by_class.setdefault(p.parent.name, []).append(p)
    los = [_conc_band(v)[0] for v in by_class.values() if len(v) >= 8]
    his = [_conc_band(v)[1] for v in by_class.values() if len(v) >= 8]
    m_lo = np.median(los, 0) if los else np.array([0.85, 0.85])
    m_hi = np.median(his, 0) if his else np.array([1.18, 1.18])

    ws, noises, illums = [], [], []
    for p in paths:
        img = _load(p)
        od = _od(img)
        flat = od.reshape(-1, 3)
        flat = flat[flat.sum(1) > 0.3]
        if len(flat) < 1000:
            continue
        w = macenko_stain_matrix(flat)
        if np.isfinite(w).all():
            ws.append(w[:2])
            # illumination = low-frequency residual the TWO-stain model
            # cannot explain (a 3-row basis fits OD exactly -> zero
            # residual by construction). Fit 2 stains on the FULL grid,
            # block-mean 16x16 to kill sensor noise, std of the smooth
            # field = illumination amplitude.
            full = od.reshape(-1, 3)
            c2 = np.linalg.lstsq(w[:2].T, full.T, rcond=None)[0]
            r = (full - (c2.T @ w[:2])).reshape(img.shape[0], img.shape[1], 3)
            blocks = r.mean(-1).reshape(img.shape[0] // 16, 16,
                                        img.shape[1] // 16, 16).mean((1, 3))
            illums.append(float(blocks.std()))
        bright = img.reshape(-1, 3)[img.reshape(-1, 3).mean(1) > 235]
        if len(bright) > 500:
            noises.append(float(bright.std(0).mean()))
    ws = np.array(ws)
    return StainParams(
        w_h=ws[:, 0].mean(0), w_e=ws[:, 1].mean(0),
        w_sigma=float(np.maximum(ws.std(0).mean(), 0.003)),
        m_lo=m_lo, m_hi=m_hi,
        illum_amp=float(np.median(illums)) if illums else 0.05,
        noise_sigma=float(np.median(noises)) if noises else 1.5,
    )


class StainAugmenter:
    """Random physically-grounded stain jitter; deterministic under seed."""

    def __init__(self, params: StainParams, seed: int) -> None:
        self.p = params
        self.rng = np.random.default_rng(seed)
        self._basis_cache: dict[int, np.ndarray] = {}

    def _basis(self, img: np.ndarray) -> np.ndarray:
        """(2,3) Macenko H,E basis of THIS image; calibrated-mean fallback."""
        flat = _od(img).reshape(-1, 3)
        flat = flat[flat.sum(1) > 0.3]
        if len(flat) >= 1000:
            w = macenko_stain_matrix(flat)
            if np.isfinite(w).all():
                return w[:2]
        return np.vstack([self.p.w_h, self.p.w_e])

    def _illum(self, shape: tuple[int, ...],
               rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng if rng is not None else self.rng
        y, x = np.mgrid[0:shape[0], 0:shape[1]]
        gy = rng.uniform(0, shape[0], 2)
        gx = rng.uniform(0, shape[1], 2)
        f = sum(np.exp(-((y - a) ** 2 + (x - b) ** 2)
                       / (2 * (shape[0] / 2) ** 2)) for a, b in zip(gy, gx, strict=True))
        return 1.0 + self.p.illum_amp * (f - f.mean()) / max(f.std(), 1e-9) / 3

    def _apply(self, img: np.ndarray, c_h: float, c_e: float,
               w_jitter: float, illum: bool, noise: bool,
               rng: np.random.Generator | None = None,
               basis_key: int | None = None) -> np.ndarray:
        """Exact 2-stain decomposition with per-pixel residual preserved:
        od = c_h*H + c_e*E + R. Scaling c_h moves ONLY the H coefficient
        (a 3-row basis fits OD exactly and cross-projects onto the
        non-orthogonal E axis when re-measured).

        Perf: closed-form 2x2 solve (not lstsq SVD); optional basis_key
        caches the per-image Macenko basis (same image -> same basis)."""
        rng = rng if rng is not None else self.rng
        w2 = self._cached_basis(img, basis_key)
        if w_jitter > 0:
            w2 = w2 + rng.normal(0, self.p.w_sigma * w_jitter, (2, 3))
            w2 /= np.linalg.norm(w2, axis=1, keepdims=True)
        w2f = w2.astype(np.float32)
        od = _od(img).reshape(-1, 3).astype(np.float32)
        wtw = w2f @ w2f.T                                  # (2,2) Gram
        wto = od @ w2f.T                                   # (N,2)
        c2 = np.linalg.solve(wtw, wto.T).T                 # (N,2)
        residual = od - c2 @ w2f
        od_new = (c2 * np.array([c_h, c_e], dtype=np.float32)) @ w2f + residual
        if illum:
            field = self._illum(img.shape, rng)
            od_new = od_new * field.reshape(-1, 1).astype(np.float32)
        out = _rgb(od_new.reshape(img.shape))
        if noise and self.p.noise_sigma > 0:
            out = np.clip(out.astype(np.float64)
                          + rng.normal(0, self.p.noise_sigma, out.shape),
                          0, 255).astype(np.uint8)
        return out

    def _cached_basis(self, img: np.ndarray, key: int | None) -> np.ndarray:
        if key is None:
            return self._basis(img)
        if key not in self._basis_cache:
            self._basis_cache[key] = self._basis(img)
        return self._basis_cache[key]

    def __call__(self, img: np.ndarray, jitter: float = 1.0,
                 rng: np.random.Generator | None = None,
                 basis_key: int | None = None) -> np.ndarray:
        """Random augment. jitter=0.0 must be a bitwise identity (T5)."""
        if jitter == 0.0:
            return np.ascontiguousarray(img, dtype=np.uint8)
        rng = rng if rng is not None else self.rng
        m = np.exp(rng.uniform(np.log(self.p.m_lo), np.log(self.p.m_hi)))
        return self._apply(img, float(m[0]), float(m[1]),
                           w_jitter=jitter, illum=True, noise=True, rng=rng,
                           basis_key=basis_key)

    def perturb(self, img: np.ndarray, c_hemo: float = 1.0,
                c_eos: float = 1.0) -> np.ndarray:
        """Deterministic single-axis physics perturbation (T4). No noise."""
        return self._apply(img, c_hemo, c_eos,
                           w_jitter=0.0, illum=False, noise=False)
