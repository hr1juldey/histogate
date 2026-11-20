"""INT8 vs FP32 biomarker drift: TSR/TIL on pseudo-slides from CRC-VAL.

The deployment question: does dynamic INT8 quantization move the CLINICAL
outputs (TSR fraction, TIL density) beyond tolerance? Pseudo-slides = 8x8
patch grids from the external cohort; predictions from fp32 torch (serving)
vs INT8 ONNX; drift = |TSR_i - TSR_f| and relative TIL change. Gate:
tsr_int8_drift_max 0.05, til_int8_drift_rel_max 0.10.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch.utils.data import DataLoader

from histogate.biomarkers import compute_til, compute_tsr
from histogate.cli import _cache_for, build_index, load_settings
from histogate.data import CLASS_ORDER, CachedPatches, PatchIndex
from histogate.export.onnx_export import export_onnx
from histogate.model import CLASS_TO_IDX, N_CLASSES, PatchClassifier, PatchDataset

RUN = Path(os.environ.get("HISTOGATE_RUN", "runs/overnight-30ep"))
CKPT = RUN / "best.ckpt.pt"
N_SLIDES = 24
G = 8  # 8x8 = 64 patches per pseudo-slide

settings = load_settings(None)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
model.load_state_dict(ckpt["state_dict"])
model.eval().to("cuda")

# int8: export fp32, strip torch's declared value_info (one annotation
# conflicts with ORT shape inference: head 768-vs-9), re-derive, quantize

src32 = RUN / "onnx" / "histogate-int8src.onnx"
export_onnx(model, src32, precision="fp32")
g = onnx.load(str(src32))  # loads external .data into memory
g.graph.ClearField("value_info")
clean = src32.with_name("histogate-int8clean.onnx")
onnx.save_model(g, str(clean), save_as_external_data=False)
int8_path = src32.with_name("histogate-int8.onnx")
quantize_dynamic(str(clean), str(int8_path), weight_type=QuantType.QUInt8)
model.to("cuda")  # export_onnx's .cpu() moves the module in place
sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
in8 = sess8.get_inputs()[0].name

# ---- class-targeted patches: enough per class to assemble MANY slides ----
full = build_index(settings.paths.data_dir / "raw" / "CRC-VAL-HE-7K")
PER_CLASS = 64
by_class: dict[str, list] = {}
for e in full.entries:
    by_class.setdefault(e.label, []).append(e)
sel = PatchIndex([e for c in ("TUM", "STR", "LYM", "MUC", "MUS", "NORM", "ADI", "DEB")
                  for e in by_class[c][:PER_CLASS]])
cache = _cache_for(sel, settings.paths.data_dir / "cache" / f"crcval_cls{len(sel)}.bin")
ds = PatchDataset(CachedPatches(cache, sel, 224), train=False)

xs, zs_fp = [], []
with torch.no_grad():
    for x, _ in DataLoader(ds, batch_size=64, num_workers=0):
        xc = x.to("cuda").to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = model(xc)
        zs_fp.append(z.float().cpu().numpy())
        xs.append(x.numpy())
z_fp = np.concatenate(zs_fp)
x_all = np.concatenate(xs)
chunks = np.array_split(x_all, max(1, len(x_all) // 512))
z_i8 = np.concatenate([sess8.run(None, {in8: b})[0] for b in chunks])

# predicted class per selected patch (index within sel)
pred_fp = z_fp.argmax(1)
pred_i8 = z_i8.argmax(1)
# class of each sel position in CLASS_ORDER terms for assembly
sel_classes = np.array([CLASS_TO_IDX[c] for c in ("TUM", "STR", "LYM", "MUC", "MUS", "NORM", "ADI", "DEB")
                        for _ in by_class[c][:PER_CLASS]])

# ---- assemble N_SLIDES grids with tumor-core/stroma-rim/LYM composition
rng = np.random.default_rng(7)
positions_by_class: dict[int, list[int]] = {}
for pos, cls in enumerate(sel_classes):
    positions_by_class.setdefault(cls, []).append(pos)

COMP = {CLASS_TO_IDX["TUM"]: 18, CLASS_TO_IDX["STR"]: 26, CLASS_TO_IDX["LYM"]: 8}  # rest: other classes
OTHERS = [CLASS_TO_IDX[k] for k in ("MUC", "MUS", "NORM", "ADI", "DEB")]

tsr_drifts, til_drifts, n_tsr, n_til = [], [], 0, 0
tsr_fp32_vals, til_fp32_vals, tsr_i8_vals, til_i8_vals = [], [], [], []
for _s in range(N_SLIDES):
    cells = []
    for cls, n in COMP.items():
        cells += list(rng.choice(positions_by_class[cls], n, replace=True))
    cells += list(rng.choice(positions_by_class[rng.choice(OTHERS)], G * G - sum(COMP.values()), replace=True))
    rng.shuffle(cells)
    g_fp = np.array(CLASS_ORDER, dtype=object)[pred_fp[cells]].reshape(G, G)
    g_i8 = np.array(CLASS_ORDER, dtype=object)[pred_i8[cells]].reshape(G, G)
    tf, ti = compute_tsr(g_fp), compute_til(g_fp)
    t8, t8i = compute_tsr(g_i8), compute_til(g_i8)
    if tf is not None and t8 is not None:
        n_tsr += 1
        tsr_drifts.append(abs(t8.tsr - tf.tsr))
        tsr_fp32_vals.append(tf.tsr)
        tsr_i8_vals.append(t8.tsr)
    if ti is not None and t8i is not None:
        n_til += 1
        til_drifts.append(
            abs(t8i.intratumoral_density_mm2 - ti.intratumoral_density_mm2)
            / max(ti.intratumoral_density_mm2, 1e-9))
        til_fp32_vals.append(ti.intratumoral_density_mm2)
        til_i8_vals.append(t8i.intratumoral_density_mm2)

n_slides = N_SLIDES

out = {
    "int8_path": str(int8_path),
    "int8_size_mb": round(int8_path.stat().st_size / 1e6, 1),
    "n_slides": int(n_slides), "grid": G,
    "argmax_agreement": round(float((z_fp.argmax(1) == z_i8.argmax(1)).mean()), 6),
    "tsr_pairs": n_tsr,
    "tsr_abs_drift_max": round(max(tsr_drifts), 4) if tsr_drifts else None,
    "tsr_abs_drift_mean": round(float(np.mean(tsr_drifts)), 4) if tsr_drifts else None,
    "til_pairs": n_til,
    "til_rel_drift_max": round(max(til_drifts), 4) if til_drifts else None,
    "til_rel_drift_mean": round(float(np.mean(til_drifts)), 4) if til_drifts else None,
    "tsr_fp32_mean": round(float(np.mean(tsr_fp32_vals)), 4) if tsr_fp32_vals else None,
    "tsr_int8_mean": round(float(np.mean(tsr_i8_vals)), 4) if tsr_i8_vals else None,
    "til_fp32_mean": round(float(np.mean(til_fp32_vals)), 4) if til_fp32_vals else None,
    "til_int8_mean": round(float(np.mean(til_i8_vals)), 4) if til_i8_vals else None,
}
(RUN / "int8_drift.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
