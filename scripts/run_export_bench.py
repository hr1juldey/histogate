"""Export the trained HistoGate checkpoint to ONNX (fp32/fp16) + bench.

Runs the PR-10 exporter against the REAL overnight checkpoint: parity vs
torch logits, p95 latency, throughput patches/s, VRAM ceiling — all numbers
the regression gate consumes. Writes onnx/export_bench.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from histogate.export.onnx_export import bench_session, export_onnx
from histogate.model import N_CLASSES, PatchClassifier

RUN = Path(os.environ.get("HISTOGATE_RUN", "runs/overnight-30ep"))
CKPT = RUN / "best.ckpt.pt"
OUT = RUN / "onnx"
BATCH = 32
VRAM_CEILING_MB = 2048.0
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 1e6)
    return 0.0


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "checkpoint": str(CKPT), "git_sha": ckpt.get("git_sha"),
        "batch": BATCH, "vram_ceiling_mb": VRAM_CEILING_MB,
    }

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (BATCH, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()

    outputs = {}
    active = None
    for prec in ("fp32", "fp16"):
        path = export_onnx(model, OUT / f"histogate-{prec}.onnx", precision=prec)
        sess = ort.InferenceSession(str(path), providers=PROVIDERS)
        active = active or sess.get_providers()
        outputs[prec] = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        stats = bench_session(sess, sess.get_inputs()[0].name, batch=BATCH,
                              n_batches=100, n_warmup=10)
        vram = peak_vram_mb()
        data = path.with_name(path.name + ".data")
        size = path.stat().st_size + (data.stat().st_size if data.exists() else 0)
        report[prec] = {
            "path": str(path),
            "size_mb": round(size / 1e6, 1),
            "p95_ms": round(stats.p95_latency_ms, 2),
            "throughput_patches_s": round(stats.patches_per_s, 1),
            "peak_vram_mb": round(vram, 1),
            "vram_within_ceiling": vram <= VRAM_CEILING_MB,
        }
        print(prec, report[prec])
    report["execution_providers"] = active

    # serving-stack throughput (LockedEngine runs TORCH on CUDA — the gate's
    # number; ORT above is the CPU portability measurement)
    if torch.cuda.is_available():
        model_cuda = model.to("cuda").to(memory_format=torch.channels_last)
        xt = torch.from_numpy(x).to("cuda")
        with torch.no_grad():
            for _ in range(10):  # warmup — cold call never timed
                model_cuda(xt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                model_cuda(xt)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        report["torch_cuda"] = {
            "batch": BATCH,
            "throughput_patches_s": round(BATCH * 100 / dt, 1),
            "mean_latency_ms": round(dt / 100 * 1e3, 2),
            "peak_vram_mb": round(peak_vram_mb(), 1),
        }
        print("torch_cuda", report["torch_cuda"])

    # parity: fp32 vs torch, fp16 vs fp32 (the gate's quant_agreement inputs)
    report["fp32_argmax_agreement_vs_torch"] = float(
        (outputs["fp32"].argmax(1) == ref.argmax(1)).mean())
    report["fp32_max_abs_delta"] = float(np.abs(outputs["fp32"] - ref).max())
    report["fp16_argmax_agreement_vs_fp32"] = float(
        (outputs["fp16"].argmax(1) == outputs["fp32"].argmax(1)).mean())

    (OUT / "export_bench.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)},
                     indent=2))


if __name__ == "__main__":
    main()
