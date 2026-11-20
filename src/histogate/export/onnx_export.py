"""ONNX FP32/FP16/INT8 export + latency/VRAM benchmark tables. Spec C.2 / PLAN PR-10.

Laws encoded here:
- The exported graph must be DYNAMIC-BATCH (1..32 without re-export) — the
  serving layer sizes batches at request time.
- bench_session times EACH batch after warmup: a cold first call (JIT/allocator
  warmup) must never be reported as steady-state latency, and throughput is
  batch/mean — never whole-loop-walltime/n_guesswork (PLAN PR-10 RED-b).
- VRAM ceiling is a LAW, not a warning: int8 inference over configs
  gate.vram_max_mb is a hard failure (RED-c).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from histogate.schemas.bench import BenchResult

_INPUT_SIZE = (1, 3, 224, 224)


@dataclass(frozen=True)
class BenchStats:
    p95_latency_ms: float
    mean_latency_ms: float
    patches_per_s: float
    n_batches: int
    batch: int


def export_onnx(model: nn.Module, path: Path | str, precision: str = "fp32") -> Path:
    """Export `model` to ONNX at `path`. precision in {fp32, fp16, int8}.

    fp16 keeps fp32 IO (casting inside the graph) so callers feed fp32 always.
    int8 uses dynamic-range quantization (deterministic; static calib at the
    full-run stage — see run_export_bench).
    """
    path = Path(path)
    model = model.cpu().eval()
    dummy = torch.zeros(_INPUT_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    if precision == "fp32":
        return path
    import onnx

    graph = onnx.load(str(path))
    if precision == "fp16":
        from onnxconverter_common import float16

        graph16 = float16.convert_float_to_float16(graph, keep_io_types=True)
        out = path.with_name(path.stem + "_fp16.onnx")
        onnx.save(graph16, str(out))
        return out
    if precision == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        out = path.with_name(path.stem + "_int8.onnx")
        quantize_dynamic(
            str(path), str(out), weight_type=QuantType.QInt8, per_channel=True
        )
        return out
    raise ValueError(f"unknown precision {precision!r}")


def bench_session(
    sess,
    input_name: str,
    batch: int,
    n_batches: int,
    n_warmup: int = 2,
    shape: tuple = (3, 224, 224),
) -> BenchStats:
    """Time an ORT-like session per batch, after warmup, on a fixed dummy input."""
    if n_warmup < 1:
        raise ValueError("n_warmup >= 1 required: cold call must never be timed")
    x = np.random.default_rng(0).normal(0, 1, (batch, *shape)).astype(np.float32)
    feed = {input_name: x}
    for _ in range(n_warmup):
        sess.run(None, feed)
    lat_ms: list[float] = []
    for _ in range(n_batches):
        t0 = perf_counter()
        sess.run(None, feed)
        lat_ms.append((perf_counter() - t0) * 1e3)
    mean_ms = float(np.mean(lat_ms))
    return BenchStats(
        p95_latency_ms=float(np.percentile(lat_ms, 95)),
        mean_latency_ms=mean_ms,
        patches_per_s=batch / (mean_ms / 1e3),
        n_batches=n_batches,
        batch=batch,
    )


def validate_bench_vram(result: BenchResult, ceiling_mb: float) -> None:
    """Hard-fail a bench result that exceeds the VRAM ceiling (PLAN PR-10 RED-c)."""
    if result.peak_vram_mb > ceiling_mb:
        raise ValueError(
            f"vram ceiling violated: {result.precision} peak "
            f"{result.peak_vram_mb:.0f} MB > {ceiling_mb:.0f} MB"
        )
