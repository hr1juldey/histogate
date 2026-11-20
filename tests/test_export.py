"""PR-10: ONNX export + bench — PLAN.md PR-10 RED contract.

Adversarial angles:
(a) PARITY: exported graph must agree with the torch model it came from —
    argmax agreement AND bounded logit delta per precision. An exporter that
    silently exports the wrong graph (wrong input size, wrong normalization
    baked in, dropped head) fails here.
(b) TIMER ORACLE: a stub session with a deliberately SLOW COLD call proves
    warmup excludes it and timing is per-batch — a naive whole-loop timer or
    a no-warmup bench reports the cold call as steady-state and must FAIL.
(c) VRAM CEILING: the bench validator must reject an int8 result over the
    2048 MB law (configs gate.vram_max_mb) — fail, not warn.
(d) DYNAMIC BATCH: the exported graph runs batch 1..32 without re-export.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from histogate.export.onnx_export import (
    bench_session,
    export_onnx,
    validate_bench_vram,
)
from histogate.model import PatchClassifier

N_UNIT = 32  # CPU-friendly parity n


def _random_model_and_inputs(n: int, seed: int = 0) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    model = PatchClassifier(backbone="convnext_tiny", pretrained=False).eval()
    x = torch.from_numpy(
        np.random.default_rng(seed).normal(0, 1, (n, 3, 224, 224)).astype(np.float32)
    )
    with torch.no_grad():
        logits = model(x).numpy()
    return model, x.numpy(), logits


class _SleepySession:
    """Stub ORT session: first run() costs 200ms (cold), rest 5ms (steady)."""

    def __init__(self, cold_s: float = 0.2, steady_s: float = 0.005) -> None:
        self.calls = 0
        self.cold_s = cold_s
        self.steady_s = steady_s

    def run(self, _names: list[str], feed: dict) -> list[np.ndarray]:
        self.calls += 1
        time.sleep(self.cold_s if self.calls == 1 else self.steady_s)
        batch = next(iter(feed.values())).shape[0]
        return [np.zeros((batch, 9), np.float32)]


def test_parity_fp32(tmp_path: Path) -> None:
    model, x, torch_logits = _random_model_and_inputs(N_UNIT)
    onnx_path = export_onnx(model, tmp_path / "m32.onnx", precision="fp32")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {"input": x})[0]
    agree = float((onnx_logits.argmax(1) == torch_logits.argmax(1)).mean())
    assert agree >= 0.999, f"argmax agreement only {agree:.4f}"
    assert float(np.abs(onnx_logits - torch_logits).max()) <= 1e-2


def test_parity_fp16_tolerance(tmp_path: Path) -> None:
    model, x, torch_logits = _random_model_and_inputs(N_UNIT, seed=1)
    onnx_path = export_onnx(model, tmp_path / "m16.onnx", precision="fp16")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {"input": x})[0]
    agree = float((onnx_logits.argmax(1) == torch_logits.argmax(1)).mean())
    assert agree >= 0.999, f"argmax agreement only {agree:.4f}"
    assert float(np.abs(onnx_logits - torch_logits).max()) <= 5e-2


def test_timer_oracle_warmup_excludes_cold_call() -> None:
    """p95 must reflect steady state (5ms), NOT the 200ms cold call —
    and throughput must be per-batch, not whole-loop-divided-by-guesswork."""
    sess = _SleepySession()
    stats = bench_session(sess, input_name="input", batch=16, n_batches=8, n_warmup=2)
    assert sess.calls == 10, f"expected 2 warmup + 8 timed calls, got {sess.calls}"
    assert stats.p95_latency_ms < 50.0, (
        f"p95 {stats.p95_latency_ms:.1f}ms includes the cold call — warmup broken"
    )
    assert stats.p95_latency_ms > 4.0, "p95 below the stub's steady cost — timer not timing"
    expected_tput = 16 / 0.005
    assert 0.5 * expected_tput <= stats.patches_per_s <= 2.0 * expected_tput, (
        f"throughput {stats.patches_per_s:.0f} far from stub steady-state {expected_tput:.0f}"
    )


def test_vram_ceiling_rejects_int8_over_2048mb() -> None:
    from histogate.schemas.bench import BenchResult

    over = BenchResult(
        precision="int8", macro_f1=0.94, ece=0.03,
        patches_per_s=100.0, p95_latency_ms=10.0, peak_vram_mb=2500.0,
    )
    with pytest.raises(ValueError, match="vram"):
        validate_bench_vram(over, ceiling_mb=2048.0)
    ok = BenchResult(
        precision="int8", macro_f1=0.94, ece=0.03,
        patches_per_s=100.0, p95_latency_ms=10.0, peak_vram_mb=1500.0,
    )
    validate_bench_vram(ok, ceiling_mb=2048.0)  # must not raise


def test_dynamic_batch_1_to_32(tmp_path: Path) -> None:
    model, _, _ = _random_model_and_inputs(2, seed=2)
    onnx_path = export_onnx(model, tmp_path / "dyn.onnx", precision="fp32")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(3)
    for batch in (1, 3, 8, 32):
        out = sess.run(None, {"input": rng.normal(0, 1, (batch, 3, 224, 224)).astype(np.float32)})[0]
        assert out.shape == (batch, 9), f"batch {batch} -> shape {out.shape}"
        assert np.isfinite(out).all()
