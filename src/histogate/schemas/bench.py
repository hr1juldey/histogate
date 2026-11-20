"""Benchmark contracts — LLD §2."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BenchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: Literal["fp32", "fp16", "int8"]
    macro_f1: float = Field(ge=0.0, le=1.0)
    ece: float = Field(ge=0.0, le=1.0)
    patches_per_s: float = Field(gt=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    peak_vram_mb: float = Field(ge=0.0)
