"""Single source of thresholds and hyperparameters (LLD §1).

Env overrides via HISTOGATE_ prefix with __ nesting, YAML via configs/.
Gate thresholds mirror docs/TESTING.md §3 exactly — no constants live anywhere else.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Paths(BaseModel):
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    benchmarks_dir: Path = Path("benchmarks")


class TrainSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch: int = Field(default=256, ge=1)
    epochs: int = Field(default=10, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    label_smoothing: float = Field(default=0.1, ge=0, le=0.5)
    amp: str = Field(default="bf16", pattern="^(bf16|fp16|fp32)$")
    ema: float = Field(default=0.999, ge=0, lt=1)
    workers: int = Field(default=8, ge=0)
    seed: int = Field(default=42, ge=0)
    backbone: str = "convnext_tiny"
    val_holdout: float = Field(default=0.08, gt=0, lt=0.5)  # of train patients


class SSLSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False  # PR-16 optional phase — off on the critical path
    epochs: int = Field(default=15, ge=1)
    batch: int = Field(default=128, ge=1)
    proj_dim: int = Field(default=256, ge=32)
    temperature: float = Field(default=0.5, gt=0, lt=2)
    stain_jitter: float = Field(default=0.2, ge=0, le=1)


class GateSettings(BaseModel):
    """Thresholds are law: docs/TESTING.md §3. Gate code reads ONLY these."""

    model_config = ConfigDict(extra="forbid")

    f1_min: float = Field(default=0.93, gt=0.5, lt=1.0)
    f1_drop_domain_max: float = Field(default=0.015, gt=0, le=0.2)
    ece_max: float = Field(default=0.05, gt=0, le=1)
    ood_auroc_min: float = Field(default=0.90, gt=0.5, le=1)
    tsr_drift_max: float = Field(default=0.05, gt=0, le=0.2)
    til_drift_rel_max: float = Field(default=0.10, gt=0, le=1)
    quant_agreement_min: float = Field(default=0.999, gt=0.5, le=1)
    throughput_min: float = Field(default=40.0, gt=0)
    vram_max_mb: float = Field(default=2048.0, gt=0)

    @model_validator(mode="after")
    def _consistency(self) -> GateSettings:
        # A quantized model must reproduce its own accuracy floor: the agreement
        # floor must sit strictly above the F1 floor, else the gate is vacuous.
        if self.quant_agreement_min <= self.f1_min:
            raise ValueError(
                f"quant_agreement_min ({self.quant_agreement_min}) must be strictly "
                f"greater than f1_min ({self.f1_min})"
            )
        return self


class ServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    precision: str = Field(default="fp16", pattern="^(fp32|fp16|int8)$")
    max_batch: int = Field(default=32, ge=1)


class MLflowSettings(BaseModel):
    """MLflow is THE tracker (user directive 2026-11-20); runs/ JSON is secondary."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    experiment: str = "histogate"
    system_metrics: bool = True  # GPU util/VRAM, CPU/RAM per run (needs psutil)
    # MLflow 3.x: file-store backend is maintenance-mode (hard exception unless
    # MLFLOW_ALLOW_FILE_STORE=true) — sqlite is the sanctioned local backend.
    tracking_uri: str = "sqlite:///mlflow.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HISTOGATE_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    paths: Paths = Paths()
    train: TrainSettings = TrainSettings()
    ssl: SSLSettings = SSLSettings()
    gate: GateSettings = GateSettings()
    serve: ServeSettings = ServeSettings()
    mlflow: MLflowSettings = MLflowSettings()
    dataset_revision: str = "master"

    def model_post_init(self, __context: object) -> None:
        # Cross-block consistency, run after nested models validated.
        if self.ssl.enabled and self.ssl.batch > self.train.batch * 2:
            raise ValueError("SSL batch unreasonably larger than train batch")


def load_settings(yaml_path: Path | str | None = None) -> Settings:
    """Load Settings from YAML (if given) with env overrides still applied."""
    data: dict = {}
    if yaml_path is not None:
        p = Path(yaml_path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        data = yaml.safe_load(p.read_text()) or {}
    return Settings(**data)
