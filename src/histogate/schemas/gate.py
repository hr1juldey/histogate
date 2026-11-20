"""Gate contracts — LLD §2. A gate report claiming PASS with a failed check is invalid."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    threshold: float
    comparator: Literal[">=", "<=", ">", "<"]
    passed: bool
    detail: str

    @model_validator(mode="after")
    def _passed_matches_comparator(self) -> GateCheck:
        ok = {
            ">=": self.value >= self.threshold,
            "<=": self.value <= self.threshold,
            ">": self.value > self.threshold,
            "<": self.value < self.threshold,
        }[self.comparator]
        if ok != self.passed:
            raise ValueError(
                f"check {self.name}: value {self.value} {self.comparator} {self.threshold} "
                f"evaluates to {ok}, but passed={self.passed}"
            )
        return self


class GateInputs(BaseModel):
    """Everything the gate consumes, from eval + bench artifacts. Fail-closed:
    every field is REQUIRED — a missing measurement is a missing file, not a 0."""

    model_config = ConfigDict(extra="forbid")

    macro_f1: float = Field(ge=0.0, le=1.0)
    macro_f1_domain: float = Field(ge=0.0, le=1.0)  # NONORM scanner domain
    ece: float = Field(ge=0.0, le=1.0)
    ood_auroc: float = Field(ge=0.0, le=1.0)
    tsr_fp32: float = Field(ge=0.0, le=1.0)
    tsr_int8: float = Field(ge=0.0, le=1.0)
    til_fp32: float = Field(ge=0.0)
    til_int8: float = Field(ge=0.0)
    quant_agreement: float = Field(ge=0.0, le=1.0)
    patches_per_s: float = Field(gt=0.0)
    peak_vram_mb: float = Field(ge=0.0)


class GateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_pass: bool
    checks: list[GateCheck] = Field(min_length=1)
    model_revision: str
    dataset_revision: str
    timestamp: str

    @model_validator(mode="after")
    def _overall_consistent(self) -> GateReport:
        all_passed = all(c.passed for c in self.checks)
        if self.overall_pass != all_passed:
            raise ValueError(
                f"overall_pass={self.overall_pass} contradicts checks "
                f"({sum(c.passed for c in self.checks)}/{len(self.checks)} passed)"
            )
        return self
