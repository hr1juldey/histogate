"""Report contracts — LLD §2. Every public JSON validates through these models only."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TissueClass(StrEnum):
    """Kather 2019 nine tissue compartments (NCT-CRC-HE-100K)."""

    ADI = "ADI"
    BACK = "BACK"
    DEB = "DEB"
    LYM = "LYM"
    MUC = "MUC"
    MUS = "MUS"
    NORM = "NORM"
    STR = "STR"
    TUM = "TUM"


class ArtifactKind(StrEnum):
    """AIRAQc's published 8-class artifact taxonomy (5 processing + 3 scanning)."""

    TISSUE_FOLD = "tissue_fold"        # processing
    PEN_MARK = "pen_mark"              # processing
    AIR_BUBBLE = "air_bubble"          # processing
    COVERSLIP_LINE = "coverslip_line"  # processing
    KNIFE_LINE = "knife_line"          # processing
    OUT_OF_FOCUS = "out_of_focus"      # scanning
    NO_TISSUE = "no_tissue"            # scanning
    MISSING_TISSUE = "missing_tissue"  # scanning


class PatchPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tissue: TissueClass
    prob: float = Field(ge=0.0, le=1.0)
    energy: float
    temperature: float = Field(gt=0.0)
    is_ood: bool
    artifact: ArtifactKind | None = None


class ArtifactFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ArtifactKind
    stage: Literal["processing", "scanning"]
    pixel_fraction: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _stage_matches_kind(self) -> ArtifactFinding:
        processing = {
            ArtifactKind.TISSUE_FOLD, ArtifactKind.PEN_MARK, ArtifactKind.AIR_BUBBLE,
            ArtifactKind.COVERSLIP_LINE, ArtifactKind.KNIFE_LINE,
        }
        if (self.kind in processing) != (self.stage == "processing"):
            raise ValueError(f"artifact kind {self.kind.value} does not belong to stage {self.stage}")
        return self


class TSRResult(BaseModel):
    """Tumor-Stroma Ratio. Patch-proxy of the van Pelt 2018 protocol — ALWAYS disclosed."""

    model_config = ConfigDict(extra="forbid")

    stroma_fraction: float = Field(ge=0.0, le=1.0)
    tumor_fraction: float = Field(ge=0.0, le=1.0)
    tsr: float = Field(ge=0.0, le=1.0)
    group: Literal["stroma-high", "stroma-low"]
    method: Literal["patch-proxy-vanpelt-2018"]
    disclosure: str = Field(min_length=1)

    @model_validator(mode="after")
    def _group_matches_cut(self) -> TSRResult:
        expected = "stroma-high" if self.stroma_fraction > 0.50 else "stroma-low"
        if self.group != expected:
            raise ValueError(
                f"stroma_fraction {self.stroma_fraction} implies group {expected}, got {self.group}"
            )
        return self


class TILResult(BaseModel):
    """Tumor-infiltrating lymphocyte densities, Saltz-2018 compartment style."""

    model_config = ConfigDict(extra="forbid")

    intratumoral_density_mm2: float = Field(ge=0.0)
    stromal_density_mm2: float = Field(ge=0.0)
    mpp: float = 0.5
    method: Literal["compartment-saltz-2018-patch-proxy"] = "compartment-saltz-2018-patch-proxy"


class SlideReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slide_id: str
    n_patches: int = Field(ge=1)
    n_gated: int = Field(ge=0)
    compartment_fractions: dict[TissueClass, float]
    artifacts: list[ArtifactFinding]
    tsr: TSRResult | None = None
    til: TILResult | None = None
    slide_label: str | None = None
    slide_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_temperature: float = Field(gt=0.0)
    model_revision: str
    ruo_disclaimer: Literal[
        "Research Use Only. Not for use in diagnostic procedures."
    ]

    @model_validator(mode="after")
    def _fractions_valid(self) -> SlideReport:
        if self.n_gated > self.n_patches:
            raise ValueError("n_gated cannot exceed n_patches")
        total = sum(self.compartment_fractions.values())
        if total > 1.0 + 1e-6:
            raise ValueError(f"compartment fractions sum to {total:.4f} > 1")
        return self
