"""Pydantic v2 contracts: one schema language for config, artifacts, API and MCP."""

from histogate.schemas.bench import BenchResult
from histogate.schemas.gate import GateCheck, GateReport
from histogate.schemas.report import (
    ArtifactFinding,
    ArtifactKind,
    PatchPrediction,
    SlideReport,
    TILResult,
    TissueClass,
    TSRResult,
)

__all__ = [
    "ArtifactFinding",
    "ArtifactKind",
    "BenchResult",
    "GateCheck",
    "GateReport",
    "PatchPrediction",
    "SlideReport",
    "TILResult",
    "TissueClass",
    "TSRResult",
]
