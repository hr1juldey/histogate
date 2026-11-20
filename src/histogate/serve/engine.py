"""Inference engine wrapper: single model instance, hard serialization.

VRAM law (PLAN PR-13 RED-d): one model, one lock — concurrent requests QUEUE.
Double-allocating the model per request on a 12 GB card is the failure this
file exists to make impossible.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from histogate.schemas.report import SlideReport

AnalyzeFn = Callable[[Path], SlideReport]


class LockedEngine:
    """Wraps an analyze(path) -> SlideReport callable behind a re-entrant lock."""

    def __init__(self, analyze: AnalyzeFn) -> None:
        self._analyze = analyze
        self._lock = threading.Lock()
        self._last: SlideReport | None = None

    def warm(self, probe_path: Path) -> None:
        """Best-effort warmup: run one probe so /health reflects the model."""
        try:
            self.analyze(probe_path)
        except Exception:  # noqa: BLE001 — warmup failure must not kill the app
            self._last = None

    @property
    def revision(self) -> str:
        return self._last.model_revision if self._last else "not-warmed"

    @property
    def ruo(self) -> str:
        if self._last:
            return self._last.ruo_disclaimer
        return "Research Use Only. Not for use in diagnostic procedures."

    def analyze(self, path: Path) -> SlideReport:
        with self._lock:
            report = self._analyze(path)
            self._last = report
            return report
