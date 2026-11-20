"""MLflow tracking for every HistoGate training run (PR-9 part 1).

Doctrine (user directive 2026-11-20): the local runs/ JSON is a secondary
artifact; MLflow is THE tracker — params, per-epoch metric history (live
charts), final metrics, checkpoint + metrics.json artifacts, all queryable
from `mlflow ui` while the run is still going.

The oracle for correctness is MLflow's own store API, never this module's
returns (see tests/test_mlflow_tracking.py).
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow


class MlflowTracker:
    """Thin, fail-loud wrapper. `enabled=False` makes every method a no-op."""

    def __init__(self, experiment: str, tracking_uri: str, enabled: bool = True) -> None:
        self.experiment = experiment
        self.tracking_uri = tracking_uri
        self.enabled = enabled
        self.run_id: str | None = None
        self.cur_metrics_path: str | None = None
        self._active = False

    # -- lifecycle ---------------------------------------------------------
    @contextlib.contextmanager
    def run(self, name: str) -> Iterator[MlflowTracker]:
        if not self.enabled:
            self._active = True
            try:
                yield self
            finally:
                self._active = False
                self.run_id = None
            return
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        with mlflow.start_run(run_name=name) as active:
            self.run_id = active.info.run_id
            self._active = True
            try:
                yield self
            finally:
                self._active = False

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("tracker method called outside .run() context")

    # -- logging -----------------------------------------------------------
    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._require_active()
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Epoch-indexed history — this is what makes live charts work."""
        if not self.enabled:
            return
        self._require_active()
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=epoch)

    def log_final(self, payload: dict[str, Any]) -> None:
        """Log run-level scalars ONLY (e.g. peak_vram_mb). The `final` dict's
        values are already in the epoch history — re-logging them would append
        a step-less point AFTER the last epoch and corrupt live charts
        (caught by the metric-history oracle in test_run_is_queryable...)."""
        if not self.enabled:
            return
        self._require_active()
        flat = {
            k: float(v) for k, v in payload.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if flat:
            mlflow.log_metrics(flat)

    def log_artifacts(self, paths: dict[str, Path]) -> None:
        if not self.enabled:
            return
        self._require_active()
        for p in paths.values():
            if p and Path(p).exists():
                mlflow.log_artifact(str(p))

    # -- cross-reference -----------------------------------------------------
    def annotate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Stamp the runs/ JSON with where the mlflow run lives."""
        payload["mlflow"] = {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "tracking_uri": self.tracking_uri,
            "enabled": self.enabled,
        }
        return payload
