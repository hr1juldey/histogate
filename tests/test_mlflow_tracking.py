"""PR-9(1/2): MLflow tracking — every training run is queryable live.

Anti-tautology: the oracle is MLflow's OWN store API (search_runs / get_run),
never the tracker's return values. A metric that cannot be read back from the
tracking store does not exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from histogate.tracking import MlflowTracker  # noqa: E402


def _uri(p: Path) -> str:
    return f"sqlite:///{p / 'mlflow.db'}"


def test_run_is_queryable_from_the_store(tmp_path: Path) -> None:
    """Params/epoch-metrics/final-metrics must round-trip through mlflow itself."""
    t = MlflowTracker(experiment="ut-histogate", tracking_uri=_uri(tmp_path))
    with t.run("ut-run-1"):
        t.log_params({"seed": 42, "lr": 1e-3, "git_sha": "abc1234"})
        t.log_epoch(0, {"train_loss": 1.7, "val_macro_f1": 0.10})
        t.log_epoch(1, {"train_loss": 1.1, "val_macro_f1": 0.34})
        t.log_final({"final": {"val_macro_f1": 0.34}, "peak_vram_mb": 8010.4})
        t.log_artifacts({"metrics.json": Path(t.cur_metrics_path or "")} if t.cur_metrics_path else {})

    client = mlflow.tracking.MlflowClient(tracking_uri=_uri(tmp_path))
    exp = client.get_experiment_by_name("ut-histogate")
    assert exp is not None, "experiment not created in the store"
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1, f"expected exactly 1 run, found {len(runs)}"
    r = runs[0]
    assert r.data.params["seed"] == "42"
    assert r.data.params["git_sha"] == "abc1234"
    m = r.data.metrics
    assert m["train_loss"] == pytest.approx(1.1), "latest epoch value must win"
    assert m["val_macro_f1"] == pytest.approx(0.34)
    assert m["peak_vram_mb"] == pytest.approx(8010.4)
    # epoch-indexed history must be present (live chart-ability)
    hist = client.get_metric_history(r.info.run_id, "val_macro_f1")
    assert [h.value for h in hist] == pytest.approx([0.10, 0.34])


def test_disabled_tracker_is_a_no_op_not_a_crash(tmp_path: Path) -> None:
    t = MlflowTracker(experiment="ut-off", tracking_uri=_uri(tmp_path), enabled=False)
    with t.run("never-logged"):
        t.log_params({"seed": 1})
        t.log_epoch(0, {"train_loss": 0.5})
        t.log_final({"final": {}})
    client = mlflow.tracking.MlflowClient(tracking_uri=_uri(tmp_path))
    exp = client.get_experiment_by_name("ut-off")
    assert exp is None or not client.search_runs([exp.experiment_id]), \
        "disabled tracker must leave no runs in the store"


def test_metrics_json_has_tracking_fields(tmp_path: Path) -> None:
    """The runs/ JSON must record WHERE the mlflow run lives (cross-reference law)."""
    t = MlflowTracker(experiment="ut-x", tracking_uri=_uri(tmp_path))
    with t.run("ut-run-2"):
        t.log_params({"seed": 7})
        t.log_epoch(0, {"train_loss": 1.0})
        payload = t.annotate({"final": {}, "peak_vram_mb": 1.0})
    assert payload["mlflow"]["experiment"] == "ut-x"
    assert len(payload["mlflow"]["run_id"]) > 8
    json.dumps(payload)  # must stay JSON-serializable
