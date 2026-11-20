"""PR-13: FastAPI service — PLAN.md PR-13 RED contract.

Adversarial angles: response_model ENFORCES the SlideReport contract (an
endpoint returning garbage fails FastAPI's own validation, not just ours);
bad path -> 404, malformed body -> 422; /health must carry model_revision
AND the RUO disclaimer (a regulated service identifies itself); and the
engine lock must SERIALIZE inference — 8 concurrent requests may never run
the model twice at once (VRAM double-alloc is the failure this catches).
"""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from histogate.schemas.report import SlideReport, TILResult, TSRResult


def _slide_report(slide_id: str = "slide-042") -> SlideReport:
    return SlideReport(
        slide_id=slide_id, n_patches=64, n_gated=3,
        compartment_fractions={"TUM": 0.5, "STR": 0.5},
        artifacts=[],
        tsr=TSRResult(stroma_fraction=0.5, tumor_fraction=0.5, tsr=0.5,
                      group="stroma-low", method="patch-proxy-vanpelt-2018",
                      disclosure="patch-proxy; RUO"),
        til=TILResult(intratumoral_density_mm2=8.0, stromal_density_mm2=20.0),
        calibration_temperature=1.2,
        model_revision="testrev01",
        ruo_disclaimer="Research Use Only. Not for use in diagnostic procedures.",
    )


@pytest.fixture()
def client(tmp_path):
    from histogate.serve.api import create_app
    from histogate.serve.engine import LockedEngine

    good_dir = tmp_path / "slide"
    good_dir.mkdir()
    (good_dir / "a.tif").write_bytes(b"0" * 16)
    engine = LockedEngine(analyze=lambda path: _slide_report(path.name))
    return TestClient(create_app(engine=engine, slides_root=tmp_path)), good_dir


def test_analyze_happy_path_contract_enforced(client) -> None:
    c, good_dir = client
    r = c.post("/analyze", json={"slide_path": str(good_dir)})
    assert r.status_code == 200, r.text
    report = SlideReport.model_validate(r.json())  # response_model already checked; we double
    assert report.n_patches == 64


def test_bad_path_404_and_malformed_422(client) -> None:
    c, _ = client
    r = c.post("/analyze", json={"slide_path": "/nonexistent/slide"})
    assert r.status_code == 404
    r2 = c.post("/analyze", data="not-json", headers={"content-type": "application/json"})
    assert r2.status_code == 422
    r3 = c.post("/analyze", json={})  # missing required field
    assert r3.status_code == 422


def test_health_carries_revision_and_ruo(client) -> None:
    c, _ = client
    h = c.get("/health").json()
    assert h["model_revision"] == "testrev01"
    assert "Research Use Only" in h["ruo_disclaimer"]


def test_lock_serializes_inference_no_double_alloc() -> None:
    """8 threads through the engine; the model fn must NEVER overlap."""
    from histogate.serve.engine import LockedEngine

    state = {"inside": 0, "max_inside": 0}
    guard = threading.Lock()

    def slow_analyze(_path: str) -> SlideReport:
        with guard:
            state["inside"] += 1
            state["max_inside"] = max(state["max_inside"], state["inside"])
        time.sleep(0.01)  # widen the race window
        with guard:
            state["inside"] -= 1
        return _slide_report()

    engine = LockedEngine(analyze=slow_analyze)
    threads = [
        threading.Thread(target=lambda i=i: engine.analyze(f"s{i}"))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max_inside"] == 1, (
        f"inference overlapped {state['max_inside']}x — lock broken, VRAM would double-alloc"
    )
