"""Guided workflow guardrails: one step at a time, no skipping, RUO on
every payload, fresh session per start."""
from __future__ import annotations

from pathlib import Path

import pytest

from histogate.serve.workflow import STEPS, Workflow


@pytest.fixture()
def wf(tmp_path: Path) -> Workflow:
    return Workflow(tmp_path / "wf")


def test_step_order_and_no_skip(wf: Workflow) -> None:
    sid = wf.start()
    assert wf.state(sid)["step"] == "upload"
    with pytest.raises(ValueError, match="guardrail"):
        wf.advance(sid)                      # nothing uploaded -> blocked
    wf.add_files(sid, ["a.tif"])
    seen = []
    for _ in range(len(STEPS)):
        out = wf.advance(sid)
        seen.append(out.step)
    assert seen == list(STEPS)
    assert wf.state(sid)["step"] == "report"


def test_ruo_on_every_payload(wf: Workflow) -> None:
    sid = wf.start()
    wf.add_files(sid, ["a.tif"])
    out = wf.advance(sid)
    assert "Research Use Only" in out.ruo
    assert out.next_step == "analyze"
    assert out.what_you_need  # plain-language guidance every step


def test_fresh_session_and_unknown(wf: Workflow) -> None:
    a, b = wf.start(), wf.start()
    assert a != b
    with pytest.raises(ValueError, match="unknown session"):
        wf.state("nope123")
    with pytest.raises(ValueError, match="bad session id"):
        wf.state("../evil")


def test_file_limit(wf: Workflow) -> None:
    sid = wf.start()
    with pytest.raises(ValueError, match="guardrail"):
        wf.add_files(sid, [f"{i}.tif" for i in range(17)])
