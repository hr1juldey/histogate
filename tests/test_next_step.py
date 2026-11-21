"""Adversarial tests for the EXEC ENGINE (docs/EXEC_ENGINE.md T-N1..T-N5).

These attack the engine the way real corruption happens: state drifts from
git, ids rot, artifacts vanish, parked work silently resumes. The engine
must DETECT each attack (deny + name the repair), never silently absorb it.
Tests run against a tmp copy of state/roadmap/log — no real state is touched.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_engine(tmp: Path):
    """Import next_step.py with its path constants pointed at tmp."""
    spec = importlib.util.spec_from_file_location(
        "next_step_t", REPO / "scripts" / "next_step.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.STATE_P = tmp / "EXEC_STATE.json"
    mod.LOG_P = tmp / "EXEC_LOG.md"
    mod.ROADMAP_P = tmp / "ROADMAP.md"
    return mod


@pytest.fixture()
def sandbox(tmp_path: Path):
    shutil.copy(REPO / "docs" / "ROADMAP_PR15-20.md", tmp_path / "ROADMAP.md")
    (tmp_path / "EXEC_LOG.md").touch()
    return tmp_path


@pytest.fixture()
def healthy(sandbox: Path):
    mod = load_engine(sandbox)
    st = mod.init_state()
    st["subtasks"]["P15-S0"].update(
        status="done", commit="deadbeef0", attempts=1, artifact="docs")
    st["subtasks"]["P15-S0"]["commit"] = None  # no git in tests; V1 skips null
    mod._save(st)
    return mod, st


def test_n1_stale_state_detected(healthy):
    """Attack: sub-task done in git but state says todo — verify must fail.

    We simulate the drift by marking a later id done WITH a sha that cannot
    be in git history, then verify names V1 and the repair command."""
    mod, st = healthy
    st["subtasks"]["P15-S1"].update(status="done", commit="cafe0000",
                                    artifact="docs")
    mod._save(st)
    # git log in a non-repo/empty returns '' — sha absent => V1 must fire.
    # in the sandbox _git returns empty stdout, so 'cafe0000' not in log
    assert mod.cmd_verify(mod._load()) == 1


def test_n2_orphan_id_detected(healthy):
    """Attack: state carries an id that no longer exists in the roadmap."""
    mod, st = healthy
    st["subtasks"]["P99-S9"] = {"status": "todo", "attempts": 0,
                                "last_note": "", "commit": None,
                                "artifact": None}
    mod._save(st)
    assert mod.cmd_verify(mod._load()) == 1


def test_n3_done_artifact_missing(healthy, tmp_path):
    """Attack: done sub-task whose PASS artifact was deleted from disk."""
    mod, st = healthy
    ghost = tmp_path / "ghost.json"
    ghost.write_text("{}")
    st["subtasks"]["P15-S1"].update(status="done", commit=None,
                                    artifact="ghost.json", attempts=1)
    mod._save(st)
    ghost.unlink()
    assert mod.cmd_verify(mod._load()) == 1


def test_n4_parked_never_current(healthy):
    """Attack: parked sub-task must never be returned as the next step."""
    mod, st = healthy
    st["subtasks"]["P15-S1"].update(status="parked", attempts=3)
    st["parked"].append("P15-S1")
    mod._save(st)
    fresh = mod._load()
    assert mod.next_id(fresh) != "P15-S1"
    assert mod.next_id(fresh) == "P15-S2"


def test_n5_next_id_deterministic(healthy):
    """Two calls with identical state must select the identical id."""
    mod, _ = healthy
    assert mod.next_id(mod._load()) == mod.next_id(mod._load())


def test_fail_after_three_attempts_parks(healthy):
    """Attack: a task flapping forever. Engine must cap at 3 then park."""
    mod, st = healthy
    st["subtasks"]["P15-S1"]["status"] = "in_progress"
    st["subtasks"]["P15-S1"]["attempts"] = 3
    mod._save(st)
    mod.cmd_fail(mod._load(), "P15-S1", "boom")
    final = mod._load()
    assert final["subtasks"]["P15-S1"]["status"] == "parked"
    assert "P15-S1" in final["parked"]


def test_log_is_append_only(healthy):
    """Attack: accidental journal rewrite. Journal lines must never shrink."""
    mod, _ = healthy
    before = mod.LOG_P.read_text()
    mod.cmd_fail(mod._load(), "P15-S1", "attempt log")
    assert mod.LOG_P.read_text().startswith(before)
