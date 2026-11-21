"""EXEC ENGINE CLI — the one command a zero-context agent runs.

Protocol: docs/EXEC_ENGINE.md. Catalog: docs/ROADMAP_PR15-20.md.
State: docs/EXEC_STATE.json. Journal: docs/EXEC_LOG.md (append-only).

Modes:
  (default)            print next sub-task: id, spec pointer, commands, PASS
  --done ID --note TXT mark done (captures HEAD sha, artifact, commits state)
  --fail ID --note TXT record failed attempt (3 attempts -> parked)
  --verify              check invariants V1..V5, print repairs if broken
  --init                (re)initialize state from the roadmap ID index
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_P = ROOT / "docs" / "EXEC_STATE.json"
LOG_P = ROOT / "docs" / "EXEC_LOG.md"
ROADMAP_P = ROOT / "docs" / "ROADMAP_PR15-20.md"

# Matches sub-task ids in the ROADMAP completeness index AND spec headings.
ID_RE = re.compile(r"\b(P1[5-9]-S\d)\b")
ORDER = [
    "P15-S0", "P15-S1", "P15-S2", "P15-S3", "P15-S4", "P15-S5", "P15-S6",
    "P16-S1", "P16-S2", "P16-S3", "P16-S4", "P16-S5", "P16-S6",
    "P17-S1", "P17-S2", "P17-S3", "P17-S4",
    "P18-S1", "P18-S2", "P18-S3", "P18-S4",
    "P19-S1", "P19-S2", "P19-S3",
]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(line: str) -> None:
    with LOG_P.open("a") as f:
        f.write(line + "\n")


def _load() -> dict:
    return json.loads(STATE_P.read_text())


def _save(st: dict) -> None:
    st["updated"] = _now()
    STATE_P.write_text(json.dumps(st, indent=2) + "\n")


def roadmap_ids() -> list[str]:
    ids = ID_RE.findall(ROADMAP_P.read_text())
    seen: list[str] = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return seen


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True,
        check=False).stdout.strip()


def init_state() -> dict:
    ids = roadmap_ids()
    missing = [i for i in ORDER if i not in ids]
    if missing:
        sys.exit(f"ROADMAP corrupted: missing ids {missing}; restore index")
    st = {"version": 1, "roadmap": "docs/ROADMAP_PR15-20.md",
          "current_pr": "PR-15",
          "subtasks": {i: {"status": "todo", "attempts": 0,
                           "last_note": "", "commit": None,
                           "artifact": None} for i in ORDER},
          "parked": [], "updated": _now()}
    _save(st)
    return st


def next_id(st: dict) -> str | None:
    for i in ORDER:
        s = st["subtasks"][i]
        if s["status"] == "in_progress":
            return i
    for i in ORDER:
        if st["subtasks"][i]["status"] == "todo":
            return i
    return None


def spec_text(tid: str) -> str:
    """Return the ROADMAP spec block for tid (heading to next blank-line
    heading of the same kind), so the executor never guesses."""
    lines = ROADMAP_P.read_text().splitlines()
    start = next((k for k, ln in enumerate(lines)
                  if ln.startswith(tid) or f" {tid}\n" in ln + "\n"
                  or re.match(rf"^{tid}\b", ln)), None)
    if start is None:
        return f"SPEC MISSING for {tid} — restore ROADMAP before executing."
    end = start + 1
    while end < len(lines) and not re.match(r"^(P1[5-9]-S\d|═{10,})", lines[end]):
        end += 1
    return "\n".join(lines[start:end]).strip()


def cmd_next(st: dict) -> int:
    tid = next_id(st)
    if tid is None:
        print("ALL SUB-TASKS DONE/PARKED. Parked:", st["parked"] or "none")
        return 0
    st["subtasks"][tid]["status"] = "in_progress"
    st["subtasks"][tid]["attempts"] += 1
    _save(st)
    print(f"NEXT: {tid} (attempt {st['subtasks'][tid]['attempts']})")
    print("SPEC:\n" + spec_text(tid))
    print("\nOn PASS: git add -A && git commit -m '" + tid + " <line>' "
          f"then: uv run python scripts/next_step.py --done {tid} --note '<measured result>'")
    print(f"On FAIL: uv run python scripts/next_step.py --fail {tid} --note '<error>'")
    return 0


def cmd_done(st: dict, tid: str, note: str) -> int:
    if tid not in st["subtasks"]:
        sys.exit(f"unknown id {tid}")
    sha = _git("rev-parse", "--short", "HEAD")
    st["subtasks"][tid].update(status="done", commit=sha, last_note=note)
    if tid in st["parked"]:
        st["parked"].remove(tid)
    _save(st)
    _log(f"{_now()} | {tid} | done | attempt {st['subtasks'][tid]['attempts']} | {note} | {sha}")
    _git("add", "docs/EXEC_STATE.json", "docs/EXEC_LOG.md")
    _git("commit", "-q", "-m", f"exec: {tid} done")
    nxt = next_id(st)
    print(f"{tid} DONE @{sha}. NEXT: {nxt}")
    return 0


def cmd_fail(st: dict, tid: str, note: str) -> int:
    if tid not in st["subtasks"]:
        sys.exit(f"unknown id {tid}")
    s = st["subtasks"][tid]
    if note.startswith("PARKED:"):
        s.update(status="parked")
        st["parked"].append(tid)
    elif s["attempts"] >= 3:
        s.update(status="parked")
        st["parked"].append(tid)
        note = f"PARKED after {s['attempts']} attempts: {note}"
    else:
        s.update(status="todo")
    s["last_note"] = note
    _save(st)
    _log(f"{_now()} | {tid} | {s['status']} | attempt {s['attempts']} | {note}")
    _git("add", "docs/EXEC_STATE.json", "docs/EXEC_LOG.md")
    _git("commit", "-q", "-m", f"exec: {tid} {s['status']}")
    print(f"{tid} -> {s['status'].upper()}. attempts={s['attempts']}")
    return 0


def cmd_verify(st: dict) -> int:
    """V1..V5 from EXEC_ENGINE.md. Print exact repairs; exit 1 on any."""
    errs: list[str] = []
    log = _git("log", "--oneline", "-50")
    for tid, s in st["subtasks"].items():
        if s["commit"] and s["commit"] not in log and s["status"] == "done":
            errs.append(f"V1 {tid}: commit {s['commit']} not in git log; "
                        f"repair: --done {tid} --note 'resha' after commit")
        artifact_gone = (s["status"] == "done" and s["artifact"]
                         and s["artifact"] != "docs"
                         and not (ROOT / s["artifact"]).exists())
        if artifact_gone:
            errs.append(f"V3 {tid}: artifact {s['artifact']} missing; "
                        f"repair: --fail {tid} --note 'artifact lost'")
        if s["attempts"] > 3 and s["status"] != "parked":
            errs.append(f"V3 {tid}: attempts={s['attempts']} >3 not parked")
    orphans = [t for t in st["subtasks"] if t not in roadmap_ids()]
    if orphans:
        errs.append(f"V2 orphan ids {orphans}; repair: remove from state")
    missing = [t for t in roadmap_ids() if t in ORDER and t not in st["subtasks"]]
    if missing:
        errs.append(f"V2 missing ids {missing}; repair: --init (loses notes)")
    for p in st["parked"]:
        if st["subtasks"].get(p, {}).get("status") != "parked":
            errs.append(f"V4/V5 {p}: parked-list/status mismatch")
    if errs:
        print("VERIFY FAIL:\n" + "\n".join(errs))
        return 1
    print("VERIFY OK: V1..V5 clean")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--done")
    ap.add_argument("--fail")
    ap.add_argument("--note", default="")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--init", action="store_true")
    a = ap.parse_args()
    st = init_state() if a.init else _load()
    if a.verify:
        return cmd_verify(st)
    if a.done:
        return cmd_done(st, a.done, a.note)
    if a.fail:
        return cmd_fail(st, a.fail, a.note)
    return cmd_next(st)


if __name__ == "__main__":
    raise SystemExit(main())
