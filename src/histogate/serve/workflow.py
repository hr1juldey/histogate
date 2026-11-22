"""Guided analysis workflow — one step at a time, guardrails enforced.

The CTX-gateway pattern scoped to two capabilities (analysis, report
generation): a session moves upload -> analyze -> review -> report and CANNOT
skip steps. State is a small JSON on disk so any streamable-HTTP worker can
resume. Images never enter the LLM context — they arrive via the web portal.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_RUO = "Research Use Only. Not for use in diagnostic procedures."
STEPS = ("upload", "analyze", "review", "report")

_STEP_HELP = {
    "upload": "open the portal URL in a browser and upload H&E image(s)",
    "analyze": "call advance to run the analysis on the uploaded files",
    "review": "call advance to accept the flagged-tile review",
    "report": "done — call histogate_report to fetch the report",
}


@dataclass
class StepOut:
    session_id: str
    step: str                      # the step just completed
    next_step: str | None          # what the agent does next
    what_you_need: str             # plain-language instruction
    result: dict = field(default_factory=dict)
    ruo: str = _RUO

    def to_json(self) -> str:
        return json.dumps({
            "session_id": self.session_id, "step": self.step,
            "next_step": self.next_step, "what_you_need": self.what_you_need,
            "result": self.result, "ruo": self.ruo})


class Workflow:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        if not sid.isalnum():
            raise ValueError("bad session id")
        return self.root / f"{sid}.json"

    def start(self) -> str:
        sid = uuid.uuid4().hex[:12]
        self._path(sid).write_text(json.dumps(
            {"step": "upload", "created": time.time(), "files": [], "flagged": []}))
        return sid

    def state(self, sid: str) -> dict:
        p = self._path(sid)
        if not p.exists():
            raise ValueError(f"unknown session {sid}")
        return json.loads(p.read_text())

    def _write(self, sid: str, st: dict) -> None:
        self._path(sid).write_text(json.dumps(st))

    def add_files(self, sid: str, files: list[str]) -> None:
        if len(files) > 16:
            raise ValueError("guardrail: at most 16 files per session")
        st = self.state(sid)
        st["files"] = files[:16]
        self._write(sid, st)

    def advance(self, sid: str, payload: dict | None = None) -> StepOut:
        st = self.state(sid)
        cur = st["step"]
        idx = STEPS.index(cur)
        if cur == "upload":
            files = st.get("files") or []
            if not files:
                raise ValueError(
                    "guardrail: upload at least one image via the portal first")
        if cur == "analyze" and not st.get("files"):
            raise ValueError("guardrail: nothing uploaded")
        nxt = STEPS[min(idx + 1, len(STEPS) - 1)]
        st["step"] = nxt
        self._write(sid, st)
        return StepOut(sid, cur,
                       None if nxt == cur else nxt,
                       _STEP_HELP[cur],
                       {"state": {"step": nxt, "files": st["files"]}})

    def needs(self, sid: str) -> str:
        return _STEP_HELP[self.state(sid)["step"]]
