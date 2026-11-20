"""FastAPI surface — PLAN PR-13. response_model is the contract; 404/422 the
negative paths; /health identifies the regulated service (revision + RUO)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from histogate.schemas.report import SlideReport
from histogate.serve.engine import LockedEngine


class AnalyzeRequest(BaseModel):
    slide_path: str


def create_app(engine: LockedEngine, slides_root: Path) -> FastAPI:
    app = FastAPI(title="HistoGate", version="0.1.0")
    engine.warm(slides_root)

    def _resolve(slide_path: str) -> Path:
        p = Path(slide_path).resolve()
        root = slides_root.resolve()
        if not p.exists():
            raise HTTPException(404, f"slide not found: {slide_path}")
        if root not in p.parents and p != root:
            raise HTTPException(404, "slide outside served root")
        return p

    @app.post("/analyze", response_model=SlideReport)
    def analyze(req: AnalyzeRequest) -> SlideReport:
        return engine.analyze(_resolve(req.slide_path))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_revision": engine.revision,
            "ruo_disclaimer": engine.ruo,
        }

    return app
