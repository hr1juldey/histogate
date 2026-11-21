"""FastAPI surface — PLAN PR-13. response_model is the contract; 404/422 the
negative paths; /health identifies the regulated service (revision + RUO).
Demo surface (PR-18 era): /ui single-page patch demo + /analyze-patch
upload endpoint (calibrated probabilities, energy OOD, Grad-CAM)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel

from histogate.schemas.report import SlideReport
from histogate.serve.engine import LockedEngine
from histogate.serve.patch_demo import PatchAnalyzer


class AnalyzeRequest(BaseModel):
    slide_path: str


def create_app(engine: LockedEngine, slides_root: Path,
               patch_analyzer: PatchAnalyzer | None = None,
               legacy_analyzer: PatchAnalyzer | None = None) -> FastAPI:
    app = FastAPI(title="HistoGate", version="0.1.0")
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # demo front end (Vercel / local) — read-only inference
        allow_methods=["*"], allow_headers=["*"],
    )
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

    @app.get("/")
    def root() -> dict[str, Any]:
        """API gateway root — this service is REST/MCP only.
        The human UI lives on the Astro front end (Vercel / local dev)."""
        return {
            "service": "HistoGate",
            "kind": "REST + MCP inference gateway",
            "endpoints": ["/health", "/analyze", "/analyze-patch",
                          "/samples", "/metrics.json", "/site-config.json",
                          "/contact", "/docs"],
            "ruo_disclaimer": engine.ruo,
        }

    @app.get("/metrics.json")
    def metrics() -> dict[str, Any]:
        from histogate.serve.metrics import landing_metrics
        return landing_metrics()

    @app.get("/site-config.json")
    def site_config() -> dict[str, Any]:
        from histogate.serve.sitecfg import site_config as cfg
        return cfg()

    if patch_analyzer is not None:
        SAMPLES = [
            ("tumor", "TUM — tumor epithelium: the cancer cells themselves",
             "CRC-VAL-HE-7K/TUM/TUM-TCGA-AAHIDTWA.tif"),
            ("normal", "NORM — healthy gland architecture for contrast",
             "CRC-VAL-HE-7K/NORM/NORM-TCGA-AASSYQPA.tif"),
            ("stroma", "STR — the connective tissue a tumor invades",
             "CRC-VAL-HE-7K/STR/STR-TCGA-AAMALCER.tif"),
            ("lymph", "LYM — immune-cell clusters, the TIL story",
             "CRC-VAL-HE-7K/LYM/LYM-TCGA-AAWGSCHH.tif"),
            ("mucus", "MUC — mucin lakes, a visually odd class",
             "CRC-VAL-HE-7K/MUC/MUC-TCGA-AADHFQAT.tif"),
            ("debris", "DEB — necrosis/debris: ugly but real tissue",
             "CRC-VAL-HE-7K/DEB/DEB-TCGA-AANNAWLE.tif"),
            ("nonorm", "TUM tile from a RAW scanner (no lab normalization) — the data that broke the first model",
             "NCT-CRC-HE-100K-NONORM/TUM/TUM-AACHEHDV.tif"),
        ]

        @app.get("/samples")
        def samples() -> list[dict[str, str]]:
            return [{"id": i, "why": w}
                    for i, w, _ in SAMPLES]

        @app.get("/samples/{sid}.png")
        def sample_png(sid: str) -> Response:
            import io

            from PIL import Image
            for i, _, rel in SAMPLES:
                if i == sid:
                    buf = io.BytesIO()
                    Image.open(Path("data/raw") / rel).convert(
                        "RGB").save(buf, format="PNG")
                    return Response(buf.getvalue(),
                                    media_type="image/png")
            raise HTTPException(404, "unknown sample")

        @app.get("/samples/{sid}", response_class=FileResponse)
        def sample_img(sid: str) -> FileResponse:
            for i, _, rel in SAMPLES:
                if i == sid:
                    return FileResponse(Path("data/raw") / rel,
                                        media_type="image/tiff")
            raise HTTPException(404, "unknown sample")

        @app.post("/analyze-patch")
        async def analyze_patch(file: UploadFile,
                                model: str = Form("ship")) -> dict[str, Any]:
            try:
                img = Image.open(file.file)
                img.load()
                analyzer = patch_analyzer
                if model == "v1" and legacy_analyzer is not None:
                    analyzer = legacy_analyzer
                return analyzer.analyze(img)
            except Exception as e:  # noqa: BLE001 — demo surface, 422 contract
                raise HTTPException(422, f"bad image: {e}") from e

    return app
