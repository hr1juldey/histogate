"""Assemble and upload the HistoGate Hugging Face Space (deploy checkpoint).

Builds a deploy tree in scratch (never inside the repo), then uploads it with
the hf CLI. Selection is explicit — the Space ships the ship checkpoint,
calibration + KNN bank + conformal artifacts, the 40 MB demo slide, the public
MLflow db, docs the MCP resources read, and code. No 23 GB Camelyon, no ONNX
dir, no last.ckpt duplicates.

Usage: uv run python scripts/deploy_space.py [--dry-run]
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path.home() / ".hermes/cache/scratch/histogate-space"
SPACE_README = """---
title: HistoGate
emoji: "\U0001F52C"
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
---

# HistoGate — quality-gated colorectal H&E tissue classifier

9-class tile classification with calibrated confidence, a KNN
out-of-distribution gate (declines unfamiliar tissue instead of guessing),
conformal prediction sets, TSR/TIL biomarkers, and an MCP server for coding
agents (streamable HTTP at /mcp). Research Use Only — not for diagnostic
procedures.

- Web demo: see the GitHub repo (frontend on Vercel)
- API: `/health`, `/analyze-patch` (upload a tile), `/slide-map.json`
- MCP: `https://<this-space>.hf.space/mcp` (protocol 2025-06-18)
"""

KEEP_RUN = [
    "best.ckpt.pt", "calibration.json", "conformal.json", "coverage.json",
    "crcval_accuracy.json", "domain.json", "eval_best.json", "gate_inputs.json",
    "gate_report.json", "metrics.json", "ood_knn.json", "ood_knn_bank.pt",
    "ood_knn_livepath.json",
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd[:6]), "…")
    return subprocess.run(cmd, check=True, cwd=DEST if kw.pop("in_dest", False)
                          else ROOT, **kw)


def main() -> None:
    dry = "--dry-run" in sys.argv
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    # code + lock
    shutil.copy2(ROOT / "Dockerfile", DEST / "Dockerfile")
    shutil.copy2(ROOT / "pyproject.toml", DEST / "pyproject.toml")
    shutil.copy2(ROOT / "uv.lock", DEST / "uv.lock")
    shutil.copytree(ROOT / "src", DEST / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (DEST / "README.md").write_text(SPACE_README)
    (DEST / "MODEL_CARD.md").write_text(
        (ROOT / "MODEL_CARD.md").read_text())

    # docs the MCP resources read (BENCHMARKS) + MCP.md for the connect guide
    (DEST / "docs").mkdir()
    for f in ("BENCHMARKS.md", "MCP.md", "ROADMAP_PR15-20.md"):
        src = ROOT / "docs" / f
        if src.exists():
            shutil.copy2(src, DEST / "docs" / f)

    # data + run artifacts + mlflow
    (DEST / "data/demo").mkdir(parents=True)
    shutil.copy2(ROOT / "data/demo/demo_slide.tif",
                 DEST / "data/demo/demo_slide.tif")
    (DEST / "runs/pr16-mixed").mkdir(parents=True)
    for f in KEEP_RUN:
        shutil.copy2(ROOT / "runs/pr16-mixed" / f,
                     DEST / "runs/pr16-mixed" / f)
    (DEST / "runs/bench-phikon").mkdir(parents=True)
    shutil.copy2(ROOT / "runs/bench-phikon/result.json",
                 DEST / "runs/bench-phikon/result.json")
    (DEST / "mlflow_public").mkdir()
    shutil.copy2(ROOT / "mlflow_public/mlflow.db",
                 DEST / "mlflow_public/mlflow.db")

    total = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"assembled {total / 1e6:.1f} MB at {DEST}")
    if dry:
        return
    import subprocess as sp

    r = sp.run(["hf", "repos", "create", "riju279/histogate", "--type",
                "space", "--space-sdk", "docker"], capture_output=True,
               text=True)
    already = "already exists" in (r.stdout + r.stderr).lower()
    print("create:", "exists" if already else r.stdout.strip() or r.stderr.strip())
    run(["hf", "upload", "riju279/histogate", ".", ".", "--repo-type",
         "space"], in_dest=True)
    print("uploaded → https://huggingface.co/spaces/riju279/histogate")


if __name__ == "__main__":
    main()
