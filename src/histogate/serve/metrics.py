"""Demo-surface metrics: landing page numbers rendered from measured
artifacts only (zero hand-typed literals — same law as the docs)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

RUN = Path("runs/pr16-mixed")


@lru_cache(maxsize=1)
def landing_metrics() -> dict:
    def j(name: str) -> dict:
        p = RUN / name
        return json.loads(p.read_text()) if p.exists() else {}

    ev, dom, dr, gate, bench = (
        j("eval_best.json"), j("domain.json"), j("int8_drift.json"),
        j("gate_report.json"), j("onnx/export_bench.json"))
    tc = bench.get("torch_cuda", {})
    return {
        "revision": f"· ship: {gate.get('checkpoint', 'pr16-mixed').split('/')[-2] if gate.get('checkpoint') else 'pr16-mixed'} ({sum(c['passed'] for c in gate.get('checks', []))}/{len(gate.get('checks', []))} gate)",
        "crcval_f1": round(ev.get("val_macro_f1", 0.0), 4),
        "domain_f1": round(dom.get("nonorm_macro_f1", 0.0), 4),
        "int8_agree": round(dr.get("argmax_agreement", 0.0), 4),
        "throughput": round(tc.get("throughput_patches_s", 0.0), 1),
        "ece": round(j("calibration.json").get("ece_after", 0.0), 4),
        "gate_checks": [
            {"name": c["name"], "value": c["value"],
             "threshold": f"{c['comparator']} {c['threshold']}",
             "passed": bool(c["passed"])}
            for c in gate.get("checks", [])],
    }
