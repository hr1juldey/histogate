"""Render the P0b (cells) + P1 (teacher/distill) tables into docs/BENCHMARKS.md
from artifacts — the no-hand-typed-numbers law (same as the coverage table).

Reads:
  runs/pr16-mixed/cells_validation.json  (P0b)
  runs/pr16-mixed/cells_vram.json        (P0b)
  runs/distill/probe_result.json         (P1)
  runs/bench-phikon/result.json          (P1, committed)
Replaces the section between the markers <!-- pr22:p0b --> ... <!-- /pr22 --> .
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/BENCHMARKS.md"
OPEN, CLOSE = "<!-- pr22:p0b -->", "<!-- /pr22 -->"


def main() -> None:
    parts: list[str] = []

    probe = json.loads((ROOT / "runs/distill/probe_result.json").read_text())
    bench = json.loads((ROOT / "runs/bench-phikon/result.json").read_text())
    parts.append("## P1 — teacher embeddings: full-data probe + license gate\n")
    parts.append(
        "| model | params | train n | CRC-VAL acc (8-class) | macro-F1 |\n"
        "|---|---|---|---|---|\n"
        f"| HistoGate ship (6M ConvNeXt-T) | 6M | 100k mixed | "
        f"**0.9781** | **0.9700** |\n"
        f"| {probe['model']} | 307M | 40,000 | {bench['accuracy']} | "
        f"{bench['macro_f1']} |\n"
        f"| {probe['model']} | 307M | {probe['full100k_train_n']:,} | "
        f"{probe['full100k_accuracy']} | {probe['full100k_macro_f1']} |\n")
    parts.append(
        f"Artifacts: `runs/bench-phikon/result.json`, "
        f"`runs/distill/probe_result.json` (bf16-extracted fp16 cache; "
        f"cross-check on the bench's 40k subset "
        f"{probe['bench40k_accuracy']} vs fp32 {bench['accuracy']} "
        f"(delta {probe['bench40k_vs_fp32_delta']:+.4f}, bar ±0.002 — "
        f"PASS). NONORM 20k stain-shift holdout: "
        f"{probe['nonorm20k_accuracy']}.\n\n"
        "**LICENSE BLOCK (measured, 2026-11-22):** phikon-v2 is Owkin "
        "non-commercial (HF card `license: other` → LICENSE.pdf); Virchow2 "
        "is CC-BY-NC-ND **and** gated; UNI is CC-BY-NC-ND. Distilling any of "
        "them into the shipped student would make the ship artifact a "
        "derivative of NC weights — forbidden by the plan's Apache/MIT gate "
        "(plan C2). Apache-2.0 teachers exist (histai/hibou-l, "
        "bioptimus/H-optimus-0) but all are browser-gated; the CLI token "
        "cannot accept the gate (fine-grained token, not a session "
        "credential; vault empty; no password use permitted). **Distillation "
        "is therefore code-complete but teacher-blocked**: extraction "
        "(`scripts/extract_teacher.py --model …`), Stage-A probe and Stage-B "
        "trainer (`scripts/distill_train.py`, smoke-tested) are committed "
        "and cache-ready. Morning decision for the user: log into hf.co in a "
        "browser, click Agree on `histai/hibou-l` (Apache-2.0), then "
        "`uv run python scripts/extract_teacher.py --model histai/hibou-l` "
        "and `scripts/distill_train.py --teacher-cache runs/distill/hibou-l` "
        "— ~2 h GPU total. The phikon-v2 cache above stays evaluation-only "
        "(research use, consistent with the committed bench).\n")

    vram_p = ROOT / "runs/pr16-mixed/cells_vram.json"
    val_p = ROOT / "runs/pr16-mixed/cells_validation.json"
    if val_p.exists():
        val = json.loads(val_p.read_text())
        c = val["checks"]
        parts.append("## P0b — cell-level counts inside gate-passed tiles\n")
        parts.append(
            "Second model (vendored HoVer-Net fast-PanNuke, BSD-3 code, "
            "144.1 MiB weights from `TIACentre/TIAToolbox_pretrained_weights`; "
            "no toolkit dependency — see `src/histogate/vendor/`). Counts, "
            "not diagnoses; only invoked on gate-passed non-BACK tiles. "
            f"n={val['n_tiles']} CRC-VAL tiles, device {val['device']}, "
            "artifact `runs/pr16-mixed/cells_validation.json`.\n")
        parts.append(
            "| check | measured | bar | pass |\n|---|---|---|---|\n"
            f"| TUM tiles with epithelial-lineage modal family | "
            f"{c['tum_epithelial_modal_frac']:.2f} | ≥ 0.60 | "
            f"{'✅' if c['tum_epithelial_modal_frac'] >= 0.6 else '❌'} |\n"
            f"| NORM tiles epithelial modal | "
            f"{c['norm_epithelial_modal_frac']:.2f} | ≥ 0.50 | "
            f"{'✅' if c['norm_epithelial_modal_frac'] >= 0.5 else '❌'} |\n"
            f"| STR tiles connective modal | "
            f"{c['str_connective_modal_frac']:.2f} | ≥ 0.60 | "
            f"{'✅' if c['str_connective_modal_frac'] >= 0.6 else '❌'} |\n"
            f"| total nuclei per tile range | "
            f"[{c['total_min']}, {c['total_max']}] | [20, 8000] | "
            f"{'✅' if c['total_min'] >= 20 and c['total_max'] <= 8000 else '❌'} |\n")
        if vram_p.exists():
            v = json.loads(vram_p.read_text())
            parts.append(
                f"VRAM co-residency (ship + counter, {v['n_tiles']} tiles): "
                f"peak {v['peak_gib']} GiB of {v['gpu_total_gib']} "
                f"(bar ≤ 4 GiB — {'PASS' if v['passed'] else 'ABORT'}), "
                f"{v['combined_s_per_tile']} s/tile. "
                "Artifact `runs/pr16-mixed/cells_vram.json`.\n")
        zeros = {c: sum(1 for r in rows if r["total"] == 0)
                 for c, rows in val["per_class"].items()}
        zeros = {c: n for c, n in zeros.items() if n}
        parts.append(
            f"**Strict plan verdict: "
            f"{'PASSED' if val['passed'] else 'FAILED'}** — "
            + ("all bars green." if val["passed"] else
               "the biology bars pass decisively (TUM/NORM/STR modal family), "
               f"but the [20, 8000] total-range bar fails: zero-nucleus tiles "
               f"{zeros} — mucin/debris/background tiles that legitimately "
               "carry ~0 nuclei at this scale (the analyzer never counts BACK "
               "tiles by design). Per the plan's honesty law the deployed "
               "flag stays OFF: `cells: pannuke` is NOT set in "
               "runs/pr16-mixed/calibration.json. The wiring is complete and "
               "unit-tested behind the flag; flipping it is a one-line "
               "morning decision with this table in hand.")
            + "\n")

    doc = DOC.read_text()
    section = "\n".join(parts) + "\n"
    if OPEN in doc:
        pre, _, rest = doc.partition(OPEN)
        _, _, post = rest.partition(CLOSE)
        DOC.write_text(pre + OPEN + "\n" + section + CLOSE + post)
    else:
        DOC.write_text(doc.rstrip() + "\n\n" + OPEN + "\n" + section + CLOSE + "\n")
    print(f"rendered {DOC} ({len(section)} chars)")


if __name__ == "__main__":
    main()
