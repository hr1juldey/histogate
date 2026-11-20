"""SlideReport findings JSON -> draft synoptic report (mTuitive/Lumea integration doctrine). Spec C.3.

Regulatory law encoded here: every rendered document carries the RUO
disclaimer, a TSR disclosure footnote whenever TSR is reported, and the
renderer VALIDATES ITS OWN OUTPUT — a mutated template that drops the
footer raises instead of shipping an unregulated report (PLAN PR-12 RED-e).
"""
from __future__ import annotations

import base64
import io

from histogate.schemas.report import SlideReport

MD_TEMPLATE = """# HistoGate Slide Analysis — {slide_id}

- Patches analyzed: {n_patches} ({n_gated} gated by QC)
- Model revision: {model_revision} | calibration T = {temperature}

## Compartment composition

{fractions_table}

{tsr_section}{til_section}{artifacts_section}

---

{ruo}
"""

HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>HistoGate {slide_id}</title></head>
<body>
<h1>HistoGate Slide Analysis — {slide_id}</h1>
<ul>
  <li>Patches analyzed: {n_patches} ({n_gated} gated by QC)</li>
  <li>Model revision: {model_revision} | calibration T = {temperature}</li>
</ul>
<h2>Compartment composition</h2>
{fractions_table}
<p><img alt="compartment donut" src="data:image/png;base64,{donut_b64}"></p>
{tsr_section}{til_section}{artifacts_section}
<hr>
<p><em>{ruo}</em></p>
</body></html>
"""


def _fractions_table(report: SlideReport, fmt: str) -> str:
    items = sorted(report.compartment_fractions.items(), key=lambda kv: -kv[1])
    rows = [
        f"| {t} | {f:.1%} |" if fmt == "md" else f"<li>{t}: {f:.1%}</li>"
        for t, f in items if f > 0
    ]
    if fmt == "md":
        return "| Tissue | Fraction |\n|---|---|\n" + "\n".join(rows)
    return "<ul>\n" + "\n".join(rows) + "\n</ul>"


def _donut_png(report: SlideReport) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [t for t, f in report.compartment_fractions.items() if f > 0]
    sizes = [report.compartment_fractions[t] for t in labels]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, wedgeprops={"width": 0.42})
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def _tsr_section(report: SlideReport, fmt: str) -> str:
    if report.tsr is None:
        return ""
    t = report.tsr
    if fmt == "md":
        return (
            "\n## Tumor-Stroma Ratio\n\n"
            f"TSR = **{t.tsr:.2f}** ({t.group}) — tumor {t.tumor_fraction:.1%} / "
            f"stroma {t.stroma_fraction:.1%}\n\n"
            f"> Disclosure: {t.disclosure} — {t.method}\n"
        )
    return (
        "\n<h2>Tumor-Stroma Ratio</h2>"
        f"<p>TSR = <strong>{t.tsr:.2f}</strong> ({t.group}) — "
        f"tumor {t.tumor_fraction:.1%} / stroma {t.stroma_fraction:.1%}</p>"
        f"<p><small>Disclosure: {t.disclosure} — {t.method}</small></p>"
    )


def _til_section(report: SlideReport, fmt: str) -> str:
    if report.til is None:
        return ""
    c = report.til
    tag = "##" if fmt == "md" else "<h2>"
    end = "" if fmt == "md" else "</h2>"
    return (
        f"\n{tag} TIL density{end}\n\nTIL intratumoral "
        f"{c.intratumoral_density_mm2:.1f}/mm², stromal {c.stromal_density_mm2:.1f}/mm² "
        f"({c.method}, mpp {c.mpp})\n"
    )


def _artifacts_section(report: SlideReport, fmt: str) -> str:
    if not report.artifacts:
        return ""
    if fmt == "md":
        lines = [
            f"- {a.kind.value} ({a.stage}): {a.pixel_fraction:.1%} of pixels"
            for a in report.artifacts
        ]
        return "\n## Pre-analytical artifacts\n\n" + "\n".join(lines) + "\n"
    items = "".join(
        f"<li>{a.kind.value} ({a.stage}): {a.pixel_fraction:.1%}</li>"
        for a in report.artifacts
    )
    return f"<h2>Pre-analytical artifacts</h2><ul>{items}</ul>"


def _self_validate(text: str, report: SlideReport, kind: str) -> None:
    if report.ruo_disclaimer not in text:
        raise ValueError(f"{kind} render lost the RUO disclaimer — refusing to ship")
    if report.tsr is not None and report.tsr.disclosure not in text:
        raise ValueError(f"{kind} render lost the TSR disclosure — refusing to ship")


def render_markdown(report: SlideReport) -> str:
    text = MD_TEMPLATE.format(
        slide_id=report.slide_id,
        n_patches=report.n_patches,
        n_gated=report.n_gated,
        model_revision=report.model_revision,
        temperature=report.calibration_temperature,
        fractions_table=_fractions_table(report, "md"),
        tsr_section=_tsr_section(report, "md"),
        til_section=_til_section(report, "md"),
        artifacts_section=_artifacts_section(report, "md"),
        ruo=report.ruo_disclaimer,
    )
    _self_validate(text, report, "markdown")
    return text


def render_html(report: SlideReport) -> str:
    png_b64 = base64.b64encode(_donut_png(report)).decode()
    text = HTML_TEMPLATE.format(
        slide_id=report.slide_id,
        n_patches=report.n_patches,
        n_gated=report.n_gated,
        model_revision=report.model_revision,
        temperature=report.calibration_temperature,
        fractions_table=_fractions_table(report, "html"),
        donut_b64=png_b64,
        tsr_section=_tsr_section(report, "html"),
        til_section=_til_section(report, "html"),
        artifacts_section=_artifacts_section(report, "html"),
        ruo=report.ruo_disclaimer,
    )
    _self_validate(text, report, "html")
    return text
