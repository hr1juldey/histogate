"""PR-12: report rendering — PLAN.md PR-12 RED contract.

Adversarial angles: RUO disclaimer must survive into BOTH output formats;
TSR disclosure footnote must exist when TSR exists and the section must NOT
exist when it can't (all-BACK slide); the donut must be a real decodable PNG;
and the renderer must SELF-VALIDATE — a corrupted template (dropped footer)
must raise, not silently ship an unregulated report.
"""
from __future__ import annotations

import base64
import io
import re

import pytest

from histogate.schemas.report import SlideReport, TILResult, TSRResult

RUO = "Research Use Only. Not for use in diagnostic procedures."


def _report(tsr: bool = True) -> SlideReport:
    return SlideReport(
        slide_id="slide-042",
        n_patches=64,
        n_gated=5,
        compartment_fractions={"TUM": 0.4, "STR": 0.3, "LYM": 0.1, "NORM": 0.2},
        artifacts=[],
        tsr=TSRResult(
            stroma_fraction=0.43, tumor_fraction=0.57, tsr=0.57, group="stroma-low",
            method="patch-proxy-vanpelt-2018", disclosure="Patch-proxy TSR per van Pelt 2018; RUO",
        ) if tsr else None,
        til=TILResult(intratumoral_density_mm2=12.0, stromal_density_mm2=30.0),
        calibration_temperature=1.31,
        model_revision="abc1234",
        ruo_disclaimer=RUO,
    )


@pytest.fixture(scope="module")
def rendered():
    from histogate.report import render_html, render_markdown

    r = _report()
    return render_markdown(r), render_html(r)


def test_ruo_present_in_markdown_and_html(rendered) -> None:
    md, html = rendered
    assert RUO in md, "RUO disclaimer missing from markdown"
    assert RUO in html, "RUO disclaimer missing from HTML"


def test_tsr_disclosure_footnote_present(rendered) -> None:
    md, html = rendered
    assert "van Pelt" in md and "van Pelt" in html
    assert re.search(r"disclos", md, re.I), "TSR disclosure footnote missing in md"


def test_donut_is_valid_png(rendered) -> None:
    from PIL import Image

    html = rendered[1]
    m = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', html)
    assert m, "no embedded donut PNG in HTML"
    raw = base64.b64decode(m.group(1))
    img = Image.open(io.BytesIO(raw))
    assert img.format == "PNG" and img.size[0] >= 200


def test_all_back_slide_has_no_tsr_section_but_validates() -> None:
    from histogate.report import render_markdown

    back = _report(tsr=False).model_copy(
        update={"compartment_fractions": {"BACK": 1.0}, "til": None}
    )
    md = render_markdown(back)
    assert "Tumor-Stroma Ratio" not in md
    assert RUO in md  # still a regulated document


def test_template_mutation_raises_instead_of_shipping() -> None:
    """Dropping the footer from the template must FAIL the render (self-check)."""
    import histogate.report as rep

    original = rep.MD_TEMPLATE
    mutated = original.replace("{ruo}", "")
    rep.MD_TEMPLATE = mutated
    try:
        with pytest.raises(ValueError, match="RUO"):
            rep.render_markdown(_report())
    finally:
        rep.MD_TEMPLATE = original  # probe restores what it mutates
