"""PR-0 bootstrap tests.

Oracle: the filesystem itself (docs committed in this repo). Adversarial core:
the suite fails if any designed artifact is missing or emptied — a doc that
was deleted or truncated cannot silently pass review.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REQUIRED_DOCS = [
    "docs/PRD.md",
    "docs/HLD.md",
    "docs/LLD.md",
    "docs/PLAN.md",
    "docs/TESTING.md",
    "docs/UI.md",
    "docs/PERF.md",
    "docs/DEPLOYMENT.md",
    "pyproject.toml",
    "Makefile",
]


@pytest.mark.parametrize("rel", REQUIRED_DOCS)
def test_required_file_exists_nonempty(rel: str) -> None:
    p = REPO / rel
    assert p.exists(), f"missing required file: {rel}"
    assert p.stat().st_size > 100, f"required file is empty/suspiciously small: {rel}"


def test_ruo_disclaimer_in_prd() -> None:
    """The RUO posture must be stated in the PRD before any code ships."""
    prd = (REPO / "docs/PRD.md").read_text()
    assert "Research Use Only" in prd or "RUO" in prd


def test_no_money_figures_in_public_docs() -> None:
    """Public-money law: no absolute money figures in repo docs.
    Oracle: independent regex for rupee/crore/lakh AMOUNTS (₹ adjacent to
    digits, or 'lakh'/'crore' tokens) — quoting the rule itself is legal."""
    import re

    money = re.compile(r"₹\s*\d|\d\s*₹|\b\d+\s*(lakh|lpa|crore)\b|\b(lakh|crore)\s*\d", re.I)
    for rel in REQUIRED_DOCS:
        text = (REPO / rel).read_text()
        hit = money.search(text)
        assert hit is None, f"money figure {hit!r} found in {rel} (public-money law)"
