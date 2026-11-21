"""Server-driven UI config: the Astro front end renders its sections and copy
from this endpoint, so content changes ship as backend config — no front-end
redeploy. Numbers still flow from measured artifacts via /metrics.json."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path("configs/site_ui.json")


@lru_cache(maxsize=1)
def site_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return _default()


def _default() -> dict:
    return {
        "product": {
            "name": "HistoGate",
            "tagline": "One of these patches is cancer. It knows which.",
            "hero_sub": (
                "HistoGate reads stained biopsy patches the way a pathologist "
                "skims them — tissue type, calibrated confidence, and whether "
                "the image deserves an answer at all. It runs live, right now."),
            "primary_cta": {"label": "See it classify", "href": "/demo"},
            "secondary_cta": {"label": "How it broke, and recovered",
                              "href": "/story"},
        },
        "journey": [
            {"run": "overnight-30ep", "title": "The run",
             "note": "100,000 expert-annotated patches. Perfect on clean data."},
            {"run": "nonorm-ft", "title": "The wrong fix",
             "note": "Default learning rate destroyed the classifier — "
                     "catastrophic forgetting in one epoch."},
            {"run": "nonorm-ft2", "title": "The gentle fix",
             "note": "Recovered most of the domain; still missed the gate."},
            {"run": "pr16-mixed", "title": "The physics fix",
             "note": "Beer-Lambert dye-concentration augmentation; the domain "
                     "failure didn't just close — it inverted."},
        ],
        "cases": [
            {"id": "collapse", "title": "Domain collapse on raw scanners",
             "what": "The first model scored near-ceiling on its own hospital's data "
                     "and collapsed on the same patients scanned without lab normalization.",
             "fix": "Stain physics augmentation (dye-concentration jitter measured on "
                    "real slides), trained on mixed normalized + raw data.",
             "metric": "raw-scanner F1"},
            {"id": "forgetting", "title": "Catastrophic forgetting",
             "what": "The obvious fix — fine-tune on raw data at the default "
                     "learning rate — destroyed the classifier in a single epoch.",
             "fix": "Warm-start at 1e-4 with a measured acceptance gate; the failed "
                    "runs are kept in the journey as documented negative results.",
             "metric": "external F1"},
            {"id": "ensemble", "title": "The ensemble that didn't ship",
             "what": "Voting gain was real and localized on disagreement patches — "
                     "but one member's domain weakness poisoned the aggregate.",
             "fix": "Rejected on measurement, not vibes. Published as a negative "
                    "result with the full battery.",
             "metric": "acceptance gate"},
        ],
        "gate_note": ("Every release faces the same nine checks on held-out and "
                      "out-of-distribution data. Eight pass. One fails. Both are shown."),
        "who": ("Biotechnology graduate turned deep-learning engineer; shipped "
                "consumer-health products at Man Matters, Bodywise and Lady Joyce."),
        "contact": {
            "headline": "Working on digital pathology or computational biology?",
            "sub": ("HistoGate is a research prototype — open code, checkpoints and "
                    "failures. Reach out for a walkthrough, the full build log, or "
                    "the résumé behind it."),
        },
    }
