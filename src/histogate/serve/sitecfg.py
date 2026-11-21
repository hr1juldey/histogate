"""Server-driven UI config: the Astro front end renders its sections and copy
from this endpoint, so content changes ship as backend config — no front-end
redeploy. Numbers still flow from measured artifacts via /metrics.json.

COPY LAW (user, 2026-11-22): understandable by common humans FIRST, specialists
second. Every claim gets a plain sentence; the technical term rides along as the
small secondary label. Nobody was ever hired by jargon.
"""
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
                "Show it a magnified piece of a biopsy. It tells you what kind of "
                "tissue that is, how sure it is — and when the picture is too poor "
                "or too strange to judge, it says so instead of guessing. "
                "It's running live, right now, on this page."),
            "primary_cta": {"label": "See it work", "href": "/demo"},
            "secondary_cta": {"label": "How it failed, and how we fixed it",
                              "href": "/story"},
        },
        "journey": [
            {"run": "overnight-30ep", "title": "The first try",
             "note": "We taught it with 100,000 tissue pictures that doctors had "
                     "already labeled. Perfect grades on the test — but the test "
                     "was too easy. It had only ever seen one hospital's machines."},
            {"run": "nonorm-ft", "title": "The fix that backfired",
             "note": "We showed it pictures from other machines — but pushed too "
                     "hard, and it forgot almost everything in a single night of "
                     "training. A classic mistake, kept here on purpose."},
            {"run": "nonorm-ft2", "title": "The careful fix",
             "note": "Slower, gentler retraining. It kept most of its skill and "
                     "learned the new machines — but not well enough to ship. "
                     "Close isn't shipped."},
            {"run": "pr16-mixed", "title": "The fix that worked",
             "note": "Instead of just more data, we taught it the physics of how "
                     "tissue staining works — so a different machine reads as a "
                     "different color, not a different disease. Now it scores "
                     "better on machines it has never seen than on its own "
                     "homework."},
        ],
        "cases": [
            {"id": "collapse", "title": "It failed on machines it hadn't seen",
             "what": "Same patients, same doctors' labels — but the slide was "
                     "scanned by a different machine, and the AI's answers fell "
                     "apart. In a hospital, that's the difference between a tool "
                     "and a liability.",
             "fix": "We rebuilt its training around the chemistry of staining "
                    "itself, so a change of machine reads as a change of color, "
                    "not a change of disease.",
             "metric": "raw-scanner score"},
            {"id": "forgetting", "title": "The obvious fix destroyed it",
             "what": "Retrain it on the new pictures — the obvious move. Done "
                     "carelessly, the AI forgets its old skills almost "
                     "overnight. Ours did, and we kept the failed run so nobody "
                     "has to take our word for it.",
             "fix": "Gentle retraining with a hard pass/fail bar before anything "
                    "ships. The failed attempts stay visible on this page.",
             "metric": "familiar-hospital score"},
            {"id": "ensemble", "title": "Two AIs voting was worse than one",
             "what": "We built a committee of models to vote on each picture. "
                     "The idea is sound — the votes really did help on the hard "
                     "pictures — but one weak member dragged the group below "
                     "the single best model.",
             "fix": "We measured it, it failed the bar, we didn't ship it. "
                    "Published as an honest negative result.",
             "metric": "ship gate"},
        ],
        "gate_note": ("Before anything ships, it must pass nine safety and quality "
                      "checks. Eight pass. One honestly fails, and we show that "
                      "too — an AI that hides its weaknesses in a hospital is a "
                      "liability, not a product."),
        "who": ("Built by a biologist who became a deep-learning engineer — the "
                "tissue, the stains and the machines are the familiar part, and "
                "the AI is the part built to pass measurable bars."),
        "about": {
            "name": "Hrijul Dey",
            "role": "AI & Backend Engineer — production agents, APIs, and measurement",
            "photo_caption": "At work, 2026. The day job is agent infrastructure; this site is the biology half.",
            "timeline": [
                {"year": "2018–22", "title": "B.Tech, Biotechnology", "org": "Vellore Institute of Technology",
                 "note": "The formal biology half — the degree the tissue on this site comes from."},
                {"year": "2019–23", "title": "Biomedical R&D — computer vision & data science", "org": "Ashva, biomedical devices",
                 "note": "The work that matters here: estimated knee and ankle joint loading from multi-sensor motion data, including immature, partially-calibrated sensors; posture detection and gait analysis on the CMU motion-capture database; test design, validation and documentation against regulated standards. This is where 'measurement or it didn't happen' was learned — on medical devices, not slide decks."},
                {"year": "2025", "title": "Product Manager & Software Engineer", "org": "Lilypad — EV marketplace",
                 "note": "Shipped production LLM features on India's first MCP-driven marketplace: tool integration, APIs, data flow. Owned requirements through release, prioritized by business impact and technical feasibility."},
                {"year": "2026–", "title": "Product Engineer, AI Platform", "org": "Mosaic Wellness",
                 "note": "Owns 'Kai' end-to-end — the agent platform a 500-person organization runs daily: an MCP gateway federating hundreds of tool servers behind one governed endpoint. Cut organization-wide LLM token cost ~93% (15x) while reliability held at org-wide scale. Trained small task models, measured them, rolled them back when the numbers favored deterministic routing. Built the automated failure-handling loop that ships micro-fixes behind adversarial tests."},
                {"year": "Nov 2026", "title": "Open source", "org": "22 PRs across 6 orgs in four days",
                 "note": "Token-accounting fix merged upstream in Hugging Face smolagents (PR #3206); 13 PRs in Trail-Code; 5 in NousResearch hermes-agent — found by running the agent daily on itself."},
                {"year": "This site", "title": "HistoGate", "org": "The two halves, joined",
                 "note": "A biotechnologist's tissue problem, attacked with a production engineer's release discipline. Every number measured, every failure published."},
            ],
            "links": [
                {"label": "GitHub — all code", "href": "https://github.com/hr1juldey"},
                {"label": "LinkedIn", "href": "https://www.linkedin.com/in/dey-h-ml972"},
                {"label": "ctx-gateway — public eval harness", "href": "https://github.com/hr1juldey/ctx-gateway"},
                {"label": "This project on GitHub", "href": "https://github.com/hr1juldey/histogate"},
                {"label": "This project on Hugging Face", "href": "https://huggingface.co/riju279/histogate"},
            ],
            "faith": (
                "Clinical AI has one unforgivable sin: confident wrong answers. "
                "So this system is built to say 'I can't judge this picture' — "
                "and this page is built to show where it failed and what it cost. "
                "That is the standard I want to build to, and be hired to."),
        },
    }
