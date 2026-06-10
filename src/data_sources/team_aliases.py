"""Reconcile team-name spellings across sources to the martj42 canonical names.

The results dataset (martj42) is our reference spelling. Kaggle's FIFA ranking and
StatsBomb use different conventions ("Korea Republic", "IR Iran", "USA", "Côte
d'Ivoire", …). ``normalize(name)`` maps a foreign spelling to the martj42 name so
joins line up. This is a **starter map** — extend ``ALIASES`` if a source uses a
spelling not covered here (unmapped names pass through unchanged).
"""
from __future__ import annotations

# foreign spelling (lower-cased) -> martj42 canonical name
ALIASES = {
    "korea republic": "South Korea",
    "korea dpr": "North Korea",
    "ir iran": "Iran",
    "iran islamic republic": "Iran",
    "usa": "United States",
    "united states of america": "United States",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "congo": "Congo",
    "cabo verde": "Cape Verde",
    "czechia": "Czech Republic",
    "china pr": "China PR",
    "chinese taipei": "Taiwan",
    "bosnia": "Bosnia and Herzegovina",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "curacao": "Curaçao",
    "the gambia": "Gambia",
    "cape verde islands": "Cape Verde",
    "north macedonia": "North Macedonia",
    "kyrgyz republic": "Kyrgyzstan",
}


def normalize(name) -> str:
    """Return the martj42 canonical spelling for a team name."""
    if name is None:
        return ""
    key = str(name).strip()
    return ALIASES.get(key.lower(), key)
