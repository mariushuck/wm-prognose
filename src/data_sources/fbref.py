"""Layer 3/4 — FBref team stats via ``soccerdata``. STUB.

Lazy optional import + politeness rate limit (see §9 legal notes: ToS / robots,
use rate limiting). Returns an empty frame when disabled or library missing.

Columns: season, team, matches, goals_for, goals_against, xg_for, xg_against
"""
from __future__ import annotations

import pandas as pd

FBREF_COLUMNS = [
    "season", "team", "matches", "goals_for", "goals_against", "xg_for", "xg_against",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in FBREF_COLUMNS})


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    if not cfg.get("enabled", False):
        return _empty()
    try:
        import soccerdata as sd  # noqa: F401  (lazy optional dependency)
    except ImportError:
        return _empty()
    # TODO: instantiate sd.FBref(...) with cfg['seasons'], honor
    # cfg['rate_limit_seconds'], pull team season stats into the contract above.
    raise NotImplementedError("fbref.load: implement soccerdata FBref pull")
