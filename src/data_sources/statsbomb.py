"""Layer 4 — StatsBomb open data (World Cup xG). STUB.

``statsbombpy`` is imported lazily so the base install does not require it. When
disabled or the library is missing, returns an empty xG frame.

Columns: date, team, opponent, xg, xa
"""
from __future__ import annotations

import pandas as pd

STATSBOMB_COLUMNS = ["date", "team", "opponent", "xg", "xa"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in STATSBOMB_COLUMNS})


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    if not cfg.get("enabled", False):
        return _empty()
    try:
        from statsbombpy import sb  # noqa: F401  (lazy optional dependency)
    except ImportError:
        return _empty()
    # TODO: for each {competition_id, season_id} pull matches + events, aggregate
    # shot xG per team per match into the column contract above.
    raise NotImplementedError("statsbomb.load: implement competition/event aggregation")
