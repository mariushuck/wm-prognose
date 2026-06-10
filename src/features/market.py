"""Market feature group (Layer 2/3) — odds-implied probabilities + squad value. STUB.

Contributes no columns until the odds (Layer 2) / transfermarkt (Layer 3) caches
exist. Planned columns: implied_home, implied_draw, implied_away, market_value_diff,
avg_age_diff.
"""
from __future__ import annotations

import pandas as pd

COLUMNS = ["implied_home", "implied_draw", "implied_away", "market_value_diff", "avg_age_diff"]


def columns() -> list:
    return list(COLUMNS)


def build(matches_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    # No odds/squad caches wired yet -> contribute nothing.
    return pd.DataFrame(index=range(len(matches_df)))


def fixture_features(home: str, away: str, neutral: bool, context: dict) -> dict:
    return {}
