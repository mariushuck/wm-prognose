"""Squad-value feature group (Layer 3) — Transfermarkt market values.

Consumes the ``transfermarkt`` cache (manually maintained). Values are static per
team, so for historical training rows they act as a constant team-strength proxy
(slightly anachronistic — documented). Zero-filled when the cache is empty.

Columns:
    market_value_log_diff   log1p(home_value) - log1p(away_value)
    avg_age_diff            home avg age - away avg age
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["market_value_log_diff", "avg_age_diff"]


def columns() -> list:
    return list(COLUMNS)


def prepare(matches_df: pd.DataFrame, context: dict) -> None:
    """Per-team {market_value, avg_age} -> context['squad']."""
    tm = context.get("sources", {}).get("transfermarkt", pd.DataFrame())
    snap = {}
    if not tm.empty:
        for r in tm.itertuples(index=False):
            snap[r.team] = {
                "mv": float(getattr(r, "market_value", 0) or 0),
                "age": float(getattr(r, "avg_age", 0) or 0),
            }
    context["squad"] = snap


def _diffs(home: str, away: str, snap: dict) -> dict:
    if home not in snap or away not in snap:
        return {"market_value_log_diff": 0.0, "avg_age_diff": 0.0}
    h, a = snap[home], snap[away]
    return {
        "market_value_log_diff": float(np.log1p(h["mv"]) - np.log1p(a["mv"])),
        "avg_age_diff": h["age"] - a["age"],
    }


def build(matches_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    n = len(matches_df)
    if "squad" not in context:
        prepare(matches_df, context)
    snap = context.get("squad", {})
    if not snap or matches_df.empty:
        return pd.DataFrame(0.0, index=range(n), columns=COLUMNS)
    rows = [_diffs(h, a, snap) for h, a in
            zip(matches_df["home_team"], matches_df["away_team"])]
    return pd.DataFrame(rows, columns=COLUMNS)


def fixture_features(home: str, away: str, neutral: bool, context: dict) -> dict:
    return _diffs(home, away, context.get("squad", {}))
