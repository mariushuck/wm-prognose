"""Advanced-stats feature group (Layer 4) — xG / xA differentials.

Consumes the ``statsbomb`` cache (per-match team xG/xA). Uses each team's average
xG/xA across the available matches (StatsBomb open data covers the 2018 & 2022
World Cups, so coverage is sparse — most historical rows get zeros). Zero-filled
when the cache is empty.

Columns: xg_diff, xa_diff   (home average minus away average)
"""
from __future__ import annotations

import pandas as pd

COLUMNS = ["xg_diff", "xa_diff"]


def columns() -> list:
    return list(COLUMNS)


def prepare(matches_df: pd.DataFrame, context: dict) -> None:
    """Per-team average {xg, xa} -> context['xg']."""
    sb = context.get("sources", {}).get("statsbomb", pd.DataFrame())
    snap = {}
    if not sb.empty:
        agg = sb.groupby("team")[["xg", "xa"]].mean()
        snap = {t: {"xg": float(r["xg"]), "xa": float(r["xa"])} for t, r in agg.iterrows()}
    context["xg"] = snap


def _diffs(home: str, away: str, snap: dict) -> dict:
    h = snap.get(home, {"xg": 0.0, "xa": 0.0})
    a = snap.get(away, {"xg": 0.0, "xa": 0.0})
    return {"xg_diff": h["xg"] - a["xg"], "xa_diff": h["xa"] - a["xa"]}


def build(matches_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    n = len(matches_df)
    if "xg" not in context:
        prepare(matches_df, context)
    snap = context.get("xg", {})
    if not snap or matches_df.empty:
        return pd.DataFrame(0.0, index=range(n), columns=COLUMNS)
    rows = [_diffs(h, a, snap) for h, a in
            zip(matches_df["home_team"], matches_df["away_team"])]
    return pd.DataFrame(rows, columns=COLUMNS)


def fixture_features(home: str, away: str, neutral: bool, context: dict) -> dict:
    return _diffs(home, away, context.get("xg", {}))
