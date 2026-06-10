"""Context feature group (Layer 5) — rest days + match-day weather.

``rest_days`` is computed directly from the results history (always available);
``weather`` comes from the ``weather`` cache, keyed by ``(home, away)`` (only
populated for dated/venued fixtures, so it is zero for historical training rows).
Travel distance is deferred — it needs a per-match venue, which the results data
lacks (``venues.haversine_km`` is ready for when fixtures carry venues).

Columns: rest_days_diff, temperature_c, wind_kph
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

COLUMNS = ["rest_days_diff", "temperature_c", "wind_kph"]
_DEFAULT_REST = 7
_MAX_REST = 30


def columns() -> list:
    return list(COLUMNS)


def _weather_map(context: dict) -> dict:
    wx = context.get("sources", {}).get("weather", pd.DataFrame())
    out = {}
    if not wx.empty:
        for r in wx.itertuples(index=False):
            out[(r.home_team, r.away_team)] = {
                "temperature_c": float(getattr(r, "temperature_c", 0) or 0),
                "wind_kph": float(getattr(r, "wind_kph", 0) or 0),
            }
    return out


def prepare(matches_df: pd.DataFrame, context: dict) -> None:
    context["weather_map"] = _weather_map(context)


def _rest_series(matches_df: pd.DataFrame) -> pd.Series:
    """rest_days_diff (home - away) per match, chronological single pass."""
    last: Dict[str, pd.Timestamp] = {}
    diffs = []
    for r in matches_df.itertuples(index=False):
        d = r.date
        h_prev, a_prev = last.get(r.home_team), last.get(r.away_team)
        h_rest = min((d - h_prev).days, _MAX_REST) if h_prev is not None else _DEFAULT_REST
        a_rest = min((d - a_prev).days, _MAX_REST) if a_prev is not None else _DEFAULT_REST
        diffs.append(float(h_rest - a_rest))
        last[r.home_team] = d
        last[r.away_team] = d
    return pd.Series(diffs, index=range(len(matches_df)))


def build(matches_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    n = len(matches_df)
    if matches_df.empty:
        return pd.DataFrame(0.0, index=range(n), columns=COLUMNS)
    if "weather_map" not in context:
        prepare(matches_df, context)
    wmap = context.get("weather_map", {})

    rest = _rest_series(matches_df.reset_index(drop=True))
    temp, wind = [], []
    for h, a in zip(matches_df["home_team"], matches_df["away_team"]):
        w = wmap.get((h, a), {"temperature_c": 0.0, "wind_kph": 0.0})
        temp.append(w["temperature_c"])
        wind.append(w["wind_kph"])
    return pd.DataFrame({"rest_days_diff": rest.values,
                         "temperature_c": temp, "wind_kph": wind}, columns=COLUMNS)


def fixture_features(home: str, away: str, neutral: bool, context: dict) -> dict:
    w = context.get("weather_map", {}).get(
        (home, away), {"temperature_c": 0.0, "wind_kph": 0.0})
    # No kickoff date for a prospective fixture -> rest_days_diff neutral (0).
    return {"rest_days_diff": 0.0, **w}