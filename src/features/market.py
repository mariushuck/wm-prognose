"""Market feature group (Layer 2) — bookmaker-implied probabilities.

Consumes the ``odds`` cache (decimal h2h odds), keyed by ``(home_team, away_team)``,
and converts to overround-free (de-vigged) implied probabilities. Only fixtures
present in the cache get non-zero values — so this is inert for historical training
unless an odds-history file is supplied (see data_sources.yaml: odds.history_file).

Columns: implied_home, implied_draw, implied_away
"""
from __future__ import annotations

import pandas as pd

COLUMNS = ["implied_home", "implied_draw", "implied_away"]


def columns() -> list:
    return list(COLUMNS)


def _implied(oh: float, od: float, oa: float) -> dict:
    raw = [1.0 / oh, 1.0 / od, 1.0 / oa]
    s = sum(raw)
    ih, idr, ia = (x / s for x in raw)
    return {"implied_home": ih, "implied_draw": idr, "implied_away": ia}


def prepare(matches_df: pd.DataFrame, context: dict) -> None:
    """(home, away) -> implied (home, draw, away) probabilities -> context['odds_map']."""
    odds = context.get("sources", {}).get("odds", pd.DataFrame())
    odds_map = {}
    if not odds.empty:
        for r in odds.itertuples(index=False):
            try:
                odds_map[(r.home_team, r.away_team)] = _implied(
                    float(r.odds_home), float(r.odds_draw), float(r.odds_away))
            except (ZeroDivisionError, ValueError, TypeError):
                continue
    context["odds_map"] = odds_map


def _lookup(home: str, away: str, context: dict) -> dict:
    odds_map = context.get("odds_map", {})
    return odds_map.get((home, away),
                        {"implied_home": 0.0, "implied_draw": 0.0, "implied_away": 0.0})


def build(matches_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    n = len(matches_df)
    if "odds_map" not in context:
        prepare(matches_df, context)
    if not context.get("odds_map") or matches_df.empty:
        return pd.DataFrame(0.0, index=range(n), columns=COLUMNS)
    rows = [_lookup(h, a, context) for h, a in
            zip(matches_df["home_team"], matches_df["away_team"])]
    return pd.DataFrame(rows, columns=COLUMNS)


def fixture_features(home: str, away: str, neutral: bool, context: dict) -> dict:
    return _lookup(home, away, context)
