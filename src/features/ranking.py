"""Ranking feature group (Layer 2) — FIFA world ranking.

Consumes the ``fifa_ranking`` cache. For training it joins each team's ranking
*as of the match date* (no leakage); for fixtures it uses the latest snapshot.
Contributes zero-filled columns when the cache is empty.

Columns:
    rank_diff          away_rank - home_rank  (positive ⇒ home ranked higher/better)
    fifa_points_diff   home_points - away_points
"""
from __future__ import annotations

import pandas as pd

COLUMNS = ["rank_diff", "fifa_points_diff"]


def columns() -> list:
    return list(COLUMNS)


def _fifa(context: dict) -> pd.DataFrame:
    return context.get("sources", {}).get("fifa_ranking", pd.DataFrame())


def prepare(matches_df: pd.DataFrame, context: dict) -> None:
    """Latest rank/points per team -> context['fifa_latest']."""
    fifa = _fifa(context)
    if fifa.empty:
        context["fifa_latest"] = {}
        return
    latest = fifa.dropna(subset=["rank_date"]).sort_values("rank_date").groupby("team").last()
    context["fifa_latest"] = {
        t: (float(r["rank"]), float(r["points"])) for t, r in latest.iterrows()
    }


def _asof(matches: pd.DataFrame, fifa: pd.DataFrame, team_col: str) -> pd.DataFrame:
    left = (matches[["date", team_col]].rename(columns={team_col: "team"})
            .reset_index().sort_values("date"))
    right = (fifa.dropna(subset=["rank_date"])[["rank_date", "team", "rank", "points"]]
             .sort_values("rank_date"))
    merged = pd.merge_asof(left, right, left_on="date", right_on="rank_date",
                           by="team", direction="backward")
    return merged.set_index("index")[["rank", "points"]].reindex(matches.index)


def build(matches_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    n = len(matches_df)
    fifa = _fifa(context)
    if fifa.empty or matches_df.empty:
        return pd.DataFrame(0.0, index=range(n), columns=COLUMNS)

    m = matches_df.reset_index(drop=True)
    home = _asof(m, fifa, "home_team")
    away = _asof(m, fifa, "away_team")
    out = pd.DataFrame({
        "rank_diff": (away["rank"] - home["rank"]),
        "fifa_points_diff": (home["points"] - away["points"]),
    }).fillna(0.0)
    return out.reset_index(drop=True)


def fixture_features(home: str, away: str, neutral: bool, context: dict) -> dict:
    snap = context.get("fifa_latest", {})
    if home not in snap or away not in snap:
        return {"rank_diff": 0.0, "fifa_points_diff": 0.0}
    (rh, ph), (ra, pa) = snap[home], snap[away]
    return {"rank_diff": ra - rh, "fifa_points_diff": ph - pa}
