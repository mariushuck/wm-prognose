"""Layer 1 — internal Elo ratings computed from the results table.

World-Football-Elo style: K scaled by margin of victory, home advantage applied
to the expected score. Processing is strictly chronological so each match gets a
*pre-game* rating snapshot (no leakage) and the final dict holds current ratings
for the simulator.

Public API:
    compute(results_df, cfg) -> (results_with_elo, final_ratings_dict)
    EloTable.from_results(results_df, cfg)   # incremental access for the simulator
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd

# Columns appended to the results table.
ELO_COLUMNS = ["home_elo_pre", "away_elo_pre", "elo_diff"]


def _expected(rating_a: float, rating_b: float) -> float:
    """Expected score of A vs B per the logistic Elo curve."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _goal_diff_multiplier(goal_diff: int) -> float:
    """World-Football-Elo margin-of-victory multiplier."""
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11 + n) / 8.0


class EloTable:
    """Running Elo ratings keyed by team name."""

    def __init__(self, cfg: dict):
        self.start = float(cfg.get("start_rating", 1500.0))
        self.k = float(cfg.get("k_factor", 40.0))
        self.home_adv = float(cfg.get("home_advantage", 65.0))
        self._ratings: Dict[str, float] = defaultdict(lambda: self.start)

    def rating(self, team: str) -> float:
        return self._ratings[team]

    def update(self, home: str, away: str, hs: int, as_: int, neutral: bool) -> Tuple[float, float]:
        """Apply one match; return the *pre-game* (home, away) ratings."""
        rh, ra = self._ratings[home], self._ratings[away]
        adv = 0.0 if neutral else self.home_adv
        exp_home = _expected(rh + adv, ra)

        if hs > as_:
            w_home = 1.0
        elif hs < as_:
            w_home = 0.0
        else:
            w_home = 0.5

        mult = _goal_diff_multiplier(hs - as_)
        delta = self.k * mult * (w_home - exp_home)
        self._ratings[home] = rh + delta
        self._ratings[away] = ra - delta
        return rh, ra

    @property
    def ratings(self) -> Dict[str, float]:
        return dict(self._ratings)

    @classmethod
    def from_results(cls, results_df: pd.DataFrame, cfg: dict) -> "EloTable":
        """Replay all matches chronologically and return the final table."""
        table = cls(cfg)
        for row in results_df.sort_values("date").itertuples(index=False):
            table.update(row.home_team, row.away_team,
                         int(row.home_score), int(row.away_score), bool(row.neutral))
        return table


def compute(results_df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Append pre-game Elo columns to results and return final ratings.

    Returns (df_with_elo_columns, final_ratings). If results are empty, returns
    the input plus empty Elo columns and an empty ratings dict.
    """
    df = results_df.sort_values("date").reset_index(drop=True).copy()
    if df.empty:
        for c in ELO_COLUMNS:
            df[c] = pd.Series(dtype="float64")
        return df, {}

    table = EloTable(cfg)
    home_pre, away_pre = [], []
    for row in df.itertuples(index=False):
        rh, ra = table.update(row.home_team, row.away_team,
                              int(row.home_score), int(row.away_score), bool(row.neutral))
        home_pre.append(rh)
        away_pre.append(ra)

    df["home_elo_pre"] = home_pre
    df["away_elo_pre"] = away_pre
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    return df, table.ratings


if __name__ == "__main__":  # pragma: no cover - manual smoke
    from ..config import source_cfg
    from . import results as results_mod

    res = results_mod.load(source_cfg("results"))
    _, ratings = compute(res, source_cfg("elo_internal"))
    top = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("Top 10 by internal Elo:")
    for team, r in top:
        print(f"  {team:<20} {r:7.1f}")
