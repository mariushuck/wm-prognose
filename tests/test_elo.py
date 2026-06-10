"""Deterministic Elo behaviour on the offline sample."""
from __future__ import annotations

from src.data_sources import elo_internal

CFG = {"start_rating": 1500.0, "k_factor": 40.0, "home_advantage": 65.0}


def test_compute_adds_columns_and_conserves_initial_total(sample_results):
    df, ratings = elo_internal.compute(sample_results, CFG)
    for col in elo_internal.ELO_COLUMNS:
        assert col in df.columns
    # Elo is zero-sum per match -> total rating stays at n_teams * start.
    n_teams = len({*sample_results.home_team, *sample_results.away_team})
    assert abs(sum(ratings.values()) - n_teams * CFG["start_rating"]) < 1e-6


def test_winner_gains_rating(sample_results):
    _, ratings = elo_internal.compute(sample_results, CFG)
    # Germany beat Spain twice and outscored most opponents; Spain lost often.
    assert ratings["Germany"] > ratings["Spain"]


def test_empty_results_safe():
    import pandas as pd
    empty = pd.DataFrame(columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral",
    ])
    df, ratings = elo_internal.compute(empty, CFG)
    assert ratings == {}
    assert list(elo_internal.ELO_COLUMNS) == [c for c in elo_internal.ELO_COLUMNS if c in df.columns]


def test_goal_diff_multiplier_monotonic():
    g1 = elo_internal._goal_diff_multiplier(1)
    g2 = elo_internal._goal_diff_multiplier(2)
    g5 = elo_internal._goal_diff_multiplier(5)
    assert g1 < g2 < g5
