"""Layer 2–5 feature builders with injected synthetic caches (hermetic)."""
from __future__ import annotations

import pandas as pd

from src.features import advanced_stats, market, ranking, squad


def _matches(home: str, away: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-06-01"]),
        "home_team": [home], "away_team": [away],
    })


def test_ranking_sign_and_parity():
    fifa = pd.DataFrame({
        "rank_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "team": ["Spain", "Panama"], "rank": [2.0, 40.0], "points": [1850.0, 1100.0],
    })
    ctx = {"sources": {"fifa_ranking": fifa}}
    ranking.prepare(_matches("Spain", "Panama"), ctx)

    block = ranking.build(_matches("Spain", "Panama"), ctx)
    # Spain higher-ranked at home -> rank_diff (away_rank - home_rank) > 0, points > 0.
    assert block.loc[0, "rank_diff"] > 0
    assert block.loc[0, "fifa_points_diff"] > 0

    fx = ranking.fixture_features("Spain", "Panama", True, ctx)
    assert fx["rank_diff"] == block.loc[0, "rank_diff"]


def test_squad_value_sign():
    tm = pd.DataFrame({"team": ["England", "Panama"],
                       "market_value": [1.3e9, 40e6], "avg_age": [25.0, 28.0]})
    ctx = {"sources": {"transfermarkt": tm}}
    squad.prepare(_matches("England", "Panama"), ctx)
    fx = squad.fixture_features("England", "Panama", True, ctx)
    assert fx["market_value_log_diff"] > 0      # richer squad at home
    assert fx["avg_age_diff"] < 0               # younger at home


def test_market_implied_probs_sum_to_one():
    odds = pd.DataFrame({
        "date": pd.to_datetime(["2025-06-01"]),
        "home_team": ["Brazil"], "away_team": ["Haiti"],
        "odds_home": [1.3], "odds_draw": [5.0], "odds_away": [9.0],
    })
    ctx = {"sources": {"odds": odds}}
    market.prepare(_matches("Brazil", "Haiti"), ctx)
    fx = market.fixture_features("Brazil", "Haiti", True, ctx)
    assert abs(sum(fx.values()) - 1.0) < 1e-9
    assert fx["implied_home"] > fx["implied_away"]   # heavy favourite


def test_advanced_stats_xg_diff():
    sb = pd.DataFrame({
        "date": pd.to_datetime(["2022-12-01", "2022-12-01"]),
        "team": ["Argentina", "Australia"], "opponent": ["Australia", "Argentina"],
        "xg": [2.4, 0.7], "xa": [1.8, 0.5],
    })
    ctx = {"sources": {"statsbomb": sb}}
    advanced_stats.prepare(_matches("Argentina", "Australia"), ctx)
    fx = advanced_stats.fixture_features("Argentina", "Australia", True, ctx)
    assert fx["xg_diff"] > 0 and fx["xa_diff"] > 0
