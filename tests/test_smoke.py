"""Smoke tests: configs load, loaders honor the contract, features build."""
from __future__ import annotations

import pandas as pd

from src import config
from src.data_sources import (
    fbref, fifa_ranking, odds, statsbomb, transfermarkt, weather,
)
from src.features import advanced_stats, basic, context as context_feat, market


def test_configs_load():
    assert config.data_sources()  # non-empty mapping
    assert "basic" in config.feature_groups()
    assert config.is_enabled("results") is True


DISABLED_LOADERS = [
    (odds, odds.ODDS_COLUMNS),
    (fifa_ranking, fifa_ranking.FIFA_COLUMNS),
    (fbref, fbref.FBREF_COLUMNS),
    (transfermarkt, transfermarkt.TM_COLUMNS),
    (statsbomb, statsbomb.STATSBOMB_COLUMNS),
    (weather, weather.WEATHER_COLUMNS),
]


def test_disabled_loaders_return_empty_with_columns():
    for module, columns in DISABLED_LOADERS:
        df = module.load({"enabled": False})
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == list(columns)
        assert df.empty


def test_basic_features_build(sample_results):
    from src.data_sources import elo_internal
    df, ratings = elo_internal.compute(sample_results, config.source_cfg("elo_internal"))
    df["result"] = ["H" if h > a else ("A" if h < a else "D")
                    for h, a in zip(df.home_score, df.away_score)]

    context = {"ratings": ratings, "elo_start": 1500.0}
    feats = basic.build(df, context)
    assert list(feats.columns) == basic.COLUMNS
    assert len(feats) == len(df)
    # snapshots were populated for the fixture path
    assert context["form"] and "Germany" in context["form"]

    fixture = basic.fixture_features("Germany", "Spain", neutral=True, context=context)
    assert set(fixture) == set(basic.COLUMNS)
    assert fixture["neutral"] == 1


def test_stub_feature_groups_contribute_no_columns(sample_results):
    ctx = {}
    for module in (market, advanced_stats, context_feat):
        block = module.build(sample_results, ctx)
        assert block.shape[1] == 0
        assert module.fixture_features("A", "B", True, ctx) == {}
