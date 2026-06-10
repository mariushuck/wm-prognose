"""Smoke tests: configs load, loaders honor the contract, features build."""
from __future__ import annotations

import pandas as pd

from src import config
from src.data_sources import fifa_ranking, odds, statsbomb, transfermarkt, weather
from src.features import (
    advanced_stats, assemble, basic, context as context_feat, market, ranking, squad,
)


def test_configs_load():
    assert config.data_sources()  # non-empty mapping
    assert "basic" in config.feature_groups()
    assert config.is_enabled("results") is True


DISABLED_LOADERS = [
    (odds, odds.ODDS_COLUMNS),
    (fifa_ranking, fifa_ranking.FIFA_COLUMNS),
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
    assert context["form"] and "Germany" in context["form"]

    fixture = basic.fixture_features("Germany", "Spain", neutral=True, context=context)
    assert set(fixture) == set(basic.COLUMNS)
    assert fixture["neutral"] == 1


# Layer 2–5 feature modules emit their fixed columns (zero-filled) even with no data,
# and fixture_features returns the same keys — keeping train/serve aligned.
DATA_FEATURE_MODULES = [ranking, squad, market, advanced_stats, context_feat]


def test_data_feature_groups_emit_fixed_columns_when_empty(sample_results):
    for module in DATA_FEATURE_MODULES:
        ctx = {"sources": {}}
        block = module.build(sample_results, ctx)
        assert list(block.columns) == module.COLUMNS
        assert len(block) == len(sample_results)
        fx = module.fixture_features("Germany", "Spain", True, ctx)
        assert set(fx) == set(module.COLUMNS)


def test_only_basic_enabled_by_default():
    mods = assemble._enabled_modules()
    assert basic in mods
    assert ranking not in mods       # Layer 2 groups default off
    assert market not in mods
