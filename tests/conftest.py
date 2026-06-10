"""Shared fixtures. Provides an offline results sample so tests need no network."""
from __future__ import annotations

from itertools import combinations

import pandas as pd
import pytest


@pytest.fixture
def sample_results() -> pd.DataFrame:
    """A small synthetic results table matching the martj42 schema."""
    rows = [
        # date, home, away, hs, as, tournament, city, country, neutral
        ("2018-06-01", "Germany", "France", 2, 1, "Friendly", "Berlin", "Germany", False),
        ("2018-06-05", "Brazil", "Spain", 1, 1, "Friendly", "Rio", "Brazil", False),
        ("2018-06-10", "France", "Brazil", 0, 2, "Friendly", "Paris", "France", False),
        ("2018-06-15", "Spain", "Germany", 1, 3, "Friendly", "Madrid", "Spain", False),
        ("2018-07-01", "Germany", "Brazil", 2, 2, "WorldCup", "Moscow", "Russia", True),
        ("2018-07-05", "France", "Spain", 3, 0, "WorldCup", "Moscow", "Russia", True),
        ("2019-06-01", "Brazil", "Germany", 1, 0, "Friendly", "Rio", "Brazil", False),
        ("2019-06-05", "Spain", "France", 2, 2, "Friendly", "Madrid", "Spain", False),
        ("2019-06-10", "Germany", "Spain", 4, 1, "Friendly", "Munich", "Germany", False),
        ("2019-06-15", "France", "Germany", 1, 1, "Friendly", "Lyon", "France", False),
        ("2020-06-01", "Brazil", "France", 2, 0, "Friendly", "Sao Paulo", "Brazil", False),
        ("2020-06-05", "Spain", "Brazil", 0, 1, "Friendly", "Seville", "Spain", False),
    ]
    df = pd.DataFrame(rows, columns=[
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral",
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def mini_fixtures() -> dict:
    """An 8-team / 2-group mini schedule (group RR + semis/3rd/final) for the engine."""
    groups = {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}
    group_stage = []
    for teams in groups.values():
        group_stage += [list(p) for p in combinations(teams, 2)]   # 12 matches (1-12)
    return {
        "format": {"best_thirds": 0},
        "points": {"win": 3, "draw": 1, "loss": 0},
        "groups": groups,
        "group_stage": group_stage,
        "semi_finals": [["1A", "2B"], ["1B", "2A"]],   # matches 13, 14
        "third_place": [["L13", "L14"]],               # match 15
        "final": [["W13", "W14"]],                     # match 16
    }
