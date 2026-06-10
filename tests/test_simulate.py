"""Simulation play-fn + bracket integration (no model / no network)."""
from __future__ import annotations

import numpy as np

from src import bracket
from src.simulate import _make_play_fn


def test_play_fn_samples_valid_scores_and_knockout_winner(mini_fixtures):
    rng = np.random.default_rng(0)
    play = _make_play_fn(pair_goals={}, ratings={}, elo_start=1500.0, rng=rng)
    hg, ag, winner = play("A1", "B2", knockout=True)
    assert hg >= 0 and ag >= 0
    assert winner in {"A1", "B2"}


def test_group_match_can_draw(mini_fixtures):
    rng = np.random.default_rng(0)
    play = _make_play_fn(pair_goals={}, ratings={}, elo_start=1500.0, rng=rng)
    # group matches may return winner=None (draws allowed)
    winners = [play("A1", "A2", knockout=False)[2] for _ in range(20)]
    assert any(w is None for w in winners)


def test_one_champion_per_simulation(mini_fixtures):
    rng = np.random.default_rng(7)
    play = _make_play_fn(pair_goals={}, ratings={}, elo_start=1500.0, rng=rng)
    teams = [t for ts in mini_fixtures["groups"].values() for t in ts]
    n = 100
    champions = 0
    for _ in range(n):
        out = bracket.play_tournament(mini_fixtures, play, ratings={}, rng=rng)
        champ = [t for t, s in out["reached"].items() if s == 6]
        assert len(champ) == 1
        champions += 1
        # 4 teams advance from the groups each run
        assert len([t for t, s in out["reached"].items() if s >= 1]) == 4
    assert champions == n
