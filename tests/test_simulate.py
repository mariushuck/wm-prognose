"""Simulation logic on a mini bracket (no model / no network needed)."""
from __future__ import annotations

from src.simulate import Simulator


def _make_sim(mini_tournament, seed=0):
    # Empty pair_probs -> Simulator falls back to (1/3, 1/3, 1/3) per match.
    return Simulator(mini_tournament, pair_probs={}, ratings={}, elo_start=1500.0, seed=seed)


def test_probs_fallback_sums_to_one(mini_tournament):
    sim = _make_sim(mini_tournament)
    px, pd_, py = sim._probs("T01", "T02")
    assert abs(px + pd_ + py - 1.0) < 1e-9


def test_single_run_has_one_champion_and_correct_advancers(mini_tournament):
    sim = _make_sim(mini_tournament)
    reached = sim.simulate_once()

    champions = [t for t, s in reached.items() if s == 6]
    assert len(champions) == 1

    advancers = [t for t, s in reached.items() if s >= 1]
    # 4 groups x top 2 = 8 teams advance (best_thirds = 0)
    assert len(advancers) == 8


def test_knockout_winner_is_a_participant(mini_tournament):
    sim = _make_sim(mini_tournament)
    w = sim._knockout_winner("T01", "T02")
    assert w in {"T01", "T02"}


def test_aggregate_title_prob_sums_to_one(mini_tournament):
    sim = _make_sim(mini_tournament, seed=7)
    teams = [t for ts in mini_tournament["groups"].values() for t in ts]
    n = 200
    wins = {t: 0 for t in teams}
    for _ in range(n):
        reached = sim.simulate_once()
        for t, s in reached.items():
            if s == 6:
                wins[t] += 1
    assert sum(wins.values()) == n  # exactly one champion per simulation
