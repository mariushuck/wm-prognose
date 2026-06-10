"""Bracket engine: token resolution + full-tournament play (hermetic)."""
from __future__ import annotations

from src import bracket


def _home_always_wins(home, away, knockout):
    # Deterministic: the nominal-home team wins 2:0 every match.
    return 2, 0, (home if knockout else None)


def test_full_mini_tournament_resolves(mini_fixtures):
    out = bracket.play_tournament(mini_fixtures, _home_always_wins)

    # 12 group + 2 semis + 1 third-place + 1 final = 16 matches.
    assert len(out["results"]) == 16

    # Group RR with home-always-wins ranks teams by how often they are listed first.
    assert out["winners"]["A"] == "A1"
    assert out["runners"]["A"] == "A2"

    # 2 groups x top 2 (best_thirds = 0) = 4 advancers.
    advancers = [t for t, s in out["reached"].items() if s >= 1]
    assert len(advancers) == 4

    # Exactly one champion at stage 6; the final is the last match.
    assert out["champion"] == "A1"
    assert out["reached"]["A1"] == 6


def test_w_and_l_tokens_reference_earlier_matches(mini_fixtures):
    out = bracket.play_tournament(mini_fixtures, _home_always_wins)
    final = out["results"][16]
    third = out["results"][15]
    # Final contestants are the semi winners; third-place contestants are the losers.
    assert final["home"] == out["results"][13]["winner"]
    assert final["away"] == out["results"][14]["winner"]
    assert third["home"] == out["results"][13]["loser"]


def test_knockout_winner_fallback_when_play_fn_returns_none(mini_fixtures):
    # play_fn reports a draw with winner=None -> engine must still pick a winner.
    out = bracket.play_tournament(mini_fixtures, lambda h, a, ko: (1, 1, None))
    assert out["champion"]  # non-empty
    for r in out["results"].values():
        if r["stage"] != "group":
            assert r["winner"] in (r["home"], r["away"])
