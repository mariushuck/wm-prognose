"""Tournament bracket engine — resolves the official WC-2026 schedule.

Drives the full 104-match schedule from ``configs/fixtures_2026.yaml``: plays the
72 group matches, computes standings, then resolves every knockout match by its
slot tokens. Model-agnostic — the caller supplies a ``play_fn``:

    play_fn(home, away, knockout) -> (home_goals, away_goals, winner)

so the *same* engine backs the deterministic score report (most-likely scores)
and the Monte-Carlo simulator (sampled scores). Token grammar:

    "1X" / "2X"   winner / runner-up of group X
    "3rd-N"       N-th best third-placed team (1 = best)
    "W<match#>"   winner of an earlier match
    "L<match#>"   loser of an earlier match (used by the third-place play-off)

Match numbers are assigned sequentially in stage order, matching the comments in
the fixtures file (group 1-72, R32 73-88, R16 89-96, QF 97-100, SF 101-102,
3rd 103, final 104).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# (config key, stage label, is_knockout)
STAGES = [
    ("group_stage", "group", False),
    ("round_of_32", "round_of_32", True),
    ("round_of_16", "round_of_16", True),
    ("quarter_finals", "quarter", True),
    ("semi_finals", "semi", True),
    ("third_place", "third_place", True),
    ("final", "final", True),
]

# Furthest stage a team *participated* in -> ladder index used for probabilities.
STAGE_REACHED = {
    "round_of_32": 1,   # advanced from the group
    "round_of_16": 2,
    "quarter": 3,
    "semi": 4,
    "final": 5,
}
CHAMPION = 6

PlayFn = Callable[[str, str, bool], Tuple[int, int, Optional[str]]]


def _apply_group(table: Dict[str, dict], home: str, away: str,
                 hg: int, ag: int, pts_cfg: dict) -> None:
    for team, gf, ga in ((home, hg, ag), (away, ag, hg)):
        s = table[team]
        s["gf"] += gf
        s["gd"] += gf - ga
        if gf > ga:
            s["pts"] += pts_cfg.get("win", 3)
        elif gf == ga:
            s["pts"] += pts_cfg.get("draw", 1)
        else:
            s["pts"] += pts_cfg.get("loss", 0)


def _rank_key(team: str, stats: dict, ratings: Optional[dict],
              rng: Optional[np.random.Generator]) -> tuple:
    tiebreak = (ratings.get(team, 0.0) if ratings else 0.0)
    tiebreak += (rng.random() if rng is not None else 0.0)
    return (stats["pts"], stats["gd"], stats["gf"], tiebreak)


def play_tournament(cfg: dict, play_fn: PlayFn, *,
                    ratings: Optional[dict] = None,
                    rng: Optional[np.random.Generator] = None) -> dict:
    """Resolve and play the whole tournament.

    Returns a dict with: ``results`` (match_no -> record), ``winners``,
    ``runners``, ``best_thirds``, ``reached`` (team -> furthest stage index) and
    ``champion``.
    """
    groups: Dict[str, List[str]] = cfg["groups"]
    team_group = {t: g for g, ts in groups.items() for t in ts}
    pts_cfg = cfg.get("points", {"win": 3, "draw": 1, "loss": 0})
    n_best = int(cfg.get("format", {}).get("best_thirds", 8))

    results: Dict[int, dict] = {}
    reached: Dict[str, int] = {}
    standings = {g: {t: {"pts": 0, "gd": 0, "gf": 0} for t in ts} for g, ts in groups.items()}

    match_no = 0

    # --- group stage ---
    for home, away in cfg.get("group_stage", []):
        match_no += 1
        hg, ag, _ = play_fn(home, away, False)
        results[match_no] = {"stage": "group", "group": team_group.get(home),
                             "home": home, "away": away, "hg": hg, "ag": ag, "winner": None}
        _apply_group(standings[team_group[home]], home, away, hg, ag, pts_cfg)

    winners, runners, thirds = {}, {}, []
    for g, table in standings.items():
        order = sorted(table, key=lambda t: _rank_key(t, table[t], ratings, rng), reverse=True)
        winners[g], runners[g] = order[0], order[1]
        thirds.append((order[2], table[order[2]]))
    best_thirds = [t for t, _ in sorted(
        thirds, key=lambda kv: _rank_key(kv[0], kv[1], ratings, rng), reverse=True)][:n_best]

    for g in groups:
        reached[winners[g]] = 1
        reached[runners[g]] = 1
    for t in best_thirds:
        reached[t] = 1

    def resolve(token: str) -> str:
        if token.startswith("3rd-"):
            idx = int(token[4:]) - 1
            return best_thirds[idx] if idx < len(best_thirds) else best_thirds[-1]
        if token[0] in "12" and token[1:] in groups:
            return winners[token[1:]] if token[0] == "1" else runners[token[1:]]
        if token[0] == "W":
            return results[int(token[1:])]["winner"]
        if token[0] == "L":
            return results[int(token[1:])]["loser"]
        raise ValueError(f"Unrecognized bracket token: {token!r}")

    # --- knockout stages ---
    for key, stage_name, _ko in STAGES[1:]:
        for pair in cfg.get(key, []) or []:
            match_no += 1
            home, away = resolve(pair[0]), resolve(pair[1])
            hg, ag, winner = play_fn(home, away, True)
            if winner is None:  # safety: knockout must have a winner
                winner = home if hg >= ag else away
            loser = away if winner == home else home
            results[match_no] = {"stage": stage_name, "group": None,
                                 "home": home, "away": away, "hg": hg, "ag": ag,
                                 "winner": winner, "loser": loser}
            r = STAGE_REACHED.get(stage_name)
            if r:
                reached[home] = max(reached.get(home, 0), r)
                reached[away] = max(reached.get(away, 0), r)

    champion = results[match_no]["winner"]  # final is the last match
    reached[champion] = CHAMPION

    return {
        "results": results,
        "winners": winners,
        "runners": runners,
        "best_thirds": best_thirds,
        "reached": reached,
        "champion": champion,
        "team_group": team_group,
    }
