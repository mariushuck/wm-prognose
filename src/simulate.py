"""Monte-Carlo tournament simulation (goals-based, official bracket).

Loads the Poisson goals model, precomputes expected goals (lambda_home,
lambda_away) for every ordered team pair once, then samples N tournaments through
the shared bracket engine ([bracket.py]) using the real WC-2026 schedule. Each
simulated match draws a scoreline from independent Poissons, so group standings
use genuine goal-difference tiebreakers; knockout draws go to an Elo-weighted
penalty flip.

Output: ``outputs/simulation_<date>.parquet`` — per-team advancement and title
probabilities.

Used via ``main.py --simulate-tournament`` or ``simulate.run(...)``.
"""
from __future__ import annotations

from datetime import date
from itertools import permutations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import bracket
from . import goals as goals_mod
from . import paths
from .config import load_yaml
from .features.assemble import (
    build_fixture_features, load_matches, prepare_inference_context,
)

FIXTURES_CONFIG = "fixtures_2026"
OrderedPair = Tuple[str, str]


def _all_teams(groups: Dict[str, List[str]]) -> List[str]:
    return [t for teams in groups.values() for t in teams]


def precompute_pair_goals(teams: List[str], context: dict,
                          bundle: dict) -> Dict[OrderedPair, Tuple[float, float]]:
    """(home, away) -> (lambda_home, lambda_away) for every ordered pair, in one batch."""
    pairs = list(permutations(sorted(set(teams)), 2))
    if not pairs:
        return {}
    rows = [build_fixture_features(h, a, True, context, bundle["feature_names"]) for h, a in pairs]
    X = pd.concat(rows, ignore_index=True)
    lam_home, lam_away = goals_mod.expected_goals(bundle, X)
    return {pair: (float(lam_home[i]), float(lam_away[i])) for i, pair in enumerate(pairs)}


def _make_play_fn(pair_goals, ratings: dict, elo_start: float, rng: np.random.Generator):
    def _elo_winner(x: str, y: str) -> str:
        ex, ey = ratings.get(x, elo_start), ratings.get(y, elo_start)
        p_x = 1.0 / (1.0 + 10 ** ((ey - ex) / 400.0))
        return x if rng.random() < p_x else y

    def play(home: str, away: str, knockout: bool):
        lam_h, lam_a = pair_goals.get((home, away), (1.3, 1.1))
        hg, ag = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
        winner = None
        if knockout:
            if hg > ag:
                winner = home
            elif ag > hg:
                winner = away
            else:
                winner = _elo_winner(home, away)  # penalty shootout
        return hg, ag, winner
    return play


# stage index -> probability column
STAGE_COLS = {1: "p_advance", 2: "p_round_of_16", 3: "p_quarter",
              4: "p_semi", 5: "p_final", 6: "p_winner"}


def run(tag: str | None = None, n_sims: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Run the Monte-Carlo simulation and write the results parquet."""
    bundle = goals_mod.load_bundle(tag) if tag else goals_mod.latest_bundle()
    cfg = load_yaml(FIXTURES_CONFIG)
    if not cfg.get("groups") or not cfg.get("group_stage"):
        raise ValueError(f"configs/{FIXTURES_CONFIG}.yaml needs `groups` and `group_stage`.")

    matches = load_matches()
    context = prepare_inference_context(matches)  # ratings + all feature snapshots
    ratings = context.get("ratings", {})
    elo_start = context.get("elo_start", 1500.0)

    teams = _all_teams(cfg["groups"])
    pair_goals = precompute_pair_goals(teams, context, bundle)

    rng = np.random.default_rng(seed)
    play_fn = _make_play_fn(pair_goals, ratings, elo_start, rng)

    counts = {t: {s: 0 for s in STAGE_COLS} for t in teams}
    for _ in range(n_sims):
        reached = bracket.play_tournament(cfg, play_fn, ratings=ratings, rng=rng)["reached"]
        for team, stage in reached.items():
            for s in STAGE_COLS:
                if stage >= s:
                    counts[team][s] += 1

    team_group = {t: g for g, ts in cfg["groups"].items() for t in ts}
    records = [
        {"team": t, "group": team_group[t],
         **{col: counts[t][s] / n_sims for s, col in STAGE_COLS.items()}}
        for t in teams
    ]
    df = pd.DataFrame(records).sort_values("p_winner", ascending=False).reset_index(drop=True)

    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = paths.OUTPUTS / f"simulation_{date.today().isoformat()}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Simulated {n_sims:,} tournaments -> {out_path}")
    return df


if __name__ == "__main__":  # pragma: no cover
    print(run().head(12).to_string(index=False))
