"""Monte-Carlo tournament simulation for World Cup 2026.

Pipeline: load the trained model + ``tournament_2026.yaml`` + the shared feature
context, precompute 1X2 probabilities for every possible team pair *once* (all
fixtures are neutral), then sample N tournaments from those probabilities.
Knockout draws are resolved by an Elo-weighted penalty coin flip.

Output: ``outputs/simulation_<date>.parquet`` — per-team advancement and title
probabilities.

Used via ``main.py --simulate-tournament`` or ``simulate.run(...)``.
"""
from __future__ import annotations

from datetime import date
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import model as model_mod
from . import paths
from .config import load_yaml
from .features.assemble import build_context, build_fixture_features, load_matches

Pair = Tuple[str, str]


def _all_teams(groups: Dict[str, List[str]]) -> List[str]:
    return [t for teams in groups.values() for t in teams]


def precompute_pair_probs(
    teams: List[str], context: dict, bundle: dict,
) -> Dict[Pair, Tuple[float, float, float]]:
    """For each unordered pair (a<b): (P(a win), P(draw), P(b win)), neutral venue.

    All pair feature rows are built once and predicted in a single batch.
    """
    pairs = [tuple(sorted(p)) for p in combinations(sorted(set(teams)), 2)]
    feature_names = bundle["feature_names"]

    rows = [
        build_fixture_features(a, b, neutral=True, context=context, feature_names=feature_names)
        for a, b in pairs
    ]
    if not rows:
        return {}
    X = pd.concat(rows, ignore_index=True)
    proba = model_mod.predict_proba(bundle, X)  # columns H/D/A; H = nominal home = a

    out: Dict[Pair, Tuple[float, float, float]] = {}
    for i, (a, b) in enumerate(pairs):
        out[(a, b)] = (float(proba.iloc[i]["H"]),
                       float(proba.iloc[i]["D"]),
                       float(proba.iloc[i]["A"]))
    return out


class Simulator:
    def __init__(self, tournament_cfg: dict, pair_probs, ratings: dict,
                 elo_start: float, seed: int = 0):
        self.cfg = tournament_cfg
        self.groups: Dict[str, List[str]] = tournament_cfg["groups"]
        self.r32_template: List[List[str]] = tournament_cfg["round_of_32"]
        self.points_cfg = tournament_cfg.get("points", {"win": 3, "draw": 1, "loss": 0})
        self.pair_probs = pair_probs
        self.ratings = ratings
        self.elo_start = elo_start
        self.rng = np.random.default_rng(seed)

    # --- match primitives -------------------------------------------------
    def _probs(self, x: str, y: str) -> Tuple[float, float, float]:
        """(P(x win), P(draw), P(y win)) regardless of stored orientation."""
        a, b = sorted((x, y))
        pa, pd_, pb = self.pair_probs.get((a, b), (1 / 3, 1 / 3, 1 / 3))
        return (pa, pd_, pb) if x == a else (pb, pd_, pa)

    def _elo_winner(self, x: str, y: str) -> str:
        """Penalty/coin-flip winner weighted by Elo (for knockout draws)."""
        ex = self.ratings.get(x, self.elo_start)
        ey = self.ratings.get(y, self.elo_start)
        p_x = 1.0 / (1.0 + 10 ** ((ey - ex) / 400.0))
        return x if self.rng.random() < p_x else y

    def _group_match(self, x: str, y: str) -> Tuple[int, int]:
        """Return (points_x, points_y) for one group match."""
        px, pd_, py = self._probs(x, y)
        r = self.rng.random()
        if r < px:
            return self.points_cfg["win"], self.points_cfg["loss"]
        if r < px + pd_:
            return self.points_cfg["draw"], self.points_cfg["draw"]
        return self.points_cfg["loss"], self.points_cfg["win"]

    def _knockout_winner(self, x: str, y: str) -> str:
        px, pd_, py = self._probs(x, y)
        r = self.rng.random()
        if r < px:
            return x
        if r < px + pd_:
            return self._elo_winner(x, y)  # draw -> shootout
        return y

    # --- stages -----------------------------------------------------------
    def _play_group(self, teams: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """Round-robin; return (teams ordered 1st..4th, points dict)."""
        pts = {t: 0 for t in teams}
        for x, y in combinations(teams, 2):
            gx, gy = self._group_match(x, y)
            pts[x] += gx
            pts[y] += gy
        # sort by points desc, random jitter breaks ties
        return sorted(teams, key=lambda t: (pts[t], self.rng.random()), reverse=True), pts

    def _resolve_r32(self, winners, runners, thirds_sorted) -> List[Pair]:
        def slot(token: str) -> str:
            if token.startswith("3rd-"):
                idx = int(token.split("-")[1]) - 1
                return thirds_sorted[idx] if idx < len(thirds_sorted) else thirds_sorted[-1]
            pos, grp = token[0], token[1:]
            return winners[grp] if pos == "1" else runners[grp]

        return [(slot(a), slot(b)) for a, b in self.r32_template]

    def simulate_once(self) -> dict:
        """Run one tournament; return {team: highest_stage_index}.

        Stage indices: 1=group-advance, 2=R16, 3=QF, 4=SF, 5=Final, 6=Winner.
        """
        winners, runners = {}, {}
        thirds = []  # (points, team)
        reached: Dict[str, int] = {}

        for grp, teams in self.groups.items():
            order, pts = self._play_group(teams)
            winners[grp], runners[grp], third = order[0], order[1], order[2]
            thirds.append((pts[third], third))

        best_thirds = [t for _, t in sorted(thirds, key=lambda kv: (kv[0], self.rng.random()),
                                            reverse=True)]
        n_best = int(self.cfg.get("format", {}).get("best_thirds", 8))
        best_thirds = best_thirds[:n_best]

        for grp in self.groups:
            reached[winners[grp]] = 1
            reached[runners[grp]] = 1
        for t in best_thirds:
            reached[t] = 1

        bracket = self._resolve_r32(winners, runners, best_thirds)
        stage_names = {1: 2, 2: 3, 3: 4, 4: 5}  # round index -> stage reached by winner

        # knockout rounds
        round_pairs = bracket
        round_idx = 1
        while len(round_pairs) >= 1:
            survivors = []
            for x, y in round_pairs:
                w = self._knockout_winner(x, y)
                survivors.append(w)
                reached[w] = max(reached.get(w, 0), stage_names.get(round_idx, 6))
            if len(survivors) == 1:
                reached[survivors[0]] = 6  # champion
                break
            round_pairs = [(survivors[i], survivors[i + 1]) for i in range(0, len(survivors), 2)]
            round_idx += 1
        return reached


def run(tag: str | None = None, n_sims: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Run the Monte-Carlo simulation and write the results parquet."""
    bundle = model_mod.load_bundle(tag) if tag else model_mod.latest_bundle()
    tournament_cfg = load_yaml("tournament_2026")
    if not tournament_cfg.get("groups"):
        raise ValueError("tournament_2026.yaml has no groups configured.")

    matches = load_matches()
    context = build_context(matches)
    # basic.build populates form/h2h snapshots used by fixture features.
    from .features import basic
    basic.build(matches, context)

    teams = _all_teams(tournament_cfg["groups"])
    pair_probs = precompute_pair_probs(teams, context, bundle)

    sim = Simulator(tournament_cfg, pair_probs, context.get("ratings", {}),
                    context.get("elo_start", 1500.0), seed=seed)

    # stage index -> probability column
    cols = {1: "p_advance", 2: "p_round_of_16", 3: "p_quarter",
            4: "p_semi", 5: "p_final", 6: "p_winner"}
    counts = {t: {s: 0 for s in cols} for t in teams}

    for _ in range(n_sims):
        reached = sim.simulate_once()
        for team, stage in reached.items():
            # a team that reached stage S also reached every earlier stage
            for s in cols:
                if stage >= s:
                    counts[team][s] += 1

    team_group = {t: grp for grp, ts in tournament_cfg["groups"].items() for t in ts}
    records = []
    for team in teams:
        rec = {"team": team, "group": team_group[team]}
        for s, col in cols.items():
            rec[col] = counts[team][s] / n_sims
        records.append(rec)

    df = pd.DataFrame(records).sort_values("p_winner", ascending=False).reset_index(drop=True)

    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = paths.OUTPUTS / f"simulation_{date.today().isoformat()}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Simulated {n_sims:,} tournaments -> {out_path}")
    return df


if __name__ == "__main__":  # pragma: no cover
    result = run()
    print(result.head(12).to_string(index=False))
