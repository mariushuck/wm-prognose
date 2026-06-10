"""Per-match score report — a predicted final score for every fixture.

Plays the official schedule deterministically with the Poisson goals model: each
match gets its **most-likely exact score**. Knockout draws are resolved (for the
single projected bracket) by the higher expected-goals side, then Elo. Writes
``outputs/match_predictions_<date>.csv`` (+ parquet) covering all 104 matches.

CLI:
    python -m src.predict_fixtures                 # write the report
    python -m src.predict_fixtures --generate-fixtures  # print group round-robin
"""
from __future__ import annotations

import argparse
from datetime import date
from itertools import combinations
from typing import List

import pandas as pd

from . import bracket
from . import goals as goals_mod
from . import paths
from .config import load_yaml
from .features.assemble import (
    build_fixture_features, load_matches, prepare_inference_context,
)

FIXTURES_CONFIG = "fixtures_2026"


def _prediction_context(matches: pd.DataFrame) -> dict:
    """Elo ratings + every enabled feature group's snapshots, for fixture features."""
    return prepare_inference_context(matches)


def _lambdas(bundle: dict, context: dict, home: str, away: str) -> tuple:
    X = build_fixture_features(home, away, True, context, bundle["feature_names"])
    lam_h, lam_a = goals_mod.expected_goals(bundle, X)
    return float(lam_h[0]), float(lam_a[0])


def _make_play_fn(bundle: dict, context: dict, ratings: dict):
    def play(home: str, away: str, knockout: bool):
        lam_h, lam_a = _lambdas(bundle, context, home, away)
        hg, ag = goals_mod.most_likely_score_by_outcome(lam_h, lam_a)
        winner = None
        if hg > ag:
            winner = home
        elif ag > hg:
            winner = away
        elif knockout:  # projected single bracket: break the tie deterministically
            home_key = (lam_h, ratings.get(home, 1500.0))
            away_key = (lam_a, ratings.get(away, 1500.0))
            winner = home if home_key >= away_key else away
        return hg, ag, winner
    return play


def warn_unknown_teams(cfg: dict, ratings: dict) -> List[str]:
    """Return configured teams absent from the results data (name mismatches)."""
    teams = {t for ts in cfg.get("groups", {}).values() for t in ts}
    unknown = sorted(t for t in teams if t not in ratings)
    if unknown:
        print(f"⚠️  {len(unknown)} team(s) not found in results data (will use the "
              f"default Elo {1500.0:.0f}); check spelling vs martj42: {unknown}")
    return unknown


def run(tag: str | None = None) -> pd.DataFrame:
    """Build and persist the full-tournament score report."""
    bundle = goals_mod.load_bundle(tag) if tag else goals_mod.latest_bundle()
    cfg = load_yaml(FIXTURES_CONFIG)
    if not cfg.get("groups") or not cfg.get("group_stage"):
        raise ValueError(f"configs/{FIXTURES_CONFIG}.yaml needs `groups` and `group_stage`.")

    matches = load_matches()
    context = _prediction_context(matches)
    ratings = context.get("ratings", {})
    warn_unknown_teams(cfg, ratings)

    outcome = bracket.play_tournament(cfg, _make_play_fn(bundle, context, ratings), ratings=ratings)

    rows = []
    for match_no in sorted(outcome["results"]):
        r = outcome["results"][match_no]
        lam_h, lam_a = _lambdas(bundle, context, r["home"], r["away"])
        rows.append({
            "match": match_no,
            "stage": r["stage"],
            "group": r["group"] or "",
            "home": r["home"],
            "away": r["away"],
            "xg_home": round(lam_h, 2),
            "xg_away": round(lam_a, 2),
            "pred_home": r["hg"],
            "pred_away": r["ag"],
            "score": f"{r['hg']}:{r['ag']}",
            "winner": r["winner"] or "",
        })
    df = pd.DataFrame(rows)

    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    csv_path = paths.OUTPUTS / f"match_predictions_{stamp}.csv"
    df.to_csv(csv_path, index=False)
    df.to_parquet(paths.OUTPUTS / f"match_predictions_{stamp}.parquet", index=False)
    print(f"Wrote {len(df)} match predictions -> {csv_path}")
    print(f"Projected champion: {outcome['champion']}")
    return df


def generate_group_fixtures() -> None:
    """Print group round-robin pairings derived from the groups (does not overwrite)."""
    cfg = load_yaml(FIXTURES_CONFIG)
    print("group_stage:")
    for g, teams in cfg.get("groups", {}).items():
        for home, away in combinations(teams, 2):
            print(f'  - ["{home}", "{away}"]   # group {g}')


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-match score report")
    parser.add_argument("--generate-fixtures", action="store_true",
                        help="print group round-robin pairings to stdout")
    parser.add_argument("--tag", default=None, help="goals-model tag (default: latest)")
    args = parser.parse_args()

    if args.generate_fixtures:
        generate_group_fixtures()
        return

    df = run(tag=args.tag)
    print("\nGroup-stage sample:")
    print(df[df.stage == "group"].head(8)[["match", "group", "home", "away", "score"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
