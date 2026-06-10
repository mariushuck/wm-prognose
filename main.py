"""WM-Prognose v2 — end-to-end entry point.

Default run:            ingest -> train goals model -> per-match score report -> simulate.
--predict-scores      : write the per-match score report only.
--simulate-tournament : run the tournament simulation only.
--no-train            : skip training, reuse the latest existing goals model.

Examples:
    python main.py                          # full pipeline
    python main.py --predict-scores         # scores for every match
    python main.py --simulate-tournament    # advancement probabilities only
    python main.py --tag v2_full --sims 5000
"""
from __future__ import annotations

import argparse

from src import goals as goals_mod
from src import ingestion, predict_fixtures, simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="WM-Prognose v2 pipeline")
    parser.add_argument("--predict-scores", action="store_true",
                        help="write the per-match score report only")
    parser.add_argument("--simulate-tournament", action="store_true",
                        help="run only the tournament simulation")
    parser.add_argument("--no-train", action="store_true",
                        help="skip training; reuse the latest goals model")
    parser.add_argument("--force", action="store_true",
                        help="force re-ingestion of all sources")
    parser.add_argument("--tag", default="v2_full", help="model tag")
    parser.add_argument("--sims", type=int, default=2000, help="number of simulations")
    args = parser.parse_args()

    report_only = args.predict_scores
    sim_only = args.simulate_tournament
    full_run = not (report_only or sim_only)

    trained_this_run = False
    if full_run:
        print("== Ingestion ==")
        summary = ingestion.run(force=args.force)
        print(f"  matches: {summary.get('matches')}")
        if not args.no_train:
            print("== Training goals model ==")
            goals_mod.train(tag=args.tag)
            trained_this_run = True

    tag = args.tag if trained_this_run else None

    if full_run or report_only:
        print("== Per-match score report ==")
        report = predict_fixtures.run(tag=tag)
        print("\nProjected group-stage scores (first 8):")
        print(report[report.stage == "group"].head(8)[
            ["match", "group", "home", "away", "score"]].to_string(index=False))

    if full_run or sim_only:
        print("== Simulation ==")
        df = simulate.run(tag=tag, n_sims=args.sims)
        print("\nTitle-probability leaderboard (top 12):")
        print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
