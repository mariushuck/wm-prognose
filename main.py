"""WM-Prognose v2 — end-to-end entry point.

Default run:        ingest -> (train) -> simulate.
--simulate-tournament : run simulation only (uses the latest / tagged model).
--no-train            : skip training, reuse the latest existing model.

Examples:
    python main.py                          # full pipeline
    python main.py --simulate-tournament    # simulate with existing model
    python main.py --tag v2_full --sims 5000
"""
from __future__ import annotations

import argparse

from src import ingestion
from src import model as model_mod
from src import simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="WM-Prognose v2 pipeline")
    parser.add_argument("--simulate-tournament", action="store_true",
                        help="run only the tournament simulation")
    parser.add_argument("--no-train", action="store_true",
                        help="skip training; reuse the latest model")
    parser.add_argument("--force", action="store_true",
                        help="force re-ingestion of all sources")
    parser.add_argument("--tag", default="v2_full", help="model tag")
    parser.add_argument("--sims", type=int, default=2000, help="number of simulations")
    args = parser.parse_args()

    trained_this_run = False
    if not args.simulate_tournament:
        print("== Ingestion ==")
        summary = ingestion.run(force=args.force)
        print(f"  matches: {summary.get('matches')}")

        if not args.no_train:
            print("== Training ==")
            model_mod.train(tag=args.tag)
            trained_this_run = True

    print("== Simulation ==")
    # Use the freshly trained tag if we trained; otherwise the latest saved model.
    df = simulate.run(tag=args.tag if trained_this_run else None, n_sims=args.sims)
    print("\nTitle-probability leaderboard (top 12):")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
