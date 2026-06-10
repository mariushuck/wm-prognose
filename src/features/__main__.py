"""`python -m src.features --rebuild` — rebuild & report the feature matrix."""
from __future__ import annotations

import argparse

from .assemble import build_training_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the feature matrix")
    parser.add_argument("--rebuild", action="store_true", help="(default action)")
    parser.parse_args()

    X, y, names, _ = build_training_matrix()
    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} cols")
    print(f"Columns: {names}")
    print(f"Label distribution:\n{y.value_counts().to_string()}")


if __name__ == "__main__":
    main()
