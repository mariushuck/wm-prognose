"""Append newly-played World Cup matches so Elo & form roll forward.

During the tournament, after each matchday the live workflow re-downloads martj42
results (which include just-played fixtures) and rebuilds the match table. This
module is the explicit entry point for that refresh; it currently delegates to
ingestion's Layer-1 rebuild and reports the latest matches found.

CLI:  python -m src.update_played_matches
"""
from __future__ import annotations

import pandas as pd

from .ingestion import build_matches_table


def update(force: bool = True) -> pd.DataFrame:
    """Force-refresh results and rebuild the match table; return it."""
    return build_matches_table(force=force)


def main() -> None:
    df = update(force=True)
    if df.empty:
        print("No matches available.")
        return
    recent = df.sort_values("date").tail(5)
    print(f"Match table rebuilt: {len(df):,} matches. Most recent:")
    for row in recent.itertuples(index=False):
        print(f"  {row.date.date()}  {row.home_team} {row.home_score}-{row.away_score} {row.away_team}")


if __name__ == "__main__":
    main()
