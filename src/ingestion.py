"""Ingestion orchestrator.

Iterates the sources declared in ``configs/data_sources.yaml``, calls each
module's ``load(cfg, force)`` (honoring the loader contract), and writes:

  * ``data/processed/matches.parquet`` — the canonical match-level table
    (raw results + pre-game Elo + 1X2 label) that feature builders extend.
  * ``data/processed/<source>.parquet`` — normalized cache for every other
    enabled source, consumed later by feature builders.

Disabled or not-yet-implemented sources are skipped gracefully.

CLI:  python -m src.ingestion [--force]
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import paths
from .config import data_sources, source_cfg
from .data_sources import (
    elo_internal, fifa_ranking, odds, results, statsbomb,
    transfermarkt, venues, weather,
)

# Source name -> loader module. `results` and `elo_internal` are Layer 1 and get
# special handling; the rest are cached generically for feature builders.
LOADERS = {
    "odds": odds,
    "fifa_ranking": fifa_ranking,
    "transfermarkt": transfermarkt,
    "statsbomb": statsbomb,
    "weather": weather,
    "venues": venues,
}


def _label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def build_matches_table(force: bool = False) -> pd.DataFrame:
    """Layer 1: results + Elo -> canonical match table, saved to parquet."""
    res = results.load(source_cfg("results"), force=force)
    res_with_elo, _ratings = elo_internal.compute(res, source_cfg("elo_internal"))

    if not res_with_elo.empty:
        res_with_elo["result"] = [
            _label(int(h), int(a))
            for h, a in zip(res_with_elo["home_score"], res_with_elo["away_score"])
        ]
    else:
        res_with_elo["result"] = pd.Series(dtype="object")

    paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    res_with_elo.to_parquet(paths.MATCHES_TABLE, index=False)
    return res_with_elo


def _cache_source(name: str, module, force: bool) -> int:
    """Load one auxiliary source and cache it; return row count (0 if skipped)."""
    cfg = source_cfg(name)
    df = module.load(cfg, force=force)
    out = paths.DATA_PROCESSED / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return len(df)


def run(force: bool = False) -> dict:
    """Run full ingestion; return a per-source summary of rows ingested."""
    summary: dict = {}

    matches = build_matches_table(force=force)
    summary["matches"] = len(matches)

    configured = data_sources()
    for name, module in LOADERS.items():
        cfg = configured.get(name, {}) or {}
        enabled_default = name == "venues"  # venues defaults on
        if not cfg.get("enabled", enabled_default):
            summary[name] = "disabled"
            continue
        try:
            summary[name] = _cache_source(name, module, force)
        except NotImplementedError:
            summary[name] = "stub"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="WM-Prognose data ingestion")
    parser.add_argument("--force", action="store_true",
                        help="re-download/re-read every source, ignoring caches")
    args = parser.parse_args()

    summary = run(force=args.force)
    print("Ingestion summary:")
    for name, value in summary.items():
        print(f"  {name:<14} {value}")


if __name__ == "__main__":
    main()
