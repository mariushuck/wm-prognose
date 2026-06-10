"""Layer 2 — bookmaker odds (The Odds API). STUB.

Functional shape is in place: when enabled with a valid ``ODDS_API_KEY`` it would
query the h2h market and return implied 1X2 odds per fixture. Until wired up (or
when disabled / no key), it returns an empty, correctly-typed frame.

Columns: date, home_team, away_team, odds_home, odds_draw, odds_away
"""
from __future__ import annotations

import pandas as pd

ODDS_COLUMNS = ["date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ODDS_COLUMNS})


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    if not cfg.get("enabled", False):
        return _empty()
    api_key = cfg.get("api_key") or ""
    if not api_key:
        # Enabled but no secret available — stay graceful.
        return _empty()
    # TODO: GET {base_url}/sports/{sport}/odds with regions/markets, normalize.
    raise NotImplementedError("odds.load: implement The Odds API request + parse")


if __name__ == "__main__":  # pragma: no cover - manual snapshot per cheat-sheet
    from ..config import source_cfg

    print(load(source_cfg("odds")))
