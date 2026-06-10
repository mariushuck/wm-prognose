"""Layer 3 — squad market values (Transfermarkt). MANUAL ONLY.

Scraping is forbidden (§9), so this reads a hand-maintained YAML at
``data/raw/transfermarkt/squad_values.yaml``:

    teams:
      Germany:
        market_value: 950_000_000
        avg_age: 26.8
        n_players: 26

Columns: team, market_value (float), avg_age (float), n_players (int)
"""
from __future__ import annotations

import pandas as pd
import yaml

from .. import paths

TM_COLUMNS = ["team", "market_value", "avg_age", "n_players"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in TM_COLUMNS})


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    if not cfg.get("enabled", False):
        return _empty()
    path = paths.RAW_TRANSFERMARKT / cfg.get("filename", "squad_values.yaml")
    if not path.exists():
        return _empty()

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    rows = [
        {"team": team, "market_value": v.get("market_value"),
         "avg_age": v.get("avg_age"), "n_players": v.get("n_players")}
        for team, v in (data.get("teams", {}) or {}).items()
    ]
    if not rows:
        return _empty()
    return pd.DataFrame(rows)[TM_COLUMNS]
