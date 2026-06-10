"""Layer 2 — FIFA world ranking (Kaggle CSV). STUB-functional.

Reads ``data/raw/fifa_ranking/<filename>`` if present and normalizes it; returns
an empty frame when disabled or the file is absent.

Columns: rank_date (datetime64), team, rank (int), points (float)
"""
from __future__ import annotations

import pandas as pd

from .. import paths

FIFA_COLUMNS = ["rank_date", "team", "rank", "points"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in FIFA_COLUMNS})


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    if not cfg.get("enabled", False):
        return _empty()
    path = paths.RAW_FIFA_RANKING / cfg.get("filename", "fifa_ranking.csv")
    if not path.exists():
        return _empty()

    df = pd.read_csv(path)
    # Kaggle schema: rank, country_full, total_points, rank_date, ...
    rename = {"country_full": "team", "total_points": "points"}
    df = df.rename(columns=rename)
    if "rank_date" in df.columns:
        df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")
    for col in FIFA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[FIFA_COLUMNS]
