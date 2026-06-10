"""Layer 1 — international match results (martj42/international_results).

Functional loader: downloads ``results.csv``, ``shootouts.csv`` and
``goalscorers.csv`` from raw GitHub, caches them under ``data/raw/results/`` and
returns a normalized results DataFrame.

Result-table columns (the canonical raw-results contract):
    date (datetime64), home_team, away_team, home_score (int), away_score (int),
    tournament, city, country, neutral (bool)
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from .. import paths

RESULT_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in RESULT_COLUMNS})


def _cache_path(filename: str) -> Path:
    return paths.RAW_RESULTS / filename


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return age_hours < max_age_hours


def _download(base_url: str, filename: str, dest: Path) -> None:
    url = f"{base_url.rstrip('/')}/{filename}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)


def _fetch(cfg: dict, key: str, force: bool) -> Path:
    """Ensure one martj42 CSV is cached locally; return its path."""
    filename = cfg.get("files", {}).get(key, f"{key}.csv")
    dest = _cache_path(filename)
    max_age = float(cfg.get("cache_max_age_hours", 24))
    if force or not _is_fresh(dest, max_age):
        _download(cfg["base_url"], filename, dest)
    return dest


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    """Return the normalized results table (newest schema, sorted by date)."""
    if not cfg.get("enabled", True):
        return _empty()

    path = _fetch(cfg, "results", force)
    df = pd.read_csv(path, parse_dates=["date"])

    # martj42 already uses our target column names; coerce types defensively.
    df = df.rename(columns={"home_score": "home_score", "away_score": "away_score"})
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[RESULT_COLUMNS].sort_values("date").reset_index(drop=True)
    return df


def load_shootouts(cfg: dict, force: bool = False) -> pd.DataFrame:
    """Penalty-shootout outcomes: date, home_team, away_team, winner."""
    if not cfg.get("enabled", True):
        return pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])
    path = _fetch(cfg, "shootouts", force)
    return pd.read_csv(path, parse_dates=["date"])


if __name__ == "__main__":  # pragma: no cover - manual smoke
    from ..config import source_cfg

    out = load(source_cfg("results"), force=False)
    print(f"results: {len(out):,} matches, {out['date'].min()} -> {out['date'].max()}")
