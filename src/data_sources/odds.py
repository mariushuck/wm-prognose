"""Layer 2 — bookmaker odds (The Odds API). Functional, graceful when absent.

Two activation paths:
  * **Live** — with a valid ``ODDS_API_KEY`` it queries the h2h market for the
    configured sport and returns the best (max) decimal odds per upcoming fixture.
  * **History file** — if ``data/raw/odds/<history_file>`` exists it is read too
    (and merged), which is what makes odds usable for *training* (the live API has
    no history). Expected columns: date, home_team, away_team, odds_home,
    odds_draw, odds_away.

Returns an empty, correctly-typed frame when disabled / no key / no file.

Columns: date, home_team, away_team, odds_home, odds_draw, odds_away
"""
from __future__ import annotations

import pandas as pd
import requests

from .. import paths
from .team_aliases import normalize

ODDS_COLUMNS = ["date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ODDS_COLUMNS})


def _parse_event(event: dict) -> dict | None:
    """Best decimal h2h odds for one Odds-API event -> one ODDS row, or None."""
    home, away = event.get("home_team"), event.get("away_team")
    if not home or not away:
        return None
    best = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for oc in market.get("outcomes", []):
                name, price = oc.get("name"), oc.get("price")
                if price is None:
                    continue
                if name == home:
                    best["home"] = max(best["home"], price)
                elif name == away:
                    best["away"] = max(best["away"], price)
                elif name and name.lower() == "draw":
                    best["draw"] = max(best["draw"], price)
    if not all(best.values()):
        return None
    return {
        "date": pd.to_datetime(event.get("commence_time"), errors="coerce"),
        "home_team": normalize(home),
        "away_team": normalize(away),
        "odds_home": best["home"],
        "odds_draw": best["draw"],
        "odds_away": best["away"],
    }


def _fetch_live(cfg: dict) -> pd.DataFrame:
    api_key = cfg.get("api_key") or ""
    if not api_key:
        return _empty()
    url = f"{cfg['base_url'].rstrip('/')}/sports/{cfg['sport']}/odds"
    params = {"apiKey": api_key, "regions": cfg.get("regions", "eu"),
              "markets": cfg.get("markets", "h2h"), "oddsFormat": "decimal"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    rows = [r for r in (_parse_event(e) for e in resp.json()) if r]
    return pd.DataFrame(rows, columns=ODDS_COLUMNS) if rows else _empty()


def _read_history(cfg: dict) -> pd.DataFrame:
    name = cfg.get("history_file", "odds_history.csv")
    path = paths.DATA_RAW / "odds" / name
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, parse_dates=["date"])
    for col in ODDS_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for side in ("home_team", "away_team"):
        df[side] = df[side].map(normalize)
    return df[ODDS_COLUMNS]


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    if not cfg.get("enabled", False):
        return _empty()
    history = _read_history(cfg)
    try:
        live = _fetch_live(cfg)
    except requests.RequestException:
        live = _empty()
    frames = [f for f in (history, live) if not f.empty]
    if not frames:
        return _empty()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["home_team", "away_team", "date"], keep="last")


if __name__ == "__main__":  # pragma: no cover - manual snapshot per cheat-sheet
    from ..config import source_cfg

    print(load(source_cfg("odds")))
