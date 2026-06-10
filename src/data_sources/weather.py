"""Layer 5 — match-day weather (Open-Meteo, CC-BY 4.0). Functional, graceful.

Open-Meteo needs no API key. Weather can only be attached to fixtures that carry a
**venue + date**, so ``load`` reads an optional ``matches`` list from
``configs/fixtures_2026.yaml`` (each item: ``home, away, venue, date``), looks the
venue's coordinates up in ``venues_2026.yaml``, and fetches the kickoff-day
forecast. Returns empty when disabled or when no dated/venued fixtures exist (the
plain ``group_stage`` pairs carry no venue/date).

Columns: date, home_team, away_team, venue, temperature_c, wind_kph, precipitation_mm
"""
from __future__ import annotations

import pandas as pd
import requests

from ..config import load_yaml, source_cfg
from .team_aliases import normalize
from .venues import load as load_venues

WEATHER_COLUMNS = ["date", "home_team", "away_team", "venue",
                   "temperature_c", "wind_kph", "precipitation_mm"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in WEATHER_COLUMNS})


def fetch_for_venue(lat: float, lon: float, day: str, cfg: dict) -> dict | None:
    """Daily mean forecast for one venue/day; None on any failure (graceful)."""
    params = {"latitude": lat, "longitude": lon, "start_date": day, "end_date": day,
              "daily": "temperature_2m_mean,wind_speed_10m_max,precipitation_sum",
              "timezone": "UTC"}
    try:
        resp = requests.get(cfg["base_url"], params=params, timeout=30)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        return {
            "temperature_c": (daily.get("temperature_2m_mean") or [None])[0],
            "wind_kph": (daily.get("wind_speed_10m_max") or [None])[0],
            "precipitation_mm": (daily.get("precipitation_sum") or [None])[0],
        }
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    if not cfg.get("enabled", False):
        return _empty()

    fixtures = load_yaml("fixtures_2026").get("matches", []) or []
    if not fixtures:
        return _empty()

    venues = load_venues(source_cfg("venues"))
    coords = {v["name"]: (v["lat"], v["lon"]) for v in venues.to_dict("records")}

    rows = []
    for m in fixtures:
        venue, day = m.get("venue"), m.get("date")
        if not venue or not day or venue not in coords:
            continue
        lat, lon = coords[venue]
        wx = fetch_for_venue(lat, lon, str(day), cfg)
        if wx is None:
            continue
        rows.append({"date": pd.to_datetime(day, errors="coerce"),
                     "home_team": normalize(m.get("home")),
                     "away_team": normalize(m.get("away")),
                     "venue": venue, **wx})
    return pd.DataFrame(rows, columns=WEATHER_COLUMNS) if rows else _empty()
