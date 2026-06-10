"""Layer 5 — match-day weather (Open-Meteo, CC-BY 4.0). STUB.

Open-Meteo needs no API key; the live workflow fetches a forecast ~24 h before
kickoff per venue. Returns an empty frame when disabled.

Columns: date, venue, temperature_c, wind_kph, precipitation_mm
"""
from __future__ import annotations

import pandas as pd

WEATHER_COLUMNS = ["date", "venue", "temperature_c", "wind_kph", "precipitation_mm"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in WEATHER_COLUMNS})


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    if not cfg.get("enabled", False):
        return _empty()
    # TODO: for each upcoming fixture/venue, GET cfg['base_url'] with lat/lon and
    # extract the kickoff-hour forecast. Requires fixtures + venue coordinates.
    raise NotImplementedError("weather.load: implement Open-Meteo forecast fetch")


def fetch_for_venue(lat: float, lon: float, when, cfg: dict) -> dict:
    """Placeholder single-venue fetch used by the live workflow. STUB."""
    raise NotImplementedError("weather.fetch_for_venue: implement Open-Meteo request")
