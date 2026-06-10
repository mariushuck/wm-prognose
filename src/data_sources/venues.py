"""Layer 5 — venue master data. Functional (reads local YAML).

Loads ``configs/venues_2026.yaml`` into a DataFrame and offers a great-circle
distance helper used by context travel features.

Columns: name, city, country, lat, lon, altitude
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import pandas as pd

from ..config import load_yaml

VENUE_COLUMNS = ["name", "city", "country", "lat", "lon", "altitude"]
_EARTH_RADIUS_KM = 6371.0


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in VENUE_COLUMNS})


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    if not cfg.get("enabled", True):
        return _empty()
    data = load_yaml(cfg.get("config_file", "venues_2026.yaml"))
    rows = data.get("venues", []) or []
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    for col in VENUE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[VENUE_COLUMNS]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * asin(sqrt(a))
