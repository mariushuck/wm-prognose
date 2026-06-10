"""Context feature group (Layer 5) — travel, weather, rest, injuries. STUB.

Contributes no columns until venue coordinates + fixtures + weather/injury data
are wired. Planned columns: travel_km_diff, rest_days_diff, temperature_c,
injuries_diff. The ``venues.haversine_km`` helper is ready for travel distances.
"""
from __future__ import annotations

import pandas as pd

COLUMNS = ["travel_km_diff", "rest_days_diff", "temperature_c", "injuries_diff"]


def columns() -> list:
    return list(COLUMNS)


def build(matches_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    return pd.DataFrame(index=range(len(matches_df)))


def fixture_features(home: str, away: str, neutral: bool, context: dict) -> dict:
    return {}
