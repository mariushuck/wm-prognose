"""Advanced-stats feature group (Layer 4) — rolling xG/xA differentials. STUB.

Contributes no columns until the StatsBomb/FBref caches exist. Planned columns:
xg_diff, xa_diff (recent rolling differentials, home minus away).
"""
from __future__ import annotations

import pandas as pd

COLUMNS = ["xg_diff", "xa_diff"]


def columns() -> list:
    return list(COLUMNS)


def build(matches_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    return pd.DataFrame(index=range(len(matches_df)))


def fixture_features(home: str, away: str, neutral: bool, context: dict) -> dict:
    return {}
