"""Assemble enabled feature groups into the model matrix.

This is the *single* place that reads ``feature_groups.yaml``. It builds the shared
``context`` (current Elo ratings, form/h2h snapshots, and the cached Layer 2–5
source tables) so the training matrix and the simulator's per-fixture vectors use
identical feature definitions.

Each feature module exposes ``prepare(matches, context)`` (fills per-team
snapshots), ``build(matches, context)`` (the training columns) and
``fixture_features(home, away, neutral, context)``.

CLI:  python -m src.features --rebuild   (see __main__.py)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .. import paths
from ..config import is_feature_group_enabled, source_cfg
from ..data_sources import elo_internal
from . import advanced_stats, basic, context as context_feat, market, ranking, squad

# Feature group name (feature_groups.yaml) -> builder module (1:1).
GROUP_MODULES = {
    "basic": basic,
    "ranking": ranking,
    "squad_value": squad,
    "market_odds": market,
    "advanced_stats": advanced_stats,
    "context": context_feat,
}

# Layer 2–5 caches that feature builders may consume (data/processed/<name>.parquet).
SOURCE_CACHES = ["fifa_ranking", "transfermarkt", "statsbomb", "odds", "weather"]


def _enabled_modules() -> List:
    """Enabled builder modules in stable (group-declaration) order."""
    return [m for g, m in GROUP_MODULES.items() if is_feature_group_enabled(g)]


def load_matches() -> pd.DataFrame:
    if not paths.MATCHES_TABLE.exists():
        raise FileNotFoundError(
            f"{paths.MATCHES_TABLE} missing — run `python -m src.ingestion` first."
        )
    return pd.read_parquet(paths.MATCHES_TABLE)


def _load_source_caches() -> Dict[str, pd.DataFrame]:
    """Read each available Layer 2–5 cache; missing ones become empty frames."""
    out: Dict[str, pd.DataFrame] = {}
    for name in SOURCE_CACHES:
        path = paths.DATA_PROCESSED / f"{name}.parquet"
        out[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return out


def build_context(matches_df: pd.DataFrame) -> dict:
    """Current Elo ratings + Elo baseline + cached source tables."""
    elo_cfg = source_cfg("elo_internal")
    table = elo_internal.EloTable.from_results(matches_df, elo_cfg)
    return {
        "ratings": table.ratings,
        "elo_start": float(elo_cfg.get("start_rating", 1500.0)),
        "sources": _load_source_caches(),
    }


def prepare_inference_context(matches_df: pd.DataFrame) -> dict:
    """Context with every enabled module's snapshots populated (for predict/sim)."""
    context = build_context(matches_df)
    for module in _enabled_modules():
        module.prepare(matches_df, context)
    return context


def build_training_matrix(
    matches_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], dict]:
    """Return (X, y, feature_names, context) for model training."""
    matches = load_matches() if matches_df is None else matches_df
    context = build_context(matches)

    parts = []
    for module in _enabled_modules():
        block = module.build(matches, context)
        if block.shape[1]:
            parts.append(block.reset_index(drop=True))

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=range(len(matches)))
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = matches["result"].reset_index(drop=True)
    return X, y, list(X.columns), context


def build_fixture_features(
    home: str, away: str, neutral: bool, context: dict, feature_names: List[str],
) -> pd.DataFrame:
    """One-row feature frame for a prospective fixture, aligned to feature_names."""
    values: Dict[str, float] = {}
    for module in _enabled_modules():
        values.update(module.fixture_features(home, away, neutral, context))
    row = {name: float(values.get(name, 0.0)) for name in feature_names}
    return pd.DataFrame([row], columns=feature_names)


if __name__ == "__main__":  # pragma: no cover
    X, y, names, _ = build_training_matrix()
    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} cols -> {names}")
