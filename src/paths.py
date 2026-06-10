"""Central filesystem paths. Importing this module guarantees the dirs exist."""
from __future__ import annotations

from pathlib import Path

# Project root = parent of the `src/` package directory.
ROOT = Path(__file__).resolve().parent.parent

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

# Per-source raw subdirectories.
RAW_RESULTS = DATA_RAW / "results"
RAW_FIFA_RANKING = DATA_RAW / "fifa_ranking"
RAW_TRANSFERMARKT = DATA_RAW / "transfermarkt"
RAW_STATSBOMB = DATA_RAW / "statsbomb"

# Canonical processed artifact: the match-level training table.
MATCHES_TABLE = DATA_PROCESSED / "matches.parquet"

_ALL_DIRS = [
    DATA_RAW, DATA_PROCESSED, MODELS, OUTPUTS,
    RAW_RESULTS, RAW_FIFA_RANKING, RAW_TRANSFERMARKT, RAW_STATSBOMB,
]


def ensure_dirs() -> None:
    """Create all working directories if they do not yet exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
