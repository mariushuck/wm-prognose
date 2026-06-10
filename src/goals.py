"""Poisson goals model — predicts expected goals per side and exact scorelines.

Two XGBoost regressors (``objective="count:poisson"``) share the same feature
matrix as the 1X2 classifier ([features/assemble.py]) but predict ``home_score``
and ``away_score``. From the two rate parameters (lambda_home, lambda_away) we
derive the most-likely exact score, a full score-probability matrix, and — when
needed — 1X2 probabilities. This is the core model behind both the per-match
score report and the tournament simulation.

CLI:  python -m src.goals --train --tag <name>
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import List, Tuple

import joblib  # safe: only loads model bundles this project writes to models/
import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance
from xgboost import XGBRegressor

from . import paths
from .features.assemble import build_training_matrix, load_matches

DEFAULT_MAX_GOALS = 10


def _make_regressor() -> XGBRegressor:
    return XGBRegressor(
        objective="count:poisson",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def model_path(tag: str) -> Path:
    return paths.MODELS / f"goals_{tag}.joblib"


# --- Poisson scoreline maths ---------------------------------------------
def _poisson_pmf(lam: float, kmax: int) -> np.ndarray:
    """PMF vector P(X=0..kmax) for X ~ Poisson(lam); last entry holds the tail."""
    lam = max(float(lam), 1e-6)
    ks = np.arange(0, kmax + 1)
    # log-pmf for stability, then normalize so the truncated grid sums to 1.
    log_pmf = ks * np.log(lam) - lam - np.array([sum(np.log(np.arange(1, k + 1))) if k else 0.0
                                                 for k in ks])
    pmf = np.exp(log_pmf)
    pmf = pmf / pmf.sum()
    return pmf


def score_matrix(lam_home: float, lam_away: float, max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    """Joint probability grid M[i, j] = P(home=i, away=j) (independent Poisson)."""
    ph = _poisson_pmf(lam_home, max_goals)
    pa = _poisson_pmf(lam_away, max_goals)
    return np.outer(ph, pa)


def most_likely_score(lam_home: float, lam_away: float,
                      max_goals: int = DEFAULT_MAX_GOALS) -> Tuple[int, int]:
    """Argmax exact score over the joint Poisson grid."""
    matrix = score_matrix(lam_home, lam_away, max_goals)
    i, j = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return int(i), int(j)


def derive_1x2(matrix: np.ndarray) -> Tuple[float, float, float]:
    """(P(home win), P(draw), P(away win)) from a score matrix."""
    p_draw = float(np.trace(matrix))
    p_home = float(np.tril(matrix, -1).sum())   # home goals > away goals
    p_away = float(np.triu(matrix, 1).sum())
    return p_home, p_draw, p_away


def most_likely_score_by_outcome(lam_home: float, lam_away: float,
                                 max_goals: int = DEFAULT_MAX_GOALS) -> Tuple[int, int]:
    """Most-likely scoreline *given the most-likely outcome*.

    First pick H/D/A (argmax of the derived 1X2 probabilities), then take the most
    probable exact score within that outcome's region of the grid. Avoids the
    global-mode bias toward 1:1 and yields varied, intuitive scores (1:0, 2:1, 0:0).
    """
    m = score_matrix(lam_home, lam_away, max_goals)
    p_home, p_draw, p_away = derive_1x2(m)
    outcome = max((p_home, "H"), (p_draw, "D"), (p_away, "A"))[1]
    region = {
        "H": np.tril(m, -1),               # home goals > away goals
        "A": np.triu(m, 1),                # away goals > home goals
        "D": np.diag(np.diag(m)),          # the draw diagonal
    }[outcome]
    i, j = np.unravel_index(int(region.argmax()), region.shape)
    return int(i), int(j)


# --- training / inference ------------------------------------------------
def train(tag: str = "v2_full", test_fraction: float = 0.2) -> Path:
    """Train both Poisson regressors and save the bundle."""
    X, _y, feature_names, _ctx = build_training_matrix()
    matches = load_matches().reset_index(drop=True)
    if X.empty or X.shape[1] == 0:
        raise ValueError("Empty feature matrix — run ingestion and check feature groups.")

    y_home = matches["home_score"].astype(float).to_numpy()
    y_away = matches["away_score"].astype(float).to_numpy()

    n = len(X)
    split = int(n * (1 - test_fraction))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]

    metrics = {}
    models = {}
    for side, y in (("home", y_home), ("away", y_away)):
        ev = _make_regressor()
        ev.fit(X_tr, y[:split])
        pred = np.clip(ev.predict(X_te), 1e-6, None)
        metrics[f"{side}_poisson_deviance"] = float(mean_poisson_deviance(y[split:], pred))
        full = _make_regressor()
        full.fit(X, y)
        models[side] = full

    print(f"Holdout ({n - split:,} matches)  "
          f"Poisson deviance  home={metrics['home_poisson_deviance']:.4f}  "
          f"away={metrics['away_poisson_deviance']:.4f}")

    bundle = {
        "model_home": models["home"],
        "model_away": models["away"],
        "feature_names": feature_names,
        "tag": tag,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": n,
        **metrics,
    }
    out = model_path(tag)
    joblib.dump(bundle, out)
    print(f"Saved goals model -> {out}")
    return out


def load_bundle(tag_or_path: str) -> dict:
    path = Path(tag_or_path)
    if not path.exists():
        path = model_path(tag_or_path)
    if not path.exists():
        raise FileNotFoundError(f"No goals model for '{tag_or_path}'")
    return joblib.load(path)


def latest_bundle() -> dict:
    candidates = sorted(paths.MODELS.glob("goals_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No goals model in models/ — run goals.train first.")
    return joblib.load(candidates[-1])


def expected_goals(bundle: dict, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lambda_home, lambda_away) arrays, clipped to be strictly positive."""
    X_aligned = X.reindex(columns=bundle["feature_names"], fill_value=0.0)
    lam_home = np.clip(bundle["model_home"].predict(X_aligned), 1e-6, None)
    lam_away = np.clip(bundle["model_away"].predict(X_aligned), 1e-6, None)
    return lam_home, lam_away


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Poisson goals model")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--tag", default="v2_full")
    args = parser.parse_args()
    if args.train:
        train(tag=args.tag)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
