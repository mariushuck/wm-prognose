"""Train and persist the 1X2 match-outcome model (XGBoost ``multi:softprob``).

The feature matrix comes from ``features/assemble.py`` so the active feature
groups fully determine the inputs. The saved bundle stores the feature list and
class order, letting the simulator build leak-free, aligned vectors later.

CLI:  python -m src.model --train --tag <name>
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import joblib  # safe: only loads model bundles this project writes to models/
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from . import paths
from .features.assemble import build_training_matrix

# Fixed label order so encoded ints are stable across runs.
CLASSES: List[str] = ["A", "D", "H"]
_CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}


def _multiclass_brier(y_true_int: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error between predicted probs and one-hot truth."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true_int)), y_true_int] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _make_model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def model_path(tag: str) -> Path:
    return paths.MODELS / f"xgb_{tag}.joblib"


def train(tag: str = "v2_full", test_fraction: float = 0.2) -> Path:
    """Train on all history (chronological holdout for metrics) and save bundle."""
    X, y, feature_names, _context = build_training_matrix()
    if X.empty or X.shape[1] == 0:
        raise ValueError("Empty feature matrix — check ingestion and feature groups.")

    y_int = y.map(_CLASS_TO_INT).to_numpy()

    # Chronological holdout: rows already arrive sorted by date.
    n = len(X)
    split = int(n * (1 - test_fraction))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y_int[:split], y_int[split:]

    eval_model = _make_model()
    eval_model.fit(X_tr, y_tr)
    proba = eval_model.predict_proba(X_te)
    ll = log_loss(y_te, proba, labels=list(range(len(CLASSES))))
    brier = _multiclass_brier(y_te, proba)
    print(f"Holdout ({len(X_te):,} matches)  LogLoss={ll:.4f}  Brier={brier:.4f}")

    # Refit on the full history for the production model.
    model = _make_model()
    model.fit(X, y_int)

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "classes": CLASSES,
        "tag": tag,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": n,
        "holdout_logloss": ll,
        "holdout_brier": brier,
    }
    out = model_path(tag)
    joblib.dump(bundle, out)
    print(f"Saved model -> {out}")
    return out


def load_bundle(tag_or_path: str) -> dict:
    """Load a saved model bundle by tag or explicit path."""
    path = Path(tag_or_path)
    if not path.exists():
        path = model_path(tag_or_path)
    if not path.exists():
        raise FileNotFoundError(f"No model found for '{tag_or_path}'")
    return joblib.load(path)


def latest_bundle() -> dict:
    """Load the most recently modified model bundle in models/."""
    candidates = sorted(paths.MODELS.glob("xgb_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No trained models in models/ — run model.train first.")
    return joblib.load(candidates[-1])


def predict_proba(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Predict class probabilities as a DataFrame with H/D/A columns."""
    X_aligned = X.reindex(columns=bundle["feature_names"], fill_value=0.0)
    proba = bundle["model"].predict_proba(X_aligned)
    return pd.DataFrame(proba, columns=bundle["classes"], index=X.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the WM-Prognose model")
    parser.add_argument("--train", action="store_true", help="train a new model")
    parser.add_argument("--tag", default="v2_full", help="model tag (filename suffix)")
    args = parser.parse_args()

    if args.train:
        train(tag=args.tag)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
