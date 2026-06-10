"""Basic feature group — always available from Layer 1 (results + Elo).

Features (all from the nominal-home perspective):
    elo_diff           home pre-game Elo minus away pre-game Elo
    form_points_diff   home vs away mean points/match over the last N matches
    form_gd_diff       home vs away mean goal difference over the last N matches
    h2h_home_winrate   home team's win rate in recent head-to-head meetings
    neutral            1 if played at a neutral venue, else 0

The same definitions back both training (chronological, leak-free) and the
single-fixture path used by the simulator, via the shared ``context`` snapshots.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

import pandas as pd

from ..config import load_yaml

COLUMNS = ["elo_diff", "form_points_diff", "form_gd_diff", "h2h_home_winrate", "neutral"]


def columns() -> list:
    return list(COLUMNS)


def _params() -> Tuple[int, int]:
    # Tunables live under feature_groups.yaml -> params.
    cfg = load_yaml("feature_groups").get("params", {})
    return int(cfg.get("form_window", 5)), int(cfg.get("h2h_window", 10))


def _points(gd: int) -> int:
    return 3 if gd > 0 else (1 if gd == 0 else 0)


def build(matches_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Compute per-match basic features over the full history (no leakage).

    Also populates ``context`` with end-of-history snapshots used for fixtures:
    ``context['form']`` (team -> {'points', 'gd'}) and ``context['h2h']``.
    """
    form_window, h2h_window = _params()

    if matches_df.empty:
        context.setdefault("form", {})
        context.setdefault("h2h", {})
        return pd.DataFrame(columns=COLUMNS)

    df = matches_df.sort_values("date").reset_index(drop=True)

    form: Dict[str, Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=form_window))
    # head-to-head: ordered pair (a, b) -> recent results from a's perspective (1/0.5/0)
    h2h: Dict[Tuple[str, str], Deque[float]] = defaultdict(lambda: deque(maxlen=h2h_window))

    rows = []
    for r in df.itertuples(index=False):
        home, away = r.home_team, r.away_team

        hf = form[home]
        af = form[away]
        home_pts = sum(p for p, _ in hf) / len(hf) if hf else 1.0
        away_pts = sum(p for p, _ in af) / len(af) if af else 1.0
        home_gd = sum(g for _, g in hf) / len(hf) if hf else 0.0
        away_gd = sum(g for _, g in af) / len(af) if af else 0.0

        key = (home, away)
        prior = h2h[key]
        h2h_rate = sum(prior) / len(prior) if prior else 0.5

        rows.append({
            "elo_diff": float(getattr(r, "elo_diff", r.home_elo_pre - r.away_elo_pre)),
            "form_points_diff": home_pts - away_pts,
            "form_gd_diff": home_gd - away_gd,
            "h2h_home_winrate": h2h_rate,
            "neutral": int(bool(r.neutral)),
        })

        # update running state AFTER recording pre-game features
        gd = int(r.home_score) - int(r.away_score)
        form[home].append((_points(gd), gd))
        form[away].append((_points(-gd), -gd))
        res_home = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        h2h[(home, away)].append(res_home)
        h2h[(away, home)].append(1.0 - res_home)

    # snapshots for the fixture path
    context["form"] = {
        t: {
            "points": (sum(p for p, _ in dq) / len(dq)) if dq else 1.0,
            "gd": (sum(g for _, g in dq) / len(dq)) if dq else 0.0,
        }
        for t, dq in form.items()
    }
    context["h2h"] = {k: (sum(v) / len(v)) for k, v in h2h.items() if v}

    return pd.DataFrame(rows, columns=COLUMNS)


def fixture_features(home: str, away: str, neutral: bool, context: dict) -> dict:
    """Feature dict for a prospective fixture, using context snapshots."""
    ratings = context.get("ratings", {})
    start = context.get("elo_start", 1500.0)
    form = context.get("form", {})
    h2h = context.get("h2h", {})

    home_elo = ratings.get(home, start)
    away_elo = ratings.get(away, start)
    hf = form.get(home, {"points": 1.0, "gd": 0.0})
    af = form.get(away, {"points": 1.0, "gd": 0.0})

    return {
        "elo_diff": float(home_elo - away_elo),
        "form_points_diff": hf["points"] - af["points"],
        "form_gd_diff": hf["gd"] - af["gd"],
        "h2h_home_winrate": h2h.get((home, away), 0.5),
        "neutral": int(bool(neutral)),
    }
