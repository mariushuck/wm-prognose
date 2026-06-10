"""Layer 4 — StatsBomb open data (World Cup xG). Functional, graceful when absent.

``statsbombpy`` is imported lazily so the base install does not require it. For each
configured ``{competition_id, season_id}`` it pulls the matches and their shot
events, summing shot xG per team per match (xA = assisted-shot xG, i.e. the xG of
shots created by a key pass). Returns an empty frame when disabled, the library is
missing, or the fetch fails.

Columns: date, team, opponent, xg, xa
"""
from __future__ import annotations

import pandas as pd

from .team_aliases import normalize

STATSBOMB_COLUMNS = ["date", "team", "opponent", "xg", "xa"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in STATSBOMB_COLUMNS})


def _match_rows(sb, match: pd.Series) -> list[dict]:
    """One row per team for a single match (team xG + assisted xG)."""
    events = sb.events(match_id=int(match["match_id"]))
    shots = events[events["type"] == "Shot"].copy()
    if shots.empty:
        return []
    shots["shot_statsbomb_xg"] = pd.to_numeric(
        shots.get("shot_statsbomb_xg"), errors="coerce").fillna(0.0)
    xg = shots.groupby("team")["shot_statsbomb_xg"].sum()

    # xA: xG credited to the team of the assisting pass (key_pass).
    xa = {}
    if "shot_key_pass_id" in shots.columns and "id" in events.columns:
        kp = events.set_index("id")["team"].to_dict()
        for _, s in shots.iterrows():
            passer_team = kp.get(s.get("shot_key_pass_id"))
            if passer_team is not None:
                xa[passer_team] = xa.get(passer_team, 0.0) + s["shot_statsbomb_xg"]

    home, away = match["home_team"], match["away_team"]
    date = pd.to_datetime(match.get("match_date"), errors="coerce")
    rows = []
    for team, opp in ((home, away), (away, home)):
        rows.append({
            "date": date,
            "team": normalize(team),
            "opponent": normalize(opp),
            "xg": float(xg.get(team, 0.0)),
            "xa": float(xa.get(team, 0.0)),
        })
    return rows


def load(cfg: dict, force: bool = False) -> pd.DataFrame:
    if not cfg.get("enabled", False):
        return _empty()
    try:
        from statsbombpy import sb
    except ImportError:
        return _empty()

    rows: list[dict] = []
    try:
        for comp in cfg.get("competitions", []):
            matches = sb.matches(competition_id=comp["competition_id"],
                                 season_id=comp["season_id"])
            for _, match in matches.iterrows():
                rows.extend(_match_rows(sb, match))
    except Exception:  # network / API hiccup -> stay graceful
        return _empty() if not rows else pd.DataFrame(rows, columns=STATSBOMB_COLUMNS)

    return pd.DataFrame(rows, columns=STATSBOMB_COLUMNS) if rows else _empty()
