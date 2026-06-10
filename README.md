# WM-Prognose v2 — FIFA World Cup 2026 Prediction

Modular, layered prediction system: international results → internal Elo → feature
groups → XGBoost 1X2 model → Monte-Carlo tournament simulation. New data sources
plug in as **a new module + one YAML entry** — no refactoring.

## Architecture

```
configs/data_sources.yaml      central on/off per source
       ▼
src/ingestion.py               orchestrates all loaders → data/processed/matches.parquet
       ▼
src/data_sources/              one module per source (Layer 1 functional, 2–5 stubs)
src/features/                  basic | market | advanced_stats | context  (+ assemble)
       ▼
src/model.py  →  src/simulate.py
```

**Five data layers**, toggled in `configs/data_sources.yaml`:

| Layer | Sources | Status |
|---|---|---|
| 1 Basis | `results` (martj42), `elo_internal` | ✅ functional, default on |
| 2 Markt | `odds` (The Odds API), `fifa_ranking` (Kaggle) | stub |
| 3 Kader | `fbref` (soccerdata), `transfermarkt` (manual) | stub |
| 4 xG | `statsbomb` | stub |
| 5 Kontext | `weather` (Open-Meteo), `venues`, `injuries` | venues functional, rest stub |

Each loader honors the contract `load(cfg, force) -> DataFrame`; disabled or
unimplemented sources return an **empty, correctly-typed** frame so the pipeline
always runs. Feature groups are toggled in `configs/feature_groups.yaml`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                     # hermetic smoke tests (no network)
python -m src.ingestion                    # Layer 1: download results, build Elo
python -m src.model --train --tag v2_full  # train XGBoost, prints LogLoss + Brier
python main.py --simulate-tournament       # Monte-Carlo → outputs/simulation_<date>.parquet
```

Full pipeline in one go: `python main.py`.

## Activating later layers

1. **Odds (Layer 2):** create a key at the-odds-api.com, `export ODDS_API_KEY=...`,
   set `odds.enabled: true`, then implement `src/data_sources/odds.py:load`.
2. **FIFA ranking:** drop the Kaggle CSV at `data/raw/fifa_ranking/fifa_ranking.csv`,
   set `fifa_ranking.enabled: true`.
3. **Squad values (Layer 3):** edit `data/raw/transfermarkt/squad_values.yaml`
   (manual only — no scraping), set `transfermarkt.enabled: true`.
4. **xG (Layer 4):** `pip install statsbombpy`, set `statsbomb.enabled: true`,
   implement the aggregation in `statsbomb.py`.
5. **Context (Layer 5):** complete `configs/venues_2026.yaml`, wire `weather.py`.

Then flip the matching flag in `configs/feature_groups.yaml` to feed the model.

## Tournament configuration

`configs/tournament_2026.yaml` encodes the 48-team / 12-group format and the
Round-of-32 bracket. **Replace the placeholder teams with the official draw**;
names must match the spelling in the martj42 results data.

## Cheat-sheet

| Purpose | Command |
|---|---|
| Re-ingest everything | `python -m src.ingestion --force` |
| Rebuild feature matrix | `python -m src.features --rebuild` |
| Train model | `python -m src.model --train --tag <name>` |
| Update played matches | `python -m src.update_played_matches` |
| Simulate tournament | `python main.py --simulate-tournament` |

## Workflows (`.github/workflows/`)

`wm_live.yml` (daily refresh + sim), `wm_odds_refresh.yml` (hourly odds snapshot,
no-op without key), `wm_retrain.yml` (manual full retrain with a `tag` input).

## Legal

martj42 (CC0), StatsBomb (non-commercial), Open-Meteo (CC-BY 4.0) — fine to use.
FBref via soccerdata: rate-limit. **Transfermarkt: manual only, no scraping.**

## Roadmap

Streamlit dashboard · stacking ensemble (XGBoost + odds-implied + Poisson) ·
daily Brier tracking · calibration plots in CI.
