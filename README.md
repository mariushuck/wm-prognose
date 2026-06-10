# WM-Prognose v2 — FIFA World Cup 2026 Prediction

Modular, layered prediction system: international results → internal Elo → feature
groups → **Poisson goals model** → per-match scorelines + Monte-Carlo tournament
simulation. New data sources plug in as **a new module + one YAML entry** — no
refactoring.

## Architecture

```
configs/data_sources.yaml      central on/off per source
       ▼
src/ingestion.py               orchestrates all loaders → data/processed/matches.parquet
       ▼
src/data_sources/              one module per source (all implemented, data-activated)
src/features/                  basic | ranking | squad | market | advanced_stats | context
       ▼
src/goals.py  (Poisson goals model: expected goals → exact scoreline)
       ├─►  src/predict_fixtures.py   per-match score report (uses src/bracket.py)
       └─►  src/simulate.py           Monte-Carlo advancement probabilities
```

`src/model.py` (1X2 classifier) remains available but the default pipeline uses
the goals model, from which 1X2 probabilities can also be derived.

**Five data layers**, toggled in `configs/data_sources.yaml`. All are implemented —
activation is **data/key-only** (drop in a file or API key, flip the flag):

| Layer | Sources | Activation |
|---|---|---|
| 1 Basis | `results` (martj42), `elo_internal` | ✅ on by default |
| 2 Markt | `fifa_ranking` (Kaggle CSV), `odds` (The Odds API) | add CSV / `ODDS_API_KEY` |
| 3 Kader | `transfermarkt` (manual YAML) | edit `squad_values.yaml` |
| 4 xG | `statsbomb` (open data) | `pip install statsbombpy` |
| 5 Kontext | `venues`, `weather` (Open-Meteo) | venues on; weather keyless |

Each loader honors the contract `load(cfg, force) -> DataFrame` and returns an
**empty, correctly-typed** frame when its data/key is absent, so the pipeline always
runs. Feature groups (`basic`, `ranking`, `market_odds`, `squad_value`,
`advanced_stats`, `context`) are toggled in `configs/feature_groups.yaml`. See
**[MAINTENANCE.md](MAINTENANCE.md)** for the per-layer activation recipes and caveats.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                     # hermetic smoke tests (no network)
python -m src.ingestion                    # Layer 1: download results, build Elo
python -m src.goals --train --tag v2_full  # train Poisson goals model
python -m src.predict_fixtures             # → outputs/match_predictions_<date>.csv
python main.py --simulate-tournament       # Monte-Carlo → outputs/simulation_<date>.parquet
```

Full pipeline in one go: `python main.py` (ingest → train → score report → simulate).

> **macOS note:** XGBoost needs the OpenMP runtime — `brew install libomp` once.

## Per-match scores (Endergebnis)

`python main.py --predict-scores` (or `python -m src.predict_fixtures`) writes a
predicted **final score for every one of the 104 matches** to
`outputs/match_predictions_<date>.csv`: columns `match, stage, group, home, away,
xg_home, xg_away, pred_home, pred_away, score, winner`. The group stage is played
directly; the knockout bracket is resolved through `src/bracket.py` using the
`W<match#>`/`L<match#>` tokens in `configs/fixtures_2026.yaml`, giving one
projected score per match through to the final.

The `score` is the **most-likely scoreline given the predicted outcome**: the model
first picks H/D/A (argmax of the 1X2 probabilities derived from the Poisson grid),
then the most probable exact score *within that outcome* — so a predicted home win
yields e.g. 2:1, not the global 1:1 mode. The `xg_home`/`xg_away` columns show the
expected-goal magnitudes behind each result.

## Activating later layers (data/key-only — no coding)

1. **FIFA ranking:** drop the Kaggle CSV at `data/raw/fifa_ranking/fifa_ranking.csv`,
   set `fifa_ranking.enabled: true`, flip `ranking: true`. (Works immediately.)
2. **Odds (Layer 2):** `export ODDS_API_KEY=...`, set `odds.enabled: true`, flip
   `market_odds: true`. Optional historical odds CSV at `data/raw/odds/odds_history.csv`
   makes it count for training too.
3. **Squad values (Layer 3):** edit `data/raw/transfermarkt/squad_values.yaml`
   (manual only — no scraping), set `transfermarkt.enabled: true`, flip `squad_value`.
4. **xG (Layer 4):** `pip install statsbombpy`, set `statsbomb.enabled: true`, flip
   `advanced_stats: true`.
5. **Context (Layer 5):** `context: true` gives rest-days immediately; weather needs
   `weather.enabled: true` + fixtures carrying `venue`/`date`.

Each step: `enabled: true` in `configs/data_sources.yaml`, flip the matching flag in
`configs/feature_groups.yaml`, re-run `python -m src.ingestion --force` and retrain.
Full details + caveats in **[MAINTENANCE.md](MAINTENANCE.md)**.

## Tournament configuration

`configs/fixtures_2026.yaml` is the **schedule source** for both the score report
and the simulation: the 48-team / 12-group `groups`, the ordered 72-match
`group_stage`, and the knockout rounds with slot tokens (`1A`/`2B`/`3rd-N` and
`W<match#>`/`L<match#>`). Team names must match the spelling in the martj42 results
data.

To scaffold group pairings from the groups: `python -m src.predict_fixtures
--generate-fixtures` (prints to stdout; never overwrites your file).

See **[MAINTENANCE.md](MAINTENANCE.md)** for the full per-layer checklist of what to
maintain by hand, what to download, and which API keys/installs each data layer
needs.

## Cheat-sheet

| Purpose | Command |
|---|---|
| Re-ingest everything | `python -m src.ingestion --force` |
| Rebuild feature matrix | `python -m src.features --rebuild` |
| Train goals model | `python -m src.goals --train --tag <name>` |
| Predict every match score | `python main.py --predict-scores` |
| Update played matches | `python -m src.update_played_matches` |
| Simulate tournament | `python main.py --simulate-tournament` |

## Workflows (`.github/workflows/`)

`wm_live.yml` (daily refresh + sim), `wm_odds_refresh.yml` (hourly odds snapshot,
no-op without key), `wm_retrain.yml` (manual full retrain with a `tag` input).

## Legal

martj42 (CC0), StatsBomb (non-commercial), Open-Meteo (CC-BY 4.0) — fine to use.
The Odds API: free-tier ToS. **Transfermarkt: manual only, no scraping.**

## Roadmap

Streamlit dashboard · Dixon-Coles low-score correction (the `score_matrix` seam is
ready) · stacking ensemble (goals + 1X2 + odds-implied) · daily Brier tracking ·
calibration plots in CI.
