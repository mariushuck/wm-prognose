# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

WM-Prognose v2 is a FIFA World Cup 2026 prediction system: international results → internal Elo → feature groups → Poisson goals model → per-match scorelines + Monte-Carlo tournament simulation. Plain `venv` + `pip` + `pytest`; no build step.

## Commands

```bash
source .venv/bin/activate          # always activate first
pip install -r requirements.txt

pytest -q                          # full suite (hermetic — no network)
pytest tests/test_goals.py -q      # one file
pytest tests/test_bracket.py::test_full_mini_tournament_resolves   # one test

python -m src.ingestion [--force]            # download martj42 results, build Elo → data/processed/matches.parquet
python -m src.goals --train --tag v2_full    # train the Poisson goals model
python -m src.predict_fixtures               # → outputs/match_predictions_<date>.csv (score per match)
python -m src.features --rebuild             # inspect the feature matrix
python main.py                               # full pipeline: ingest → train → score report → simulate
python main.py --predict-scores | --simulate-tournament   # one stage only
```

**macOS:** XGBoost needs OpenMP — `brew install libomp` once, or `import xgboost` fails at load.

Only `src.ingestion`/`results.py` hit the network (martj42 GitHub, cached to `data/raw/`). Tests are hermetic: loaders are exercised only via the disabled→empty path; features via injected caches (`tests/conftest.py` provides `sample_results`, `mini_fixtures`).

## Architecture

**Three contracts hold the system together** (honor them when adding sources/features):
1. **Raw results table** (martj42 schema): `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`.
2. **Processed match table** `data/processed/matches.parquet`: results + pre-game Elo + 1X2 label; sorted by date (features rely on this ordering). Written by `ingestion.build_matches_table`.
3. **Loader contract** — every `data_sources/*.py` exposes `load(cfg, force) -> DataFrame` with a fixed column set, returning an *empty, correctly-typed* frame when its data/key/lib is absent. So the pipeline never breaks regardless of which layers are active.

**Data layers are config-gated**, not code-gated. `configs/data_sources.yaml` toggles each source `enabled`; `ingestion.LOADERS` caches each enabled aux source to `data/processed/<name>.parquet`. `configs/feature_groups.yaml` toggles feature groups. Activating a layer = add data/key + flip both flags + re-ingest + retrain — never a code change. All five layers are implemented; Layer 1 (results, elo_internal) is always on.

**Feature pipeline** (`src/features/`): `assemble.py` is the single reader of `feature_groups.yaml` and the only place that joins groups. Each feature module implements a **prepare()/build()/fixture_features()** triad:
- `prepare(matches, context)` fills per-team snapshots into the shared `context` dict.
- `build(matches, context)` returns the training columns (an enabled group emits its **fixed COLUMNS**, zero-filled where data is missing — this keeps train/serve feature sets aligned).
- `fixture_features(home, away, neutral, context)` returns the same columns for a prospective (neutral-venue) match.

`assemble.build_training_matrix()` drives training; `assemble.prepare_inference_context(matches)` builds the shared context (Elo ratings + every enabled module's snapshots + cached source tables under `context['sources']`) and is what `predict_fixtures`/`simulate` call. `GROUP_MODULES` maps group→module 1:1. Group↔source: `basic`←Layer 1, `ranking`←fifa_ranking, `market_odds`←odds, `squad_value`←transfermarkt, `advanced_stats`←statsbomb, `context`←results(rest)+weather.

**Models** (`src/goals.py` is the core): two XGBoost `count:poisson` regressors predict home/away expected goals (λ). `score_matrix(λh,λa)` → `derive_1x2` and `most_likely_score_by_outcome` (pick H/D/A by argmax, then the most-probable score within that outcome — avoids the global 1:1 mode). `src/model.py` (1X2 classifier) still exists but is **off the default path**; the goals model derives 1X2 when needed.

**Tournament** (`src/bracket.py` is shared): `play_tournament(cfg, play_fn, ratings, rng)` resolves the official schedule from `configs/fixtures_2026.yaml` — plays the 72 group matches, computes standings (points→GD→GF), then resolves knockout slot tokens (`1A`/`2B`/`3rd-N` group slots, `W<match#>`/`L<match#>` references). The engine is model-agnostic via the `play_fn(home, away, knockout) -> (hg, ag, winner)` callback:
- `predict_fixtures.py` passes a **deterministic** play_fn (most-likely score; knockout draws broken by λ then Elo) → single projected bracket with a score for all 104 matches.
- `simulate.py` passes a **sampling** play_fn (independent-Poisson scorelines; knockout draws → Elo-weighted penalty flip) → Monte-Carlo advancement/title probabilities.

`configs/fixtures_2026.yaml` is the single schedule source (groups + ordered `group_stage` + knockout tokens). Team names must match the martj42 spelling; `data_sources/team_aliases.py` normalizes foreign spellings (Kaggle/StatsBomb → martj42).

## Gotchas
- `matches.parquet` must stay date-sorted — `basic.build` re-sorts, but other feature builders and the `result` label assume the input order is chronological.
- Odds and weather only have data for upcoming fixtures, so they are inert for training until a historical file is supplied (documented in MAINTENANCE.md).
- `joblib` model bundles in `models/` are trusted (self-written); don't load bundles from untrusted sources.
- See `MAINTENANCE.md` (German) for the per-layer activation recipes and data caveats.
