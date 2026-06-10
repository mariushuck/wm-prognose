# Pflege- & Aktivierungs-Checkliste

Vollständige Übersicht: Was du **manuell pflegst**, was du **runterladen** musst und
welche **API-Keys/Installs** je Datenlayer nötig sind. Alle Layer sind **fertig
implementiert** — Aktivierung = nur Datei/Key nachlegen, `enabled: true` setzen und
das passende Feature-Flag anschalten. Kein weiterer Code nötig.

---

## Was JETZT schon ohne dein Zutun läuft

- **Ergebnis-Download** (martj42) inkl. Cache und `update_played_matches`.
- **Internes Elo** aus den Ergebnissen.
- **Basic-Features** (Elo-Diff, Form, H2H, neutral).
- **Tor-Modell** (Poisson) → **Endergebnis pro Spiel** + **Turnier-Simulation**.

Dein wiederkehrender Befehl ist nur: `python main.py`.

---

## Was DU regelmäßig pflegst / prüfst

| Datei | Was | Wann |
|---|---|---|
| `configs/fixtures_2026.yaml` | Gruppen, Spielplan, Bracket-Tokens. **Teamnamen exakt wie in martj42** | vor Turnier / bei Änderungen |
| `configs/venues_2026.yaml` | 16 Stadien (drin) — Koordinaten ggf. verfeinern | einmalig prüfen |
| `data/raw/transfermarkt/squad_values.yaml` | Kaderwerte je Team (nur falls Layer 3 genutzt) | Kader-Updates |
| Retraining | nach Gruppenphase / Achtelfinale neu trainieren (`--tag`) | siehe Strategie |

**Während des Turniers:** `python -m src.update_played_matches` (zieht gespielte
Spiele nach, Elo & Form rollen weiter), dann `python main.py --predict-scores` bzw.
`--simulate-tournament`.

**Namens-Check:** Der Score-Report warnt, wenn ein Team aus `fixtures_2026.yaml`
nicht in den martj42-Daten gefunden wird (→ Default-Elo 1500). Aktuell sind alle
Namen sauber.

---

## Layer-für-Layer: Download / Keys / Aktivierung

Ablauf je Layer: **(1)** Voraussetzung besorgen → **(2)** `enabled: true` in
`configs/data_sources.yaml` → **(3)** `python -m src.ingestion --force` →
**(4)** passendes Flag in `configs/feature_groups.yaml` auf `true` → **(5)** neu
trainieren (`python -m src.goals --train --tag v2_full`).

### Layer 1 — Basis ✅ (aktiv, nichts zu tun)
| Quelle | Du brauchst | Feature-Flag |
|---|---|---|
| `results` (martj42) | nichts — Auto-Download | `basic` (an) |
| `elo_internal` | nichts — berechnet | `basic` (an) |

### Layer 2 — Markt & Quoten
| Quelle | Du brauchst | Feature-Flag | Hinweis |
|---|---|---|---|
| `fifa_ranking` | Kaggle-Dataset `cashncarry/fifaworldranking` → `data/raw/fifa_ranking/fifa_ranking.csv` | `ranking` | **wirkt sofort** (Training + Prognose) |
| `odds` (The Odds API) | Account auf the-odds-api.com (Free-Tier 500/Monat), Key als `ODDS_API_KEY` | `market_odds` | live nur für kommende Spiele; fürs Training optional `data/raw/odds/odds_history.csv` nachlegen |

### Layer 3 — Kader & Marktwerte
| Quelle | Du brauchst | Feature-Flag | Hinweis |
|---|---|---|---|
| `transfermarkt` | `data/raw/transfermarkt/squad_values.yaml` von Hand pflegen (kein Scraping) | `squad_value` | Werte sind statisch je Team (für Historie konstant) |

### Layer 4 — Advanced Stats (xG)
| Quelle | Du brauchst | Feature-Flag | Hinweis |
|---|---|---|---|
| `statsbomb` | `pip install statsbombpy` (Open Data, **kein Key**) | `advanced_stats` | deckt WM 2018 & 2022 ab → dünn (historisch meist 0) |

### Layer 5 — Kontext
| Quelle | Du brauchst | Feature-Flag | Hinweis |
|---|---|---|---|
| `venues` | nichts — `configs/venues_2026.yaml` gefüllt | `context` | liefert Distanz-Helper |
| `weather` (Open-Meteo) | **kein Key**, `weather.enabled: true` | `context` | nur für Fixtures mit `venue`+`date` (im `matches:`-Block von `fixtures_2026.yaml`) |

Der `context`-Layer liefert **Ruhetage** (immer berechenbar aus der Historie) plus
Wetter (wenn vorhanden). Reisedistanz ist vorbereitet (`venues.haversine_km`), aber
noch nicht je Spiel verdrahtet.

---

## Wichtige Hinweise (ehrlich)

- **Quoten & Wetter** haben keine Historie → als Trainingsmerkmal wirkungslos, bis du
  historische Dateien nachlegst (Odds-CSV) bzw. Fixtures mit `venue`+`date` füllst.
  Für die *kommenden* Spiele liefern sie Werte.
- **Namens-Aliase:** Kaggle/StatsBomb schreiben Länder anders (z. B. „Korea Republic")
  als martj42 („South Korea"). Eine Startliste steckt in
  `src/data_sources/team_aliases.py` — bei exotischen Schreibweisen dort ergänzen,
  sonst wird das Team nicht gematcht (Merkmal = 0).
- **Ablation:** Nach Aktivierung prüfen, ob die Holdout-Poisson-Deviance besser wird —
  sonst Feature-Flag wieder aus.

---

## Einmalige Systemvoraussetzung

- **macOS:** `brew install libomp` (OpenMP-Runtime für XGBoost) — bereits erledigt.

---

## Rechtliches (Kurzform)

martj42 (CC0), StatsBomb (non-commercial), Open-Meteo (CC-BY 4.0) → ok.
**Transfermarkt → nur manuell, kein Scraping.** Kaggle FIFA-Ranking → Dataset-Lizenz
beachten. The Odds API → ToS / Free-Tier.
