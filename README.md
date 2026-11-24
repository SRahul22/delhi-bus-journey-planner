# Delhi Bus Journey Planner (Transport Stack Hackathon 2026)

Google-Maps-style API: give two locations and a departure time, get bus journey options with estimated travel times. Travel times come from a traffic model built from October 2026 DTC bus GPS data.

**Source of truth:** `E:\hackathon-project` on the Windows machine (git repo). Raw data: `E:\hackathon-data` (read-only). Heavy processing output: `E:\hackathon-work`.

**Start here:** read this file, then [`docs/LOG.md`](docs/LOG.md) for what happened and what's next.

## How it works

```
GPS points (124 GB, E:\hackathon-data\bus-gps)
   │  model/scripts/process_day.py   (runs on Windows, 1 process per day)
   ▼
stop-to-stop travel times per trip
   │  model/scripts/aggregate.py
   ▼
model/segment_times.csv   median & p85 seconds per (route, stop pair, hour, weekday/weekend)
model/stop_headways.csv   observed gap between buses per (stop, route, hour, daytype)
model/route_patterns.csv  stop sequence of each route
   │
   ▼
planner/  (GTFS network + RAPTOR router + FastAPI + Leaflet map)
   │  + model/v3/  XGBoost ride-time correction (on by default, V3_MODE=xgb; V3_MODE=off disables)
   │  + realtime/  live GPS conditions (28 Oct replayed as a live feed, ?realtime=1)
   GET /journey?from=Kashmere Gate&to=Nehru Place&depart=2026-10-27T08:30
   GET /journey?from_lat=..&from_lng=..&to_lat=..&to_lng=..&depart=2026-10-20T09:00
```

Design: a **statistical base model** (historical medians by segment and time of day) with an **XGBoost correction** of each ride on top (`model/v3/XGB_FINAL.md`). Each option also reports the usual time for the same journey at a quieter hour (`quiet_hour`), so the effect of traffic is visible.

Validation: model built from 1–24 Oct; 25–31 Oct held out. `planner/evaluate.py` compares predictions against actual held-out rides and against Google Maps times.

## Layout

| Path | What |
|---|---|
| `model/` | GPS → travel-time model. Details: [`model/README_model.md`](model/README_model.md) |
| `model/scripts/` | Processing scripts (run on Windows) |
| `planner/` | Router, API, map UI, evaluation. Details: [`planner/README.md`](planner/README.md) |
| `docs/LOG.md` | Progress log, decisions, open issues, next steps |
| `model/v3/` | Final XGBoost ride-time correction. Details: [`model/v3/XGB_FINAL.md`](model/v3/XGB_FINAL.md) |
| `realtime/` | Real-time layer (replayed live feed). Details: [`realtime/README.md`](realtime/README.md) |
| `eval/` | Test pairs, actual-ride and Google Maps comparison. Details: [`eval/README.md`](eval/README.md) |
| `docs/report/report.pdf` | Technical note and validation report |
| `sync.sh` | Old Mac→Windows sync script. Not used any more: commit directly in this repo |

## Data (on Windows, `E:\hackathon-data`)

| Dataset | Notes |
|---|---|
| `bus-gps\bus_movements_2026_10_DD.db` | 31 SQLite files, ~29 M rows/day, ~10 s pings, ~6000 buses. Table `vehicle_feed(vehicle_id, lat, lng, speed, route_id, trip_id, timestamp, date, time)`. GPS `route_id` matches GTFS `route_id` (99%). |
| `gtfs\` | Static GTFS: 6,862 stops, 2,573 routes, 80k trips, shapes |
| `ticketing\October-Delhi.csv` | 2.56 M tickets (route, boarding stop, alighting stop, time). Used to pick realistic test pairs. |
| `README.md` | Official data README |

Rules: never write to `E:\hackathon-data`. Never copy raw GPS to a laptop (keep transfers under 5 GB).

## Run the planner

Needs Python 3.10+. One command does everything (creates `planner/.venv` from `planner/requirements.txt`,
builds the GTFS network cache on first run, starts the API on port 8000):

```bat
planner\run.bat            :: Windows  (set PORT=8080 first to change the port)
```
```bash
cd planner && ./run.sh     # macOS / Linux
```

Then open http://127.0.0.1:8000 (map UI) or http://127.0.0.1:8000/docs (interactive API docs). Try:

```
http://127.0.0.1:8000/journey?from=Kashmere%20Gate&to=Nehru%20Place&depart=2026-10-27T08:30
```

Manual setup, if you prefer (from `planner/`):
```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python preprocess.py
.venv\Scripts\python -m uvicorn app:app --port 8000
```
`requirements.txt` pins `xgboost==2.1.4`. Without xgboost the planner still runs but skips the final XGBoost
correction; `GET /health` shows which is active (`ride_correction`).

Served defaults ("config C", since 2026-11-24): XGBoost ride correction only on bus legs of 60 min or more
(`V3_MIN_RIDE_S=3600`) and pooled waits over parallel routes (`V2_WAIT=pooled`) from the stop-visits index
`POOLED_INDEX` (default `E:\hackathon-work\v2\planner\pooled_index_visits_1-24.npz`, built by
`planner/build_pooled_index.py --source visits --days 1-24`) plus `stop_headways_v2.csv`. Without these files the
planner uses v1 waits. Roll back with `set V2_WAIT=v1` and `set V3_MIN_RIDE_S=0` before starting. `GET /health`
shows the active values under `flags`.

### Inputs the planner needs

| Input | Where | In git? | If missing |
|---|---|---|---|
| Static GTFS | `GTFS_DIR`, default `../data/gtfs`, then `E:\hackathon-data\gtfs` | No (official data) | Required |
| Model tables `model/segment_times.csv`, `stop_headways.csv`, `route_patterns.csv`, `holdout_trips.csv` (~133 MB) | `MODEL_DIR`, default `model/` | No (too big, gitignored) | Planner falls back to timetable times × a fitted hour-of-day factor, much less accurate |
| XGBoost model `model/v3/xgb_final.json`, `v3_lookup.npz` | `model/v3/` | Yes | Correction skipped |
| Real-time replay `realtime/data/` | `realtime/` | No (gitignored) | `?realtime=1` answers with historical times only |

To regenerate the model tables from the raw GPS (about 1 hour on the Windows machine), follow
[`model/README_model.md`](model/README_model.md) "How to run": `build_patterns.py` → `run_all.py` (per-day
`process_day.py`) → `aggregate.py`. The real-time replay is built by `realtime/scripts/extract_replay.py`.
To hand the system to someone without the raw GPS, copy the four model CSVs into their `model/` folder.

## Working together

- Two people + Claude agents. Code started on the Mac and was synced with `sync.sh`.
- **Edit here on Windows and commit with git. Don't use `sync.sh`** (it would overwrite changes made here).
- Log progress and decisions in `docs/LOG.md`.
