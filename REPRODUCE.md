# Reproducing the Delhi Bus Journey Planner

How to get from the official raw data to the running API and to every number we report, on Linux, macOS or
Windows. The original build ran on one Windows machine (20 logical CPUs, 64 GB RAM, RTX 3090, Python 3.12.2);
paths default to that machine (`E:\hackathon-data`, `E:\hackathon-work`) and can be changed with environment
variables (section 2).

There are three levels. Pick the one you need:

| Level | You need | Gets you | Time |
|---|---|---|---|
| A. Run the API | repo + GTFS (+ the 4 model CSVs, optional) | `/journey` answers, map UI | 5 min |
| B. Recompute the evaluation numbers | A + the intermediates (`WORK_DIR`, ~7 GB) | every table in section 6 from existing files | minutes per table |
| C. Full rebuild from raw GPS | A + 140 GB raw data, 64 GB RAM | all intermediates, model tables, ML model | ~2–3 h wall |

## 1. Prerequisites

- Python 3.12 (tested 3.12.2; the planner alone runs on 3.10+). Git. 7-Zip (`7z`) to unpack the GPS archive.
- Disk: 16 GB for the GPS archive + 124 GB unpacked; ~7 GB for intermediates (`WORK_DIR`); ~0.5 GB for the rest.
- RAM: 64 GB recommended for level C (`aggregate.py` holds 35 M stop passings; `process_day.py` needs 2–3 GB per
  parallel process). The API needs ~1.5 GB (~2 GB with the realtime layer).
- GPU: optional. Only `model/v3/scripts/xgb_final.py` (training the final XGBoost) uses CUDA; set `XGB_DEVICE=cpu`
  without an NVIDIA GPU (results can differ in the last decimal). Serving and everything else is CPU only.
- Shell: every Python script runs on all three OSes. `model/scripts/launch.py` (detached start that survives an SSH
  drop) is Windows-only; on Linux/macOS use `nohup … &` instead.

## 2. Data and folder layout

Downloads (Transport Stack / IIIT Delhi, see the official data README, reproduced as `E:\hackathon-data\README.md`):

| File | Size | SHA-256 |
|---|---|---|
| `delhi-dtc-bus-gtfs-rt-2026-10.7z` (bus GPS, 31 SQLite DBs) | 16.1 GB | `703bd6ea5237883e1ad8f094a661d365edc74b7555cf83ce4c554408a79c16c9` |
| `October-Delhi.csv.zip` (ticketing) | 66 MB | `3b0d051b13a7082697df1c0f03140611350e50b574b44e21d1fb62d4f23df7be` |
| `delhi_buses_static_gtfs_v1.zip` (static GTFS, https://delhi.transportstack.in) | 30 MB | `8eb05d9c21e7a6b42d7b49de45cb095b70ddee5b8f7716859330c9784c322f05` (computed by us; not in the official SHA256SUMS) |

Verify with `shasum -a 256 -c SHA256SUMS` (macOS), `sha256sum -c SHA256SUMS` (Linux) or
`certutil -hashfile <file> SHA256` (Windows). Unpack into this layout (`DATA_DIR`, read-only after this):

```
DATA_DIR/                         default E:\hackathon-data
  bus-gps/bus_movements_2026_10_01.db … _31.db    7z x delhi-dtc-bus-gtfs-rt-2026-10.7z -obus-gps
  gtfs/{agency,calendar,feed_info,routes,shapes,stop_times,stops,trips}.txt   unzip, drop __MACOSX/
  ticketing/October-Delhi.csv
```

Opening the DBs read-only (`mode=ro`) still creates empty `-shm`/`-wal` files next to them; that is harmless.

Environment variables (all optional; defaults keep the original Windows machine working):

| Variable | Default | Used by |
|---|---|---|
| `DATA_DIR` | `E:\hackathon-data` | GPS processing, ticket aggregation, replay extraction |
| `WORK_DIR` | `E:\hackathon-work` | every intermediate (pass files, v2 data, v3 parquet, replays, eval outputs) |
| `GTFS_DIR` | `<repo>/data/gtfs`, then `DATA_DIR/gtfs` (`E:/hackathon-data/gtfs`) | planner, eval, stop_visits |
| `MODEL_DIR` | `<repo>/model` (planner); `E:\hackathon-project\model` (eval scripts) | model CSVs |
| `POOLED_INDEX` | `WORK_DIR/v2/planner/pooled_index_visits_1-24.npz` | served pooled waits (config C) |
| `V2_HEADWAYS` | `MODEL_DIR/v2/stop_headways_v2.csv`, then `WORK_DIR/v2/data/stop_headways_v2.csv` | served pooled waits |
| `NET_PATH` | `E:\hackathon-project\planner\cache\network.pkl` | eval scripts that load the planner network |
| `V2_EVAL_OUT`, `V2_DATA_DIR`, `RT_OUT_DIR`, `RT_V2_OUT` | under `WORK_DIR` | eval / realtime outputs |
| `XGB_DEVICE` | `cuda` | `xgb_final.py` (`cpu` without a GPU) |
| `XGB_FINAL_COMMIT` | `0` | `xgb_final.py`: `1` git-commits its outputs, as the original run did |

On Linux/macOS set at least `DATA_DIR`, `WORK_DIR` and `GTFS_DIR`, e.g.

```bash
export DATA_DIR=$HOME/delhi-data WORK_DIR=$HOME/delhi-work GTFS_DIR=$HOME/delhi-data/gtfs
```
```bat
:: Windows cmd (PowerShell: $env:DATA_DIR = "D:\delhi-data")
set DATA_DIR=D:\delhi-data
set WORK_DIR=D:\delhi-work
set GTFS_DIR=D:\delhi-data\gtfs
```

## 3. Environment

Two pinned files, both taken from the build machine (`pip freeze`, Python 3.12.2):

- `requirements-pipeline.txt` (repo root): everything (GPS processing, ML, eval, planner, tests).
- `planner/requirements-lock.txt`: exact copy of the API venv. `planner/requirements.txt` (used by `run.sh` /
  `run.bat`) is unpinned except `xgboost==2.1.4`.

```bash
python3.12 -m venv .venv-repro
.venv-repro/bin/pip install -r requirements-pipeline.txt
PY=$PWD/.venv-repro/bin/python
```
```bat
py -3.12 -m venv .venv-repro
.venv-repro\Scripts\pip install -r requirements-pipeline.txt
set PY=%CD%\.venv-repro\Scripts\python.exe
```

`xgboost` must be 2.1.4: on the build machine 3.x silently trains on CPU. On macOS, xgboost and lightgbm wheels need
`brew install libomp`.

## 4. Full rebuild (level C)

Commands below run from the repo root. `DD` lists are written in full in the real commands
(`01,02,03,…,31`; a helper: `seq -w -s, 1 31` in bash, or copy the lists below).

```
D1_24=01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
D25_31=25,26,27,28,29,30,31
D1_18=01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18
```

| # | Stage | bash (Linux/macOS) | Output (size) | Time on the build machine |
|---|---|---|---|---|
| 1 | Route patterns + shapes from GTFS | `$PY model/scripts/build_patterns.py $GTFS_DIR model && mkdir -p model/work $WORK_DIR && cp model/route_shapes.csv model/route_patterns.csv $WORK_DIR/ && mv model/route_shapes.csv model/work/` | `model/route_patterns.csv` (1.6 MB), `route_shapes.csv` (26 MB) in `WORK_DIR` and `model/work` | 6 s |
| 2 | GPS → stop passings, 31 days, 10 parallel | `$PY model/scripts/run_all.py 10 $D1_24,$D25_31` (skips days already done; per-day log `WORK_DIR/log_DD.txt`) | `WORK_DIR/pass/pass_DD.npz` (~40 MB/day, 1.2 GB) | ~47 min (5 min/day alone, I/O bound) |
| 3 | Model tables, train 1–24 | `$PY model/scripts/aggregate.py $D1_24 - $WORK_DIR/out_train2` | `segment_times.csv` (81 MB), `stop_headways.csv` (49 MB), `stats.json` | ~3 min |
| 4 | Holdout rides 25–31 (seed 42) | `$PY model/scripts/aggregate.py "" $D25_31 $WORK_DIR/out_hold` | `holdout_trips.csv` (20k rides, 0.9 MB) | ~1 min |
| 5 | Install the tables for the planner | `cp $WORK_DIR/out_train2/* $WORK_DIR/out_hold/holdout_trips.csv model/` | `model/*.csv` (gitignored) | |
| 6 | Raw stop visits (v2 waits, realized journeys) | `$PY model/scripts/run_visits.py 8 $D1_24,$D25_31` | `WORK_DIR/v2/data/stop_visits_DD.npz`, `route_trips_DD.csv` (~3 GB total) | not logged precisely (~1 h) |
| 7 | Passing index (needs 2 and 6) | `$PY model/scripts/passing_index.py $D1_24,$D25_31` | `WORK_DIR/v2/data/passing_index_DD.npz` | minutes |
| 8 | Headways v2 (holidays 02,20 pooled with weekends) | `$PY model/scripts/build_headways_v2.py $D1_24 $WORK_DIR/v2/data/stop_headways_v2.csv` and `… $D1_18 $WORK_DIR/v2/data/stop_headways_v2_1_18.csv` | `stop_headways_v2*.csv` | minutes |
| 9 | Pooled-wait index (served: 1–24; validation: 1–18) | `$PY planner/build_pooled_index.py --source visits --days 1-24` and `--days 1-18` | `WORK_DIR/v2/planner/pooled_index_visits_1-24.npz` (~300 MB) | minutes |
| 10 | v2 tables with holiday day type (validation only, not served) | `$PY model/scripts/run_tables.py` | `WORK_DIR/v2/data/tables_1_18`, `tables_1_24` | ~6 min |
| 11 | v3 ride datasets (out-of-fold, see `model/v3/README.md`) | `$PY model/v3/scripts/build_rides.py <18 lookup days> <6 ride days> foldK 2 300000` for K=1..4, then `$PY model/v3/scripts/build_rides.py $D1_24 $D25_31 test 2 600000 --export` | `WORK_DIR/v3/fold1..4.parquet` (25 MB each), `test.parquet` (52 MB), `v3_lookup.npz` | ~30 s each |
| 12 | Final XGBoost (train 1–18, val 19–24, refit 1–24) | `XGB_DEVICE=cuda $PY model/v3/scripts/xgb_final.py` then `cp $WORK_DIR/v3/v3_lookup.npz model/v3/` | `model/v3/xgb_final.json`, `xgb_final_meta.json`, `XGB_FINAL.md` (committed in git) | minutes on GPU |
| 13 | Optional: LightGBM comparison table | `$PY model/v3/scripts/train.py` | `WORK_DIR/v3/results.txt` etc. (`model/v3/results.txt`) | 3–8 min |
| 14 | Replay files for the realtime layer | `$PY realtime/scripts/extract_replay.py 28`; for v2: `RT_OUT_DIR=$WORK_DIR/v2/realtime $PY realtime/scripts/extract_replay.py 22` (and `30`) | `WORK_DIR/realtime/replay_28.npz` (73 MB) | ~3.5 min/day |
| 15 | Ticket aggregation (test-pair selection) | `$PY eval/agg_tickets.py` | `WORK_DIR/eval/od_route.csv`, `stop_names.csv` | ~40 s |

Windows: the same commands with `%PY%`, `%WORK_DIR%`, `copy`/`move` instead of `cp`/`mv`, and the day lists written
out. To survive an SSH disconnect, wrap long stages: `%PY% model\scripts\launch.py %WORK_DIR%\run_all.log
model\scripts\run_all.py 10 01,02,…,31` (detached; `launch.py` runs the job with `WORK_DIR` as its working directory,
so give it absolute script paths).

Notes:
- Stage 3 output is what `model/segment_times.csv` is today (md5-identical to `E:\hackathon-work\out_train2`; the
  older `out_train` lacks the `mean_s` column). Leave `HOLIDAYS` unset for the served v1 tables.
- Stage 11's day lists for folds 1–4 are in `model/v3/README.md` ("How to run").
- Realtime (`realtime/feed.py`) looks for `realtime/data/replay_28.npz` first, then `WORK_DIR/realtime/replay_28.npz`.
- `model/scripts/diag.py`, `model/v3/scripts/make_check.py`, `fetch.sh`, `planner/bench_all.py`,
  `planner/validate_on_time.py` and `model/v3/v2/scripts/*` are one-off analysis scripts that still hard-code
  `E:\` paths; they are not needed for any headline number.

## 5. Run the API (level A)

Minimum: the repo and GTFS. Everything else degrades gracefully:

| Input | Default location | If missing |
|---|---|---|
| GTFS | `GTFS_DIR` | required |
| `model/segment_times.csv`, `stop_headways.csv`, `route_patterns.csv`, `holdout_trips.csv` (stage 5) | `MODEL_DIR` = `model/` | GTFS schedule × hour factor (much less accurate: 32.5% vs 12.6% ride error) |
| `model/v3/xgb_final.json`, `v3_lookup.npz` | in git | correction skipped (`/health` → `ride_correction`) |
| `pooled_index_visits_1-24.npz` (stage 9) + `stop_headways_v2.csv` (stage 8) | `POOLED_INDEX`, `V2_HEADWAYS` (defaults under `WORK_DIR`) | v1 waits (headway/2 from `stop_headways.csv`) |
| `replay_28.npz` (stage 14) | `realtime/data/` or `WORK_DIR/realtime/` | `?realtime=1` answers with historical times; set `REALTIME=0` to skip loading |

Served defaults ("config C", since 2026-11-24): `V3_MODE=xgb` with `V3_MIN_RIDE_S=3600` (XGBoost correction only on
bus legs of 60 min or more), `V2_WAIT=pooled`, `SEG_MODE=blend`, `BLEND_S=600`. `GET /health` shows them under
`flags`. Roll back to the pre-config-C behaviour with `V2_WAIT=v1 V3_MIN_RIDE_S=0`.

```bash
cd planner
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
.venv/bin/python preprocess.py                         # cache/network.pkl (17 MB), ~10 s
PORT=8000 ./run.sh                                     # or: .venv/bin/python -m uvicorn app:app --port 8000
curl 'http://127.0.0.1:8000/journey?from=Kashmere%20Gate&to=Nehru%20Place&depart=2026-10-27T08:30'
```
```bat
cd planner
py -3.12 -m venv .venv && .venv\Scripts\pip install -r requirements-lock.txt
.venv\Scripts\python preprocess.py
set PORT=8000 && run.bat
curl "http://127.0.0.1:8000/journey?from=Kashmere%20Gate&to=Nehru%20Place&depart=2026-10-27T08:30"
```

Then open http://127.0.0.1:8000 (map UI) or `/docs`. Tests: `.venv/bin/python -m pytest tests -q` (planner),
`python -m unittest discover eval/tests` (metrics). Startup takes ~15–30 s with the realtime layer and the pooled index.

## 6. Results → command → output

Run from the repo root with the pipeline env (`$PY`) unless "planner venv" is said. All need the files of the stages
in the second column. Seeds are fixed in the code (holdout 42, ride sampling `day*7+2`, random ODs 2026, realtime 0),
so reruns on the same inputs give the same numbers, except the GPU XGBoost fit (tiny differences possible).

| Reported result (where) | Needs | Command | Output |
|---|---|---|---|
| Model coverage 90.5%, 33.4 M segment obs (README_model, report) | 3 | printed by stage 3 | `out_train2/stats.json` = `model/stats.json` |
| Holdout ride error, 20k rides, pure segment sums: median 13.2%, mean 13.9% (README_model) | 5 | `$PY model/scripts/eval_holdout.py model median_s` (and `mean_s`) | stdout |
| Planner ride error on 20k holdout: blend 12.6% / 68.4%, schedule 32.5% (planner README, report) | 5 | planner venv: `cd planner && .venv/bin/python evaluate.py` (`SEG_MODE`, `BLEND_S` for the other rows) | `planner/eval_out/holdout_predictions.csv`, `holdout_by_hour.csv`; stdout |
| 600k test rides: sum of medians 12.69%, blend 12.45%, XGBoost 12.28% / 69.0%, GTFS 27.9% (report, XGB_FINAL.md) | 11, 12 | `$PY eval/ride_baselines.py --set test` (and stage 12 prints the XGBoost row) | `WORK_DIR/v2/eval/rides_test_summary.md`, `rides_test_metrics.csv`; `model/v3/XGB_FINAL.md` |
| LightGBM 12.32% and calibration 12.43% (model/v3/README) | 11 | stage 13 | `model/v3/results.txt` |
| 60 test pairs: 60/60 answered; ride vs actual 6.7% median APE, 84% within ±20% (eval/README, LOG 18:05) | 2, 5, API running | `$PY eval/actual_rides.py`; planner venv: `$PY eval/run_planner.py --url http://127.0.0.1:8010`; `$PY eval/compare.py` | `eval/actual_rides.csv`, `planner_results.csv`, `comparison.csv` |
| Test-pair selection (60 pairs from tickets) | 15 | `$PY eval/build_pairs.py --work $WORK_DIR/eval` | `eval/test_pairs.csv`, `google_pairs_todo.csv`, `match_report.txt` |
| Google Maps, 30 pairs: like-for-like 19.1% (eval/GOOGLE_RESULTS.md) | 5 | planner venv: `cd eval && ../planner/.venv/bin/python gcmp.py` | `eval/google_comparison.csv`; stdout |
| 18 pairs vs actual: ours 9.6%, Google 16.3%, schedule 21.2% | `google_comparison.csv`, `comparison.csv` | **no script**: computed by hand from those two files (join on pair id, pairs with `n_obs ≥ 3`) | GOOGLE_RESULTS.md §2 |
| Realtime: ride 13.0% → 12.3%, next bus 512 s → 110 s (realtime/README, v1.5 tables) and 503 s → 107 s (report, full tables) | 5, 14 | planner venv: `$PY realtime/validate.py` (~90 s) | `realtime/results/validation.json` |
| v2 live layer: KF, next bus 28/30 Oct (LOG 19:48, handoff T3.1/T3.3) | 5, 14 (22, 28, 30) | planner venv: `$PY realtime/v2_live_eval.py 22 --fit`, then `28`, `30` | `WORK_DIR/v2/realtime/results_DD.json`, `kf_params.json` |
| Noise floor 10.28% val / 10.30% test | 2, 11 | `$PY eval/noise_floor.py --set val` / `--set test` | `WORK_DIR/v2/eval/noise_floor_<set>.json` |
| Realized door-to-door, validation 19–24 (5,240 OD-times) | 5, 6, 7, planner cache | `$PY eval/realized_journeys.py --set val --workers 3` with the config's env (below) | `WORK_DIR/v2/eval/realized_val.csv` |
| v2 validation report (docs/results/v2_2026-11-24_val.md) | 11, realized_val, noise floor | `$PY eval/val_blend.py`; `$PY eval/ride_baselines.py --set val`; `$PY eval/report.py --set val` | `docs/results/v2_<date>_val.md` |
| Rerun A/B/C (docs/results/integration/rerun_val_ABC.md) | realized runs per config | for each config: `V2_EVAL_OUT=$WORK_DIR/v2/rerun/<cfg>` + flags (A: `V3_MIN_RIDE_S=0 V2_WAIT=v1`; B: `V3_MIN_RIDE_S=3600 V2_WAIT=v1`; C: `V3_MIN_RIDE_S=3600 V2_WAIT=pooled POOLED_INDEX=…/pooled_index_visits_1-18.npz V2_HEADWAYS=…/stop_headways_v2_1_18.csv`) `$PY eval/realized_journeys.py --set val`; then `$PY eval/rerun_compare.py A=$WORK_DIR/v2/eval/realized_val.csv B=…/rerun/B/realized_val.csv C=…/rerun/C/realized_val.csv C24=…/rerun/C24/realized_val.csv` | stdout (markdown) |
| d2d floor / bias / live (docs/results/integration/d2d_floor_bias_live.md) | rerun files, replay_22 | `$PY eval/d2d_plan_extra.py`, `d2d_oracle.py`, `d2d_calib.py`, `d2d_live.py` | `WORK_DIR/v2/analysis/*.json, *.csv` |
| Latency (LOG 20:05, 22:45) | API env | planner venv: `cd planner && .venv/bin/python bench_latency.py` | stdout / json |

## 7. Verification log

2026-11-24, on the build machine, from a fresh `git clone` of this branch and a new venv from
`requirements-pipeline.txt` (Python 3.12.2), pointing at the existing intermediates:

- `aggregate.py 01..24 - <new dir>` re-run from the stage 2 pass files: `segment_times.csv` and `stop_headways.csv`
  are byte-identical (md5) to the served `model/` files; coverage 90.5%, 33,417,974 segment observations.
- `planner/preprocess.py` + API on a side port with the served defaults: `/health` flags match production
  (`V2_WAIT=pooled`, `V3_MIN_RIDE_S=3600`, pooled index resolved from `WORK_DIR`); Kashmere Gate → Nehru Place
  (27 Oct 08:30) returns the same three options and durations as production, 0.3–0.5 s per query.
- `planner/evaluate.py` (20k holdout): 12.6% median APE, 68.4% within ±20%, bias −10.5 s (documented: 12.6% / 68.4% / −11 s).
- `eval/rerun_compare.py` on the A/B/C/C24 files: the common-subset table matches `rerun_val_ABC.md` exactly
  (C 24.4% / 42.0% / −871 s).
- `pytest planner/tests eval/tests`: 32 passed.
- `eval/actual_rides.py` (now local, no SSH) reproduces `eval/actual_rides.csv` byte for byte (38/50 direct pairs);
  `eval/compare.py`: 6.7% median APE, 84% within ±20%, bias +2.0% (as documented).
- `process_day.py 15 300000` (row-limited dry run) with a fresh `WORK_DIR` holding only `route_shapes.csv` and
  `route_patterns.csv`: runs in 3 s and writes `pass/pass_15_test.npz` (`routes.txt` now falls back to `GTFS_DIR`).
- Not re-run: the full 31-day GPS pass (~47 min), stop visits, v3 rides/XGBoost training, replay extraction,
  realized journeys (each 10+ min); their commands were checked against the scripts' argument parsing.

## 8. What cannot be reproduced exactly

- **Google Maps times** were looked up by hand in the Google Maps web UI (no API key), for Wed 2026-11-25 at each
  pair's clock time, because Google cannot plan in the past. Google's live timetable changes, so a new lookup will
  differ. Some of Google's fastest options use the Metro (noted in `google_notes`). The values used are frozen in
  `eval/google_pairs_todo.csv`.
- **The "18 pairs vs actual" table** (ours 9.6%, Google 16.3%, schedule 21.2%) has no script; it was computed ad hoc
  from `google_comparison.csv` and `comparison.csv`.
- **The realtime layer is simulated**: a replay of held-out days (28 Oct; 22 and 30 for v2) behind a simulated clock,
  not a live feed. Its numbers are reproducible from the replay files, but they are not live-feed numbers.
- **realized_test.csv** (door-to-door on test days 25–31) was lost in a reboot at ~20:09 IST and was never
  reported; any test-week d2d numbers for later configs (e.g. D/F) may be missing or incomplete. All reported
  door-to-door numbers are on validation days 19–24.
- **eval/README "Findings"** were measured against model v1.5 (11 training days). The numbers with the full 1–24
  model are in `docs/LOG.md` (18:05) and `planner/tests/eval_compare_full.txt`.
- **GPU training**: XGBoost `hist` on CUDA vs CPU can differ slightly; the served model is the committed
  `model/v3/xgb_final.json`, so the API does not depend on retraining.
- Runtimes are from the build machine while other jobs were running; expect variation.
