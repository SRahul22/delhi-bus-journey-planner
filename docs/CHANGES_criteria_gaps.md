# Changes: closing gaps against the Problem 1 brief

Branch `fix/criteria-gaps` (from `main` at `2c0bc3c`), 2026-11-24. The `v2/*` branches are untouched.

An audit of `main` against the Problem 1 / Phase 1 brief found three gaps, apart from deployment (left for later)
and the small Google Maps sample (18 of 60 pairs, needs manual lookups). This branch closes those three.

| # | Brief requirement | Gap on `main` | Fix |
|---|---|---|---|
| 1 | §2.2: "say how long the same trip would usually take at a quieter hour, so the user can see the effect of traffic" | Not in the API or the UI | Each option now has `quiet_hour`. The UI shows it as a line under each option. |
| 2 | §2.1: "accept an origin and a destination, as coordinates **or stop names**" | `/journey` only took `from_lat/from_lng/to_lat/to_lng`. Names only worked in the UI. | `/journey` accepts `from=` / `to=`, each a stop name or `"lat,lng"` |
| 3 | §4: "code and instructions that let someone else run your system" | The root README had out-of-date run steps: a Mac `run.sh`, and a pip list without `xgboost`, so the final model was silently skipped. It did not say the 133 MB model tables are gitignored or how to get them. | README run section rewritten, plus a table of the inputs the planner needs. `/health` now reports whether the XGBoost correction is active. |

## 1. Quiet-hour comparison (`planner/router.py`, `planner/app.py`, `planner/static/index.html`)

- `Planner.plan()` keeps each candidate's leg skeleton (the same routes, boarding and alighting stops). For each
  **returned** option, `_quiet_hour()` re-times that skeleton with `time_journey()` at departures
  `QUIET_HOURS = (6, 11, 15, 22)` on the same date, so the weekday or weekend type is the same. It keeps the hour with the least
  time on the bus.
- New field per option:
  ```json
  "quiet_hour": {"depart": "2026-10-27T06:00:00+05:30", "duration_s": 2580, "ride_s": 2100, "wait_s": 300,
                 "traffic_delay_s": 960, "note": "same routes and stops, leaving at the quietest of 06:00, 11:00, 15:00, 22:00 on the same day"}
  ```
  `traffic_delay_s` is the time on the bus now minus the time at the quiet hour. It is the traffic effect: total time also
  includes waiting, which depends on headways rather than traffic. A negative value means the requested time is itself
  the quieter one. It is `null` for a walk-only option.
- With `?realtime=1`, the reference is always timed with the **historical** planner (`quiet_planner=planner`). The
  live model applies its ratio to hours before the replay clock too, which would distort a 06:00 reference.
- `plan(..., compare_quiet=False)` turns the comparison off for batch callers that do not need it.
- UI: a yellow line under each option, for example: *"At a quieter hour (06:00): 43 min door to door, 35 min on the bus
  vs 51 min now · traffic adds ~16 min now"*.

## 2. Stop names in the API (`planner/app.py`)

- `GET /journey?from=Kashmere Gate&to=Nehru Place&depart=2026-10-27T08:30` works. Each endpoint is a stop name (best
  match from the existing `/stops/search` ranking: prefix match first, then the shortest name) or `"lat,lng"`.
- `from_lat/from_lng/to_lat/to_lng` still work and win when both forms are given, so existing clients and
  `evaluate.py` are unaffected.
- The matched stop is echoed in `query.resolved` (`name`, `stop_id`, `lat`, `lng`), so a client can see which
  "Kashmere Gate" was picked.
- Errors: no match → 404 with a hint to use `/stops/search`; endpoint missing → 400; outside Delhi → 400 (as before).

## 3. Run instructions (`README.md`, `planner/README.md`)

- Root README "Run the planner": `planner\run.bat` / `run.sh` (they install `requirements.txt`, which pins
  `xgboost==2.1.4`), the `/docs` OpenAPI page, an example query, and manual steps with `pip install -r requirements.txt`.
- A new "Inputs the planner needs" table covers GTFS, the four model CSVs (gitignored, ~133 MB), the XGBoost files (in git)
  and the realtime replay (gitignored). For each it says what happens if it is missing and how to regenerate it.
- Removed out-of-date statements: "deterministic model, not ML", and "use `sync.sh`".
- `planner/README.md`: documents `from`/`to`, `query.resolved`, `quiet_hour`, `/docs` and `ride_correction`.
- `GET /health` has a new field `ride_correction`, for example `"model v3 (xgb)"` or `"off"`.

## Verification (Windows, test server on port 8001; production server on 8000 left running)

| Check | Result |
|---|---|
| `/health` | `ride_correction: "model v3 (xgb)"` |
| `from=Kashmere Gate&to=Nehru Place`, Tue 27 Oct 08:30 | Resolved to Kashmere Gate City Bus Terminal → Nehru Place. 3 options. 425 direct: 59 min (51 on the bus); at 06:00: 43 min (35 on the bus), traffic +16 min |
| Same trip at 23:00 (`from` given as `"lat,lng"`) | 425: 39 min on the bus vs 35 at the quiet hour, traffic +5 min |
| Old coordinate params | Same options as before, with `quiet_hour` added |
| `?realtime=1` | `realtime.applied: true`, `quiet_hour` present (historical) |
| Unknown name / missing endpoint / outside Delhi | 404 / 400 / 400 with clear messages |
| Latency, same query, old vs new server | ~0.68–0.93 s vs ~0.67–0.74 s: no measurable cost |
| `planner/tests/sanity.py` (4 pairs × 4 times) | All plan normally, median 806 ms |
| UI JavaScript | `node --check` passes; the quiet-hour line renders from real API output |

## Not done here

- Deployment or public access (by request: later).
- Google Maps times for the remaining 42 of the 60 pairs (needs manual lookups; template in `eval/google_pairs_todo.csv`).
- The running server on port 8000 still serves `main`. To demo this branch, restart it from this branch.
