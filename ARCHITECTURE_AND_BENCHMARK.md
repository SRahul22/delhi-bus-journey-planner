# Delhi Bus Journey Planner: architecture and benchmark

A Google-Maps-style API for Delhi buses: give two locations (or stop names) and a departure time, get up to three bus journey options with door-to-door time estimates and a heavy-traffic range. It is built from 31 days of DTC bus GPS (October 2026) and the Delhi static GTFS. State as of 2026-11-27; `main` serves **config F**.

## 1. Architecture

```
 RAW DATA (E:\hackathon-data, read-only)
 ├─ bus-gps\bus_movements_2026_10_DD.db   31 SQLite files, ~29 M pings/day, ~10 s apart, ~6,000 buses
 ├─ gtfs\                                 2,574 routes, 6,862 stops, 80k trips, shapes
 └─ ticketing\October-Delhi.csv           2.56 M tickets (used for evaluation only)
            │
            ▼
 1. ROUTE MODEL (model/scripts/build_patterns.py)
    one stop pattern per route_id (most common trip), stop distances along the GTFS shape
            │
            ▼
 2. GPS → STOP PASSINGS (model/scripts/process_day.py, one process per day)
    project pings onto the route shape → distance along route; drop pings >150 m off, backward jumps,
    terminal dwell, >70 km/h, gaps >300 s; interpolate the time each stop was passed (~68% of trips usable)
            │                                         │
            ▼                                         ▼
 3a. RIDE-TIME TABLES (aggregate.py, days 1–24)   3b. STOP VISITS (stop_visits.py, raw pings)
    median / mean / p85 seconds per                    every bus of a route within 60 m of its stop
    (route, stop pair, hour, weekday|weekend)          (track densified every 40 m) → departure gaps
    → model/segment_times.csv (90.5% coverage)         → stop_headways_v2.csv (expected wait per stop/route/hour)
            │                                         │
            ▼                                         ▼
 4. TRAVEL-TIME MODEL (planner/ttmodel.py)
    ride: route segment (n≥3) → same stop pair any route → schedule × hour factor → distance/15 km/h;
          median for the first 10 min of a ride, mean after (skewed delays)
    ride correction: XGBoost (model/v3, GPU-trained) on legs ≥ 60 min only (V3_MIN_RIDE_S=3600)
    wait (config F): mean(H²)/(2·mean H) from observed gaps (≥3 gaps), cap 60 min, long wait for hours
          with no bus, timetable wait × learned factor for stops with no GPS
            │
            ▼
 5. JOURNEY PLANNER (planner/router.py)
    frequency-based RAPTOR, up to 3 transfers; walk to/from stops within 800 m (wider if none),
    transfers between stops within 400 m (1.2 m/s × 1.3 detour); candidates = fastest, walk-averse,
    route-banning diversity runs; re-timed hour by hour; ranked with a 10-min transfer penalty
            │
            ▼
 6. API + MAP (planner/app.py, planner/static/index.html)
    FastAPI: /journey (coords or stop names, depart, arrive_by, realtime=1), /stops/search, /health, /docs
    Leaflet map: click or search stops, options drawn on the map, quiet-hour comparison, live-traffic switch
            ▲
 7. LIVE LAYER (realtime/)  replays 28 Oct as a live feed; per-segment slow-down ratios (Kalman option,
    V2_LIVE=kf) and next-bus ETAs from live bus positions; used with ?realtime=1
```

| Component | Key files | Output / role |
|---|---|---|
| Route model | `model/scripts/build_patterns.py` | `route_patterns.csv` (2,574 routes, 82k rows), stop positions along shapes |
| GPS processing | `model/scripts/process_day.py`, `run_all.py` | `E:\hackathon-work\pass\pass_DD.npz`: ~1.5 M stop passings/day, 35 M total |
| Ride tables | `model/scripts/aggregate.py` | `segment_times.csv` (1.8 M rows), `stop_headways.csv` |
| Stop visits / waits | `model/scripts/stop_visits.py`, `build_headways_v2.py` | `stop_visits_DD.npz`, `stop_headways_v2.csv` (1.68 M rows) |
| ML ride correction | `model/v3/scripts/xgb_final.py`, `model/v3/predict.py` | `xgb_final.json` (XGBoost 2.1.4, 100 trees) |
| Travel-time model | `planner/ttmodel.py`, `planner/v2ext.py` | ride and wait lookups (dense arrays per route) |
| Planner | `planner/router.py`, `planner/preprocess.py` | RAPTOR over a cached GTFS network (`cache/network.pkl`) |
| API and UI | `planner/app.py`, `planner/static/index.html` | FastAPI on port 8000; Leaflet map |
| Live layer | `realtime/` | replay feed, live ratios, next-bus ETA |
| Evaluation | `eval/`, `planner/evaluate.py` | metrics, realized journeys, Google comparison |

**Served configuration (config F):** `V2_WAIT=gaps`, `V2_WAIT_MIN_N=3`, `V2_WAIT_CAP=3600`, `V2_WAIT_FILL=1`, `V2_WAIT_UNOBS=1`, `V2_HEADWAYS=stop_headways_v2.csv` (days 1–24), `V3_MODE=xgb`, `V3_MIN_RIDE_S=3600`. Rollback: `V2_WAIT=pooled` (config C) or `V2_WAIT=v1 V3_MIN_RIDE_S=0` (v1). `/health` shows the active flags.

**Data splits:** tables and model fit on 1–24 Oct (ML trains on 1–18 and is tuned on 19–24); **validation 19–24**; **test 25–31**, run once and never used for tuning. Live-layer tuning day 22 Oct; live test days 28 and 30 Oct.

**Running it:** see `README.md` (run the API) and `REPRODUCE.md` (rebuild every stage and number; on branch `docs/reproducibility`).

## 2. Benchmark (main = config F vs previous config C)

| Benchmark | C | **F (live)** |
|---|---|---|
| a. Ride time, 20k held-out rides (25–31 Oct) | 12.6% median error | 12.6% (F only changes waits) |
| b. 60 ticket pairs, ride vs actual GPS rides (38 with rides) | 6.8%, 76% within ±20% | **6.0%, 84%** |
| c. Door-to-door, validation 19–24 (common subset n=2,040) | 23.9%, 42.9% within ±20% | **18.8%, 52.5%** |
| f. Door-to-door, test week (run once, common n=1,997) | 24.1%, 42.5% | **18.8%, 52.9%** |
| d. Ride vs actual: ours vs Google Maps bus-only (38 pairs) | ours 7.2% vs Google 16.3% | **ours 6.4%** vs Google 16.3% |
| e. Live API latency (200 HTTP queries) | – | p50 320 ms, **p95 398 ms** |

Reference points: GTFS timetable 32.5% on the held-out rides; the **perfect typical-day historical model** (true average wait and ride for every stop and hour) would get **16.4%** door-to-door, so F (18.8%) is about 2.4 points from that limit.

### a. Ride time (`planner/evaluate.py`)
| | Median error | Within ±20% | Bias |
|---|---|---|---|
| Model | 12.6% | 68.4% | −10.5 s |
| GTFS schedule | 32.5% | 32.6% | +62.5 s |
| Model, rides > 1 h | 14.0% | 63.5% | −474 s |

By ride length: <10 min 13.3%, 10–30 min 12.0%, 30–60 min 11.8%, >60 min 14.0%. Noise floor (same route, stop pair and hour on different days): about 10.3%.

### b. 60 ticket pairs (`eval/run_planner.py`, `eval/compare.py`)
The pairs are chosen from 2.56 M tickets: short, medium and long trips across five regions, at 09:00, 13:00 and 18:30. The actual time is the median GPS ride on the ticketed route, from held-out weekdays 26–30 Oct.

| | C | F |
|---|---|---|
| Planner ride vs actual (n=38) | 6.8% / 76% / −0.1% | **6.0% / 84% / +0.1%** |
| GTFS schedule (n=37) | 20.7% / 49% / −12.4% | same |

### c. Door-to-door realized journeys, validation (`eval/realized_journeys.py`, `eval/rerun_compare.py`)
Each planned journey is replayed on that day's real buses: walk as planned, board the first real bus of the planned route, ride, transfer, walk.

| Config | d2d median error | Within ±20% | Median bias | First wait | Transfers | Under p85 |
|---|---|---|---|---|---|---|
| C | 23.9% | 42.9% | −868 s | −305 s | −451 s | 59.3% |
| **F** (waits from days 1–18: fair) | **18.8%** | **52.5%** | −322 s | +17 s | −179 s | 75.8% |

F scores 3,137 journeys on its own, against 2,495 for C.

### f. Test week (teammate's one-off run, after F was chosen; `docs/handoff/v2-waits.md`)
On each config's own scored journeys:

| | v1 | C | **F** |
|---|---|---|---|
| d2d median error | 28.3% | 25.7% | **19.6%** |
| Within ±20% | 37.3% | 39.8% | **51.0%** |
| First-wait bias | −512 s | −431 s | **−81 s** |
| Under p85 | 47.6% | 52.1% | **70.4%** |

Common subset (n=1,997): 25.9 / 24.1 / **18.8%**, the same as on validation.

### d. Google Maps, bus only (`eval/google_bus_bench.py`, `eval/google_bus_alloptions_bench.py`, 56 pairs)
Google times are bus only, departing Wed 2 Dec 2026 at the same clock time as each pair. Our time leaves out the first wait so it matches Google's departure-to-arrival time.

| | C | F |
|---|---|---|
| Median difference from Google | 18.2% | 19.6% |
| Median bias vs Google | +1.7% | +8.3% |
| 18:00 departures: difference / bias | 24.4% / +12.6% | 34.3% / +26.6% |
| Same first-choice route as Google | 18/56 | 21/56 |
| **Ride vs actual GPS (38 pairs): ours / Google** | 7.2% / 16.3% | **6.4% / 16.3%** |

All options (ours 3 per pair, Google 5), with F: our option 1 shares a route with one of Google's 73% of the time, options 2 and 3 about 33%. In 91% of pairs at least one route set is shared. Google's five options have identical durations in 13 of 56 pairs, so its alternatives look timetable-based.

### e. Latency
200 random queries over HTTP to the live API, idle machine: p50 320 ms, p95 398 ms, max 609 ms. Under heavy CPU load it rises to about 1 s.

### Live layer (`realtime/validate.py`, replay of 22, 28 and 30 Oct)
| | Half headway | GTFS schedule | Live ratio | Live Kalman |
|---|---|---|---|---|
| Next-bus wait, median abs error (28 Oct) | 503 s | 594 s | 107 s | **105 s** |
| Within ±2 min | 14% | 13% | 53% | 53% |

Door-to-door on 22 Oct, journeys where a live bus is visible (55%): 22.6% → **18.8%** with live waits.

## 3. Caveats and open issues
- "Actual" in (b) and (d) is the in-bus ride on the ticketed route, not a full door-to-door trip. Google times are for 2 Dec 2026; the real rides are from 26–30 Oct.
- Door-to-door scores only journeys that can be replayed from GPS (about 60%).
- Still optimistic: about 5 min short at the median, and about 75% of trips finish under p85 (target 85%).
- Alternatives (options 2–3) are weaker than option 1. The `fix/alternatives` branch (`ALT_V2`) filters bad ones but is not merged yet.
- About 17% of GPS trips carry the opposite direction's route id. Only one stop pattern per route is modelled, and walking is straight-line × 1.3.
- Full results: `docs/results/benchmark_final.md`, `docs/results/integration/`, `docs/handoff/`. Decisions and timeline: `docs/LOG.md`.
