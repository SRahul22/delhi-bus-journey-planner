# Delhi Bus Journey Planner

A Google-Maps-style API and map UI for Delhi buses (DTC and DIMTS). You give it two points and a departure time. It returns up to 3 bus journey options with estimated walk, wait and ride times.

## Run

```bash
cd planner
./run.sh            # macOS/Linux: http://localhost:8000  (PORT=8080 ./run.sh to change)
run.bat             # Windows
```

On first run, the script creates `.venv` from `requirements.txt` (fastapi, uvicorn, pandas, numpy, scipy, xgboost 2.1.4). Set `PYTHON` to choose the base interpreter. It also builds `cache/network.pkl` from GTFS, which takes about 7 s. If the GTFS changes, delete the cache or run `python preprocess.py` again.

Data locations (`paths.py`) are relative to the repo and can be overridden with env vars:
- `GTFS_DIR`: defaults to `../data/gtfs`. If that has no `stops.txt`, it falls back to `E:/hackathon-data/gtfs`.
- `MODEL_DIR`: defaults to `../model`, where the model agent writes `segment_times.csv`, `stop_headways.csv` and `holdout_trips.csv`. If those files are missing, the planner falls back to GTFS schedule times.

## API

| Endpoint | Description |
|---|---|
| `GET /journey?from=Kashmere Gate&to=Nehru Place&depart=2026-10-27T08:30&max_options=3` | Journey options. `from` / `to` are a stop name (best match from `/stops/search`) or `"lat,lng"`. The matched stop is echoed in `query.resolved`. `depart` is ISO; no offset means IST; default is now. |
| `GET /journey?from_lat=&from_lng=&to_lat=&to_lng=&depart=2025-11-25T09:00` | Same, with coordinates as separate params (these win over `from` / `to` when both are given). |
| `GET /stops/search?q=nehru` | Stop name search, with duplicate names merged |
| `GET /health` | Status, which travel-time source is active, and `ride_correction` (whether the XGBoost v3 correction is on) |
| `GET /docs` | Interactive OpenAPI docs (FastAPI) |
| `POST /admin/reload-model` | Re-read the model CSVs after the model agent updates them, without restarting |
| `GET /journey?...&realtime=1` | Uses live GPS conditions from `../realtime`, a replay of 28 Oct. With no `depart`, it departs at the replay clock's "now". Live data applies only within 90 min of that clock; otherwise the response falls back to historical times and `realtime.applied` is false. The first bus leg gets `live_next_buses_s` (next buses, in seconds from the clock), `live_catchable_bus_s` and `live_wait_s`. |
| `GET /realtime/status` | Replay clock, observation counts and live-ratio spread |
| `POST /realtime/clock?start=2026-10-28T18:00&speed=10` | Jumps or speeds up the replay clock for demos |
| `GET /realtime/eta?stop_id=&route_id=` | Live next-bus ETAs |
| `GET /` | Leaflet map UI. Click to set origin and destination, or search stop names. Pick a time, see the options and the legs drawn on the map. |

Each option includes `duration_s`, `duration_p85_s` (heavy-traffic estimate), `walk_m`, `wait_s`, `ride_s`, `transfers`, `labels` (recommended / fastest / fewest_transfers / least_walking / alternative), `legs` and `quiet_hour`. `quiet_hour` is the same journey (same routes and stops) re-timed with the historical model at the quietest of 06:00, 11:00, 15:00 and 22:00 on the same day: `depart`, `duration_s`, `ride_s`, `wait_s` and `traffic_delay_s` (time on the bus now minus at the quiet hour; negative means the requested time is the quieter one). It is null for a walk-only option. A walk leg has distance, duration and a polyline. A bus leg has route, headsign, board and alight stops, `num_stops`, `wait_s`, `ride_s`, `ride_p85_s`, depart and arrive times, `alt_routes` (other routes serving the same stop pair) and a polyline cut from the GTFS shape.

## Architecture

- `preprocess.py` turns GTFS into `cache/network.pkl`. For each route_id it keeps a canonical stop pattern (the most common trip pattern), scheduled segment times by hour, trips per hour, inter-stop distances along the shape, and the shape with each stop's position on it. It also stores footpaths between stops within 400 m (walking at 1.2 m/s with a 1.3 detour factor).
- `ttmodel.py` holds `TravelTimeModel`, which reads `../model/segment_times.csv` and `stop_headways.csv` when they exist. It precomputes dense `[segment, daytype, hour]` arrays. A segment's ride time comes from the first source that has data:
  1. the observed median for that route segment (n ≥ 3)
  2. the observed median for the same stop pair on any route (n-weighted)
  3. the GTFS scheduled time × an hour-of-day factor (fitted from observed/scheduled ratios when observed data exists, otherwise a prior profile)
  4. distance ÷ 15 km/h

  Segment statistic (`SEG_MODE`, default `blend`): each ride uses per-segment **medians for the first `BLEND_S` = 600 s, then per-segment means**. Segment times are right-skewed, so summing medians under-predicts long rides. Holdout (20k rides, 25–31 Oct):

  | mode | median abs error | within ±20% | median bias | rides > 1 h: error / within ±20% / bias |
  |---|---|---|---|---|
  | median | 12.8% | 67.8% | −19 s | 17.5% / 54.7% / −819 s |
  | mean | 13.7% | 64.9% | +8 s | 13.3% / 65.2% / −411 s |
  | blend 1800 s | 12.7% | 67.9% | −17 s | 15.4% / 58.7% / −609 s |
  | **blend 600 s** | **12.6%** | **68.4%** | −11 s | 14.0% / 63.5% / −474 s |
  | GTFS schedule × prior | 32.5% | 32.6% | +63 s | |

  If a cell has no observations, it falls back to the schedule × hour factor × that segment's own observed/scheduled ratio. Missing weekend factors are taken from the weekday factors.

  Wait = min(headway/2, 20 min), using the observed headway when available, otherwise GTFS trips per hour at the time the bus passes that stop. If a route has no service in that hour, the wait is 45 min.
- `router.py` holds `Planner`, a frequency-based RAPTOR. Each round adds one bus leg, with up to 4 legs (3 transfers). Access and egress use stops within 800 m, widening to 1.5 km and then 2.5 km if none are found. Candidates come from a time-optimal run, a walk-averse run (walking cost ×3), and up to 3 diversity runs that ban the route numbers already found. Each candidate is then re-timed exactly, segment by segment, as the hour changes. The planner drops silly journeys (re-boarding the same route, 1-stop hops, multi-bus options containing a ride under 4 min), removes duplicates, and labels and picks 3 options. Options are ranked by a generalized cost: duration + 10 min per transfer + 0.5 × wait + 0.5 × walk time. The best **direct** bus is always kept if it is within 20% of the fastest. Labels: recommended / direct / fastest / least_walking / fewest_transfers / alternative. Long sentence-style route names are shortened for `route`; the raw name stays in `route_short_name`. `duration_p85_s` = median + summed ride p85 excess + wait spread (wait p85 = 1.7 × mean wait, combined in quadrature). The UI shows a range (e.g. "59–83 min") for options with transfers or long waits.
- `app.py` is the FastAPI app (CORS enabled). At startup it also loads the optional real-time layer `../realtime/service.py` (adds about 3.5 s and 250 MB). It fails soft to historical times. Set `REALTIME=0` to skip it, and `RT_START` / `RT_SPEED` to set the replay clock. See `../realtime/INTEGRATION.md`. In the UI, the "Live traffic (replay 28 Oct)" checkbox switches to the replay clock and shows the live next-bus ETAs on the first bus leg. `static/index.html` is the UI.
- `evaluate.py`:
  - `evaluate.py` alone uses `../model/holdout_trips.csv` and compares the GPS model and the schedule baseline against actual ride times: median APE, % within ±20 %, bias, and a breakdown by hour.
  - `evaluate.py --pairs pairs.csv` (columns `from_lat,from_lng,to_lat,to_lng,depart,google_s`) compares the planner's fastest door-to-door time against Google Maps.

  Both write their output to `eval_out/`.

## Tests

- `python tests/sanity.py [ISO departs...]` plans 4 reference pairs (Kashmere Gate→CP, Rohini 22→Azadpur, Anand Vihar→Nehru Place, Dwarka→AIIMS) and prints the options and latency.
- `python tests/sanity.py --realtime 2026-10-28T18:00` runs the same pairs with live replay conditions at that clock time.

## Known limitations

- Each route_id uses one canonical stop pattern. Short-turn and variant trips are not modelled separately.
- Waits are frequency-based (headway/2, capped at 20 min), so the planner does not know about specific departures. The exception is `realtime=1`, which uses live ETAs.
- `duration_p85_s` adds up per-segment p85 times, so it is a pessimistic estimate.
- Walking is straight-line distance × 1.3. There is no street network.
