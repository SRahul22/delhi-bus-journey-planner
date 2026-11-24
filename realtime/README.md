# Real-time layer

Adds live traffic conditions to the historical travel-time model. It is a separate, pluggable module: the
planner works unchanged without it, and with it `/journey?...&realtime=1` uses recent GPS.

There is no live DTC feed for Oct 2026, so a **replay** of a held-out day (Wed 28 Oct 2026, not used to
train the historical model) stands in for it, behind a simulated clock.

## What it does

1. **Feed** (`feed.py`): `ReplayFeed` gives a simulated clock (any start time, speed multiplier). `poll()`
   returns the pings that arrived since the last poll; `latest()` returns the current position of each
   vehicle. To use a real feed, swap in a client that returns the same batch dict.
2. **Live state** (`live_state.py`): each vehicle trip is projected onto its route shape with the same
   rules as the historical pipeline (`model/scripts/process_day.py`: within 150 m of the shape, moving
   forward, at most 25 m/s, gaps of at most 300 s, terminal departure excludes layover). The data uses the
   UP/DOWN-paired route as a second hypothesis, because about 17% of GPS route_ids are in the wrong
   direction. Stop passings become segment observations, and each observation gets a ratio: observed time ÷
   the historical time the planner would use for that segment, hour and day type.
   - `live_ratio(route_id, from_stop, to_stop)` uses the last 30 min of observations. It pools them by
     level, and each level is shrunk toward the next one (the weights are observed minutes): **segment →
     3×3 km area around it → whole city → 1.0**. The result is clipped to [0.6, 2.5]. Every segment gets a
     ratio, so a segment with no recent bus still follows its area.
   - `next_bus_eta(stop_id, route_id)` covers live buses of that route still before the stop. The ETA is
     the remaining part of the current segment plus the historical times of the remaining segments, each
     multiplied by its live ratio.
3. **Planner glue**:
   - `rt_model.py`: `RealtimeTravelTimeModel` has the same interface as `planner/ttmodel.TravelTimeModel`.
     - `ride()` and the RAPTOR `cum()` table multiply each segment by `ratio ** w`. The weight `w` is 1 for
       the first 30 min after "now" and falls linearly to 0 at 75 min.
     - `wait_s()` uses the next live bus that reaches the stop after the passenger does. The wait is capped
       at the historical headway, since another bus may start from the terminal in the meantime. With no
       such bus, it falls back to the historical headway/2.
   - `service.py`: `RealtimeService` loads the replay and warms up on the 45 min before the start time
     (about 3.5 s). A background thread then updates every 15 s. It also builds a shallow copy of the
     `Planner` that uses the realtime model. The planner's own model object is never modified.
4. **Validation** (`validate.py`): see below.

## Run

```bash
# replay file (73 MB; made on Windows by scripts/extract_replay.py, then copied with scp)
ls realtime/data/replay_28.npz        # on Windows: E:\hackathon-work\realtime\replay_28.npz (picked up automatically)

# validation (about 90 s, uses the planner venv)
planner/.venv/bin/python realtime/validate.py      # -> realtime/results/validation.json

# in the app: apply the patch (see INTEGRATION.md), then
RT_START=2026-10-28T17:30 RT_SPEED=10 ./planner/run.sh
curl 'localhost:8000/realtime/status'
curl 'localhost:8000/journey?from_lat=28.6328&from_lng=77.2197&to_lat=28.5672&to_lng=77.2100&realtime=1'
curl -X POST 'localhost:8000/realtime/clock?start=2026-10-28T18:00&speed=1'
curl 'localhost:8000/realtime/eta?stop_id=367&route_id=3475'
```

If `depart` is omitted, a `realtime=1` request departs at the replay clock. Live ratios apply only when
`depart` is on the replay day, between 15 min before and 90 min after the clock. Other requests get the
historical model, and the response says `"applied": false`.

Making the replay file (Windows, one process, about 3.5 min, read-only on the source DB):
`python extract_replay.py 28` keeps one ping per vehicle every 30 s or more, 06:00–22:00 IST, columns ts,
vehicle, route, trip, lat/lng (×1e5 int). The result is 6.7 M rows and 73 MB as compressed npz, from 31.4 M
raw rows.

## Results (replay day 28 Oct 2026, historical model = v1.5 tables, 11 training days: 1–10 and 15 Oct)

The replay is evaluated every 30 min from 07:00 to 21:00. At each time T, the live snapshot uses GPS up to
T only. Rides that board after T are then scored against the actual stop-to-stop times in the day's GPS.
The rides are random board/alight pairs on observed trips, at least 3 stops and 2 min apart, about 130 k
per lead window.

| ride time (board → alight) | n | median abs % error | within ±20% | median bias |
|---|---|---|---|---|
| historical only, boarding 0–30 min after T | 129,445 | 13.0% | 68.2% | −8.3% |
| **historical + live ratios**, boarding 0–30 min after T | 129,445 | **12.3%** | **70.5%** | −4.7% |
| historical only, boarding 30–60 min after T | 128,163 | 13.1% | 67.7% | −8.3% |
| **historical + live ratios**, boarding 30–60 min after T | 128,163 | **12.5%** | **69.4%** | −6.4% |

For rides boarding 0–30 min after T, the live version has the smaller error on 56% of rides.

The gain is largest in the afternoon and evening peak. At 18:00 the error drops from 17.8% to 14.9%; from
13:00 to 18:30 it drops by 1.5–3 points. In the morning (07:30–08:30), live ratios make things slightly
worse (for example 8.5% → 10.2% at 08:00), so the gain is not uniform.

Ablations (same day; parameters were not tuned further, to avoid fitting the test day):

| variant | median abs % error (lead 0–30) |
|---|---|
| city-wide ratio only | 12.8% |
| segment → area → city (default: prior 3 / 8 / 30 min) | 12.3% |
| stronger shrinkage (10 / 20 / 30) | 12.2% |

**Next-bus wait** (9,165 random route/stop pairs that had a live bus approaching at T; actual wait = next
observed passing − T, median 13 min):

| predictor | median abs error | within ±2 min |
|---|---|---|
| historical headway / 2 | 512 s | 13.6% |
| **live ETA from vehicle positions** | **110 s** | **52.3%** |

The live ETA is the main real-time gain: much better first-leg waits.

## Limitations

- The replay stands in for a live feed. A production setup needs a feed client and should also handle
  feed latency and outages; on an exception the service keeps serving historical answers.
- The ground truth comes from the same projection pipeline, with 30 s pings instead of the 10 s pings used
  for the historical model. That interpolation difference could cause part of the historical model's −8%
  bias, and the live ratios partly remove that bias. The comparison of historical vs live is fair, since
  both are scored against the same truth.
- The historical model used here is v1.5 (11 training days). When the 1–24 Oct tables arrive, rerun
  `validate.py`; the live gain will probably shrink as the historical model gets better.
- Only the most common stop pattern per route is tracked, as in the historical model. Buses on branch
  variants or loops can be missed or mis-projected.
- `next_bus_eta` sees only buses already moving on the route (at least 200 m of progress, position less
  than 3 min old). A bus about to leave the terminal is invisible, which is why the wait is capped at the
  historical headway.
- The RAPTOR search uses live ratios but historical waits, and only the final re-timing of each option uses
  live waits. So the choice of options is only partly live-aware.
- Memory: about 250 MB for the replay arrays plus the state; CPU: about 0.3 s per 15 s update.

## Files

| file | what |
|---|---|
| `feed.py` | `ReplayFeed` (simulated clock, `poll`, `latest`, `window`) |
| `live_state.py` | `LiveState` (projection, passings, ratios, ETAs, snapshots) |
| `rt_model.py` | `RealtimeTravelTimeModel` wrapper for the planner |
| `service.py` | `RealtimeService` (background updater + realtime planner copy) for `app.py` |
| `validate.py` | offline validation → `results/validation.json` |
| `scripts/extract_replay.py` | builds the replay file on Windows |
| `planner_patch.diff`, `INTEGRATION.md` | the app.py change (not applied by the realtime agent) |
| `data/replay_28.npz` | replay file (not in git; on Windows in `E:\hackathon-work\realtime\`) |
