# Bus travel-time model (DTC GPS, Oct 2026)

Deterministic statistical model: historical stop-to-stop travel times per route, hour of day and day type,
plus observed headways. No ML.

## Outputs (in `model/`, small; copied from `E:\hackathon-work\out_train` and `out_hold`)

| file | columns | notes |
|---|---|---|
| `segment_times.csv` | route_id,from_stop,to_stop,hour,daytype,median_s,p85_s,n,mean_s | consecutive stops of the route pattern; hour = IST hour the bus passed `from_stop`; daytype weekday/weekend; time is passing-to-passing (includes dwell at from_stop) |
| `stop_headways.csv` | stop_id,route_id,hour,daytype,median_headway_s,n | gaps between consecutive (different) vehicles of the route passing the stop, 20 s – 3 h, within one day |
| `route_patterns.csv` | route_id,stop_sequence,stop_id,dist_m | the GTFS stop pattern used per route (most common stop sequence; dist along the most common shape) |
| `holdout_trips.csv` | route_id,board_stop,alight_stop,board_seq,alight_seq,depart_ts_ist,actual_s | 20k random observed rides (one random stop pair per GPS trip) from held-out days 25–31 Oct |
| `stats.json` | coverage stats of the last aggregation run | |

## Current version: trained on 1–24 Oct (24 days), holdout 25–31 Oct

- GPS trips (vehicle,trip_id) per day ~80–115k; ~68% yield stop passings. 35.4M stop passings, 33.4M segment
  observations (after filters) -> 1.80M rows in segment_times.csv (81 MB), 1.61M rows in stop_headways.csv (49 MB).
- Coverage: 72,002 of 79,592 GTFS pattern segments (90.5%) have data; 70,788 have n>=5; 2,203 of 2,574 routes.
- Sanity: weekday network speed by hour: ~32 km/h at 02–03h, 21 km/h at 08h, 18 km/h midday, 15 km/h at 18–19h.
  Sum of segment medians vs GTFS scheduled trip time (1,175 fully covered routes): ratio p10/p50/p90 = 0.78/1.02/1.38.
- Holdout (20k random rides, 25–31 Oct, `scripts/eval_holdout.py`, pure segment sums, no planner):
  - median_s: median APE 13.2%, MAPE 19.0%, 66.8% within 20%, bias -121 s (rides >1 h: bias -1148 s, medAPE 19.7%)
  - mean_s:   median APE 13.9%, MAPE 21.3%, 64.2% within 20%, bias -38 s (rides >1 h: bias -762 s, medAPE 14.1%)
  - Sums of medians under-predict long rides (segment times are right-skewed, so the sum of medians < median of
    sums). `mean_s` was added as an extra column for this; best is probably median for short hops and mean for
    rides > ~30 min. Remaining long-ride bias likely comes from stalls dropped by the 300 s ping-gap / 30 min
    filters (those segment observations are discarded, but are present in real rides).
- Runtime: build_patterns 6 s (Mac); process_day ~5 min/day alone, ~17–20 min/day with 10–20 in parallel (I/O
  bound reading 3–4.5 GB sqlite); all 31 days ~47 min wall; aggregate 24 days ~3 min; holdout ~20 s.

## Method

1. **Patterns** (`scripts/build_patterns.py`, runs on the Mac from GTFS, ~6 s): for each GTFS route_id take the most
   common stop sequence and the most common shape among its trips; project stops onto the shape monotonically
   (local equirectangular metres) -> `dist_m`. Also writes `work/route_shapes.csv` (shape vertices + cumulative
   distance) which the GPS processor needs. 285 of 82k stops are >300 m from their shape (bad GTFS geometry).
2. **Per-day GPS processing** (`scripts/process_day.py DD`, runs on Windows, ~5 min/day alone, ~12 min with 10 in
   parallel, ~2–3 GB RAM each). One sequential scan of `vehicle_feed` (read-only). Trips = (vehicle_id, trip_id);
   GPS `route_id` is the GTFS route_id. Per trip: sort, drop repeated GPS `timestamp` (stale pings are repeated
   with the same fix time), project every point onto the route shape (nearest segment), drop points >150 m off the
   shape, keep a plausible forward-moving chain (no >100 m backward jumps, no progress faster than 25 m/s + 100 m),
   take running max of distance-along, then linearly interpolate the time the bus first reached each stop's
   distance (only if the bracketing pings are <=300 s apart). For the first stop the *departure* (last time within
   50 m of it) is used, so terminal layover is excluded. Output `E:\hackathon-work\pass\pass_DD.npz`
   (route, trip, veh, seq, stop, t). Trips with <5 pings, <300 m progress or <2 stops are dropped.
3. **Aggregation** (`scripts/aggregate.py <train days> <holdout days|-> <outdir>`, Windows): segment observation
   = consecutive pattern stops (seq k -> k+1) of the same trip. Filters: duration >= 0, implied speed <= 70 km/h,
   duration <= max(30 min, dist/1 m/s). Group by (route, from, to, hour, daytype) -> median, p85, n.
   Headways from consecutive passings at a (route, stop) on the same day by different vehicles.
   Holdout: one random (board<alight) pair per trip of days 25–31, speed <= 70 km/h, first 20k.

## How to run (Windows, from Mac via the SSH master)

```
S="ssh -o ControlPath=~/.ssh/cm-%r@%h:%p -o BatchMode=yes -o LogLevel=ERROR <user>@<windows-host>"
O="-o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o BatchMode=yes -o LogLevel=ERROR"
python3 model/scripts/build_patterns.py data/gtfs model && mv model/route_shapes.csv model/work/
scp $O model/work/route_shapes.csv model/route_patterns.csv data/gtfs/routes.txt model/scripts/*.py <user>@<windows-host>:E:/hackathon-work/
# all days, 10 in parallel, detached (survives SSH drop; launch.py uses DETACHED_PROCESS + job breakaway):
$S "cd /d E:\hackathon-work && python launch.py E:\hackathon-work\run_all.log run_all.py 10 01,02,...,31"
# (skips days whose pass_DD.npz exists; per-day logs in E:\hackathon-work\log_DD.txt)
$S "cd /d E:\hackathon-work && python aggregate.py 01,02,...,24 - E:\hackathon-work\out_train"
$S "cd /d E:\hackathon-work && python aggregate.py \"\" 25,26,27,28,29,30,31 E:\hackathon-work\out_hold"
scp $O "<user>@<windows-host>:E:/hackathon-work/out_train/*" model/
```
Remote needs `pip install --user numpy pandas` (done). `scripts/diag.py` is a one-off diagnostic (which
route shape a GPS trip fits best), not part of the pipeline.

Large intermediates stay on Windows only: `E:\hackathon-work\pass\pass_DD.npz` (~40 MB/day),
`E:\hackathon-work\out_*`. Raw data `E:\hackathon-data` is never modified or copied.

## Caveats

- ~30% of GPS trips are unusable: too short, off-shape, or no forward progress. ~17% of sampled trips fit their
  own route's shape poorly; many of them run the opposite direction (UP vs DOWN route_id mislabelled) or a
  different route. `USE_PAIR=1` env var makes process_day also match against the paired UP/DOWN route
  (tested on a 3M-row sample: +5% matched trips at ~2x CPU) - not used in the production run.
- Only the single most common pattern per route_id is modelled; branch variants are ignored.
- Loops / out-and-back shapes that overlap themselves can be mis-projected (nearest segment wins).
- Segments missing for a given hour/daytype: the planner should fall back to other hours of the same
  segment, then to distance / typical speed (~17 km/h overall median).
- Hour cells often have small n (median per cell only a handful): use n as a confidence weight.
- Segment times include dwell at the from-stop; the first segment excludes terminal layover.
