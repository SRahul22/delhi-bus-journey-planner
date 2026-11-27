# Final benchmark: main with config F (live) vs config C

Main = `6bdd746` (config F default: `V2_WAIT=gaps`, `V2_WAIT_MIN_N=3`, `V2_WAIT_CAP=3600`, `V2_WAIT_FILL=1`, `V2_WAIT_UNOBS=1`, serving headways days 1–24, `V3_MIN_RIDE_S=3600`). C = previous live config (`V2_WAIT=pooled`). Outputs: `E:\hackathon-work\bench\final\` and `E:\hackathon-work\bench\`. Run 2026-11-27.

## Headline
| Benchmark | C (previous) | **F (live now)** |
|---|---|---|
| a. Ride time, 20k held-out rides (median error) | 12.6% | 12.6% (rides unchanged by F) |
| b. 60 ticket pairs, ride vs actual (38 with rides) | 6.8% / 76% within ±20% | **6.0% / 84%** |
| c. Door-to-door, validation 19–24 (common subset n=2,040, waits from days 1–18) | 23.9% / 42.9% | **18.8% / 52.5%** |
| f. Door-to-door, test week 25–31 (run once by the teammate after choosing F; common n=1,997) | 24.1% / 42.5% | **18.8% / 52.9%** |
| d. Ride vs actual, ours vs Google bus-only (38 pairs) | ours 7.2%, Google 16.3% | **ours 6.4%**, Google 16.3% |
| e. Live API latency, 200 queries over HTTP | – | p50 320 ms, **p95 398 ms** |

## a. Ride time (`planner/evaluate.py`, 20k held-out rides from 25–31 Oct)
Model 12.6% median error, 68.4% within ±20%, bias −10.5 s; GTFS schedule 32.5% / 32.6%. Rides over 1 h: 14.0% / 63.5%, bias −474 s. Identical for C and F: F only changes waits.

## b. 60 ticket pairs vs actual rides (`eval/run_planner.py` + `eval/compare.py`)
| | C | F |
|---|---|---|
| Planner ride vs actual (n=38) | 6.8%, 76% within ±20%, bias −0.1% | **6.0%, 84%, bias +0.1%** |
| GTFS schedule (n=37) | 20.7%, 49%, bias −12.4% | same |
| Planner within the actual IQR | 68% | 71% |

## c. Door-to-door realized journeys, validation 19–24 (`eval/realized_journeys.py`, `eval/rerun_compare.py`)
Common subset scored by every config (n=2,040):

| config | d2d median error | within ±20% | median bias total | first wait | transfer | under p85 |
|---|---|---|---|---|---|---|
| C | 23.9% | 42.9% | −868 s | −305 s | −451 s | 59.3% |
| **F (main, waits from days 1–18: fair)** | **18.8%** | **52.5%** | −322 s | +17 s | −179 s | 75.8% |
| F (main, serving waits 1–24: partly in-sample) | 18.5% | 53.0% | −282 s | +20 s | −113 s | 77.5% |

F on main reproduces the teammate's F run exactly (same 3,137 scored journeys and numbers). F also scores more journeys on its own (3,137 vs 2,495).

## d. Google Maps, bus only (`google_bus_bench.py`, `google_bus_alloptions_bench.py`; 56 pairs)
Our time without the first wait vs Google's time from its first departure:

| | C | F |
|---|---|---|
| Median difference from Google | 18.2% | 19.6% |
| Within ±20% of Google | 59% | 52% |
| Median bias vs Google | +1.7% | +8.3% |
| 18:00 departures: difference / bias | 24.4% / +12.6% | 34.3% / +26.6% |
| Same first-choice route as Google | 18/56 | 21/56 |

Against actual bus rides on the 38 pairs (ride time on the ticketed route): ours C 7.2%, **ours F 6.4% (84% within ±20%)**, Google bus-only 16.3% (61%, bias −5.2%). F disagrees with Google more in the evening, but it is closer to the real buses: Google is optimistic for Delhi buses, most of all at the evening peak.

All options (ours 3 per pair, Google 5):

| our rank | C: same route as a Google option / time diff | F: same route / time diff |
|---|---|---|
| 1 | 73% / 16.8% | 73% / 15.1% |
| 2 | 48% / 15.2% | 32% / 20.7% |
| 3 | 32% / 22.3% | 33% / 21.5% |

At least one shared route set: C 95%, F 91% of pairs. Google's #1 among ours: C 61%, F 59%. Google's own five options have identical durations in 13/56 pairs, so its alternatives look timetable-based.

## e. Latency
200 random queries over HTTP to the live API on port 8000, after the switch to F: p50 320 ms, p95 398 ms, max 609 ms.

## f. Test week (teammate's run, once, after choosing F; `docs/handoff/v2-waits.md`)
5,300 OD-times. On each config's own scored journeys: d2d median error v1 28.3%, C 25.7%, **F 19.6%**; within ±20% 37.3 / 39.8 / **51.0%**; first-wait bias −512 / −431 / **−81 s**; under p85 47.6 / 52.1 / **70.4%**. Common subset (n=1,997): 25.9 / 24.1 / **18.8%**. These match validation, so F was not tuned to the validation week.

## Caveats
- "Actual" in (b) and (d) is the in-bus ride on the ticketed route from GPS, not a full door-to-door trip. Google times are for Wed 2 Dec 2026; the real rides are from 26–30 Oct.
- Door-to-door (c, f) scores only journeys whose buses can be replayed from GPS (about 60%).
- Still optimistic: plans are about 5 min short at the median, and only about 75% of trips finish under p85 (target 85%).
- Not in main yet: `fix/alternatives` (ALT_V2) and `docs/reproducibility`. Both branched from main before the F merge; expect conflicts in `planner/router.py`, `planner/ttmodel.py`, `docs/LOG.md` and `README.md`.
