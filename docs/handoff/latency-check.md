# Latency check: /journey p95 with all planner flags on (G4 input)

Status: measured. **The 500 ms target is not met on this machine right now, with or without the v2 flags.** Gate decisions are for humans.

## Setup
- `planner\bench_latency.py`: 200 random queries (seed 7; stops 2-20 km apart; departures 21 Oct 06:00-22:00; `arrive_by` = depart + 75 min, so the T2.4 path runs every time), in-process through FastAPI's TestClient (no port; port 8000 not touched). REALTIME=0, V3_MODE=xgb.
- `planner\bench_all.py` runs it twice in sequence: flags off (v1), and all planner flags on (V2_WAIT=pooled, V2_BAND=1 with ml's predict.band from branch v2/ml b6c38ca staged in E:\hackathon-work\v2\planner\mlpredict, ADMIN_TOKEN set).
- The machine was shared with four other agents: system CPU was 76% busy (GetSystemTimes over 3 s) during the runs.

## Results (E:\hackathon-work\v2\planner\latency_*.json)
| run | p50 ms | p95 ms | max ms | errors |
|---|---|---|---|---|
| v1, flags off | 761 | 1,198 | 1,847 | 0 |
| all planner flags on | 747 | 1,055 | 1,421 | 0 |

Smoke runs of 20 queries before the speed-ups below: flags off p95 979 ms, flags on p95 1,240 ms.
The v2 features add no measurable latency after the fixes; the remaining cost is v1's search.
For reference, docs/LOG.md 17:50 measured one v1 query at 406 ms on the idle machine.

## Speed-ups made (same outputs)
- XGBoost single-row predict with `nthread=1` (set at preload): 9.0 ms -> 0.32 ms per call (micro-benchmark, 200 calls). With V2_BAND each leg makes two calls and each query times ~7 candidate journeys.
- RAPTOR: per-query hour caches for waits/cumulative ride tables instead of millions of tuple-key dict lookups.
- Regression check: flags off, the current router and the original v1 router (commit 2c0bc3c) gave identical options (summary, duration, p85, labels) on 60/60 random queries.

## Where the time goes (cProfile, 15 queries, all flags on)
RAPTOR is ~80% (5 passes per query: fastest, walk x3, and 3 passes with banned routes for diversity); time_journey ~18% (of which XGBoost before the nthread fix was most of it); pooled waits ~3% (cache misses on the first call of each stop/set/hour).

## Open issues / options (not done; each changes v1 behaviour, so it needs a flag and a human decision)
- Run the diversity passes (3 x banned RAPTOR) only when fewer than max_options distinct options were found, or cap them at max_rounds=2.
- Rerun the benchmark on an idle machine before judging G4.
