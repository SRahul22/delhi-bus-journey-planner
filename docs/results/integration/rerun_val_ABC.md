# Door-to-door rerun on validation (days 19-24): A / B / C

Integration agent, 2026-11-24 ~20:30-21:20 IST (Windows clock). Branch v2/integration (flag commit 6c9441f).
Only `--set val` (19-24 Oct) was run; nothing from 25-31.

## Configs
| config | V3_MODE | V3_MIN_RIDE_S | V2_WAIT | wait data | output |
|---|---|---|---|---|---|
| A | xgb | 0 | v1 | v1 | `E:\hackathon-work\v2\rerun\A2\realized_val.csv` (fresh run); reference `E:\hackathon-work\v2\eval\realized_val.csv` |
| B | xgb | 3600 | v1 | v1 | `E:\hackathon-work\v2\rerun\B\realized_val.csv` |
| C | xgb | 3600 | pooled | POOLED_INDEX=pooled_index_visits_**1-18**.npz, V2_HEADWAYS=stop_headways_v2_**1_18**.csv (no val-day leakage) | `...\rerun\C\realized_val.csv` |
| C24 | as C | 3600 | pooled | serving files: pooled_index_visits_**1-24**.npz, stop_headways_v2.csv (includes val days 19-24: leaky, reference only) | `...\rerun\C24\realized_val.csv` |

The same 5,240 OD-times as the existing realized_val.csv were used (60 pairs x val weekdays + 5,000 random, seed 2026). Each config planned from scratch (`V2_EVAL_OUT=<dir>`, so `plans_val.pkl` is per config and never reused). Comparison: `eval/rerun_compare.py` (uses eval/metrics.py).

**A reproduces the existing file:** the fresh A run has an identical coverage summary (scored share 0.56927, 54/60 pairs, same unscored-reason counts). The full metric diff against the original was cut off by the SSH drop (see "Not done"). The tables below use the original realized_val.csv as A. Its plans were rebuilt from csv (`--plans-from-csv`) and carry no per-leg ride, so the ride-leg rows for A cover direct journeys only.

## Each config on its own scored journeys
| config | n | d2d median APE | within ±20% | median bias total (s) | first wait (s) | ride (s) | transfer (s) | direct APE (n) | transfer APE (n) | under p50 | under p85 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 2983 | 26.6% | 38.4% | -1260 | -462 | -6 | -598 | 24.5% (1131) | 27.9% (1852) | 18.7% | 50.9% |
| B | 3011 | 26.8% | 38.4% | -1267 | -444 | -40 | -614 | 23.9% (1111) | 28.5% (1900) | 17.5% | 50.0% |
| C | 2495 | 25.2% | 40.9% | -1030 | -342 | -42 | -529 | 23.8% (1195) | 27.1% (1300) | 24.9% | 56.8% |
| C24 | 2473 | 24.5% | 41.4% | -961 | -357 | -33 | -447 | 23.6% (1221) | 25.8% (1252) | 25.2% | 57.9% |

## Common subset: OD-times scored in all configs (n = 1,996), the fair comparison
| config | n | d2d median APE | within ±20% | median bias total (s) | first wait (s) | ride (s) | transfer (s) | direct APE (n) | transfer APE (n) | under p50 | under p85 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1996 | 26.3% | 39.8% | -1064 | -439 | -8 | -566 | 24.3% (947) | 28.1% (1049) | 20.3% | 52.5% |
| B | 1996 | 25.8% | 40.5% | -1030 | -400 | -38 | -574 | 22.5% (926) | 28.2% (1070) | 19.5% | 52.7% |
| C | 1996 | 24.4% | 42.0% | -871 | -313 | -28 | -454 | 23.8% (1058) | 25.8% (938) | 26.4% | 58.8% |
| C24 | 1996 | 24.4% | 41.4% | -874 | -323 | -25 | -433 | 23.8% (1065) | 25.7% (931) | 26.6% | 58.6% |

Direct/transfer counts differ by config within the common subset because each config plans its own itinerary.
Versus A (all 5,240 OD-times), the itinerary changed for 663 (B), 3,191 (C) and 3,320 (C24). The duration alone changed for 4,141 (B), 1,877 (C) and 1,783 (C24).
Under C, the scored share falls from 57% to 48%: pooled waits favour routes with fewer GPS-tracked buses, so fewer journeys can be replayed.

## Ride-only leg error (plan leg ride_s vs realized ride), by realized leg length
Identical legs (same route, board and alight stop, same OD-time) scored in all configs: 592 legs, all direct journeys (because of A's rebuilt plans).

| config | bucket | legs | median APE | within ±20% | median bias (s) | median bias % |
|---|---|---|---|---|---|---|
| A | all | 592 | 13.0% | 67.6% | +16 | +0.6% |
| A | ≤60 min | 452 | 12.7% | 69.5% | +51 | +3.7% |
| A | >60 min | 140 | 14.4% | 61.4% | -510 | -9.9% |
| B | all | 592 | 12.6% | 68.2% | +3 | +0.3% |
| B | ≤60 min | 452 | 12.1% | 70.1% | +30 | +1.9% |
| B | >60 min | 140 | 14.2% | 62.1% | -444 | -9.7% |

On their own scored legs, B gives 12.1% for ≤60 min rides (4,690 legs) and 18.0% for >60 min rides (588 legs, median bias -743 s). A's own legs (direct journeys only) give 12.2% and 16.2% (244 legs). These are not the same legs, so this comparison is not like-for-like. Full output: `E:\hackathon-work\v2\rerun\compare_ABC.md`.

## Verdict
1. **B (XGBoost only on rides over 60 min) helps slightly:** on the same legs, ride error falls from 13.0% to 12.6%, mostly on short rides (12.7% to 12.1%, bias +51 s to +30 s). Door to door on the common subset it falls from 26.3% to 25.8%. Long rides remain about 10% too short in both.
2. **C (B + pooled visit waits, 1-18 index) helps more:** door-to-door error falls from 25.8% to 24.4%, within ±20% rises from 40.5% to 42.0%, the total bias shrinks from -1,030 s to -871 s, and transfer-journey error falls from 28.2% to 25.8%. However, planned journeys are still about 15 min too short, the band is far from calibrated (58.8% under p85; G2 asks 82-88%), and C scores 9 points fewer journeys.
3. The serving index (1-24, C24) performs the same as the fair 1-18 index on the common subset. Recommend B as the new default now. C is a clear improvement, but the d2d calibration still needs work before G2 can pass.

## Flag
`V3_MIN_RIDE_S` (default **0** = current behaviour, unchanged). The v3 correction is applied to a bus leg only when the planner's own ride estimate (ttmodel blend) is ≥ V3_MIN_RIDE_S; legs below it keep the blend. With V2_BAND=1 the band call is gated the same way. `/health` flags report it. Tests: `planner/tests/test_v3_min_ride.py`. **The default was not changed: humans flip it to 3600 after reading this.**

## Not done
- The SSH control connection to Windows dropped at about 21:20, after all runs finished. The full metric diff of fresh A (`rerun\A2`) against the original realized_val.csv was not printed; only the coverage summaries were checked, and they match exactly. The first A run (`rerun\A`, 5 workers, slow because XGBoost runs on every leg) may still be writing; A2 (15 workers) is the one to use.
