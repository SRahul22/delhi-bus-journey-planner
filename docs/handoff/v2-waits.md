# v2/waits: honest bus waits (handoff)

Branch `v2/waits-main` = `main` 96cdce7 (config C) + `v2/waits`. Worktree `E:\hackathon-wt\waits-main`. Plan and diagnosis: `docs/handoff/v2-waits-plan.md`. Outputs: `E:\hackathon-work\v2\waits\`.

## What changed
`planner/ttmodel.py`, used only when `V2_WAIT=gaps` (or `pooled`, as its base wait):

| Flag | Default | Recommended | Effect |
|---|---|---|---|
| `V2_WAIT_MIN_N` | 20 | 3 | use the stop-visit gap wait mean(H²)/(2·mean H) from 3 gaps per cell, not 20 |
| `V2_WAIT_CAP` | 1200 | 3600 | cap on gap waits, in s |
| `V2_WAIT_FILL` | 0 | 1 | hours where a stop-route has gap data at other hours but no bus within 3 h get the cap |
| `V2_WAIT_UNOBS` | 0 | 1 | stop-routes with no GPS gap data: timetable wait × k, where k = median(gap wait / v1 wait) over cells with n ≥ 20, learned from the gap table (2.4–2.5); capped |

Recommended serving config ("F"):
`V2_WAIT=gaps V2_WAIT_MIN_N=3 V2_WAIT_CAP=3600 V2_WAIT_FILL=1 V2_WAIT_UNOBS=1`. It uses the 1–24 gap table, `stop_headways_v2.csv`, found automatically. This **replaces pooled** waits.

Main's other config C setting stays (`V3_MIN_RIDE_S=3600`). With all defaults, behaviour is main's config C, unchanged.

Tests: `planner/tests/test_v2_waits.py` (5). All tests on this branch: 44 pytest passed, 1 skipped; eval 11 unittest OK.

## Validation (19–24 Oct; gap table and pooled index from 1–18 Oct; 5,240 OD-times)

| config | scored / total | d2d median error | within ±20% | median bias | first-wait bias (median) | transfer bias (median, transfer trips) | ride leg sum bias (median) | under p50 | under p85 | under p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 2983 / 5240 | 26.6% | 38.4% | -1260 s | -462 s | -598 s | -6 s | 18.7% | 50.9% | 57.7% |
| main_C | 2495 / 5240 | 25.2% | 40.9% | -1030 s | -342 s | -529 s | -42 s | 24.9% | 56.8% | 63.3% |
| fix_D | 2445 / 5240 | 19.6% | 50.9% | -394 s | -46 s | -154 s | +8 s | 37.3% | 72.9% | 78.0% |
| chosen_F | 3137 / 5240 | 19.1% | 51.7% | -424 s | -12 s | -155 s | -24 s | 37.0% | 74.1% | 79.2% |

Notes:
- `main_C` is the integration rerun C (pooled, visits index 1–18, `V3_MIN_RIDE_S=3600`).
- `fix_D` = gaps + MIN_N 3 + CAP 3600 + FILL, on the v2/integration code (`V3_MIN_RIDE_S=0`).
- `chosen_F` = fix_D + UNOBS, on this branch (`V3_MIN_RIDE_S=3600`). So F vs D also differs in the ride-correction setting. Ride leg error: v1 12.8%, C 12.3%, D 12.4%, F 12.2% (no regression).
- **UNOBS fixes a coverage problem.** Without it (D), the planner drifted to routes with no GPS data, which can't be checked: only 47% of OD-times were scored. F scores 60%, more than v1's 57%.
- On the 1,689 OD-times scored in all four configs, median error was v1 25.2%, C 23.7%, D 19.5%, F 19.0%. Full tables: `E:\hackathon-work\v2\waits\compare_val_ACDF.md`.
- **Not measured:** the fix combined with pooled waits (run G was stopped to save time). Pooled waits keep their own 20-min cap and the ≥20-gap rule. Recommendation: use gaps (F) instead of pooled.

## Test week (25–31 Oct): run once, after choosing F on validation

Test week, 5,300 OD-times. Each config is scored on its own scored journeys. Arrows show which way is better.

| metric | v1 | main_C | **v2 (F)** |
|---|---|---|---|
| scored (↑) | 3034 | 2516 | **3170** |
| d2d median error (↓) | 28.3% | 25.7% | **19.6%** |
| within ±20% (↑) | 37.3% | 39.8% | **51.0%** |
| median bias, total (→0) | −1404 s | −1240 s | **−626 s** |
| first-wait bias (→0) | −512 s | −431 s | **−81 s** |
| transfer bias (→0) | −644 s | −503 s | **−170 s** |
| ride leg error (↓) | 11.9% | 11.8% | **11.4%** |
| under p50 (target 50%) | 14.0% | 20.2% | 32.8% |
| under p85 (target 85%) | 47.6% | 52.1% | 70.4% |
| under p90 (target 90%) | 53.9% | 59.5% | 77.1% |

Common subset (n=1,997, scored in all three): d2d median error v1 25.9%, C 24.1%, F 18.8%; within ±20% 40.7 / 42.5 / 52.9%.
Full tables: `E:\hackathon-work\v2\waits\compare_test_ACF.md`. Test results match validation (19.1% there), so F was not tuned to the validation week.

## Caveats
- GPS gaps are about 1.27× the GTFS timetable gaps. Either buses are missing from GPS or service runs below the timetable. The planner's waits and the realized truth come from the same GPS, so they agree with each other. Absolute waits may be somewhat overstated.
- The d2d band is still too narrow: under p85 is 74% on validation (target 85%). The next step would be a door-to-door p85 calibration (T2.6) on top of F.
- Run G (fix + pooled) and an earlier test run of D were stopped; their folders hold partial logs only. `test_A` (v1 on the test week) ran before the choice; its numbers were not looked at until the final comparison.
