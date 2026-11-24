# v2 integration (branch v2/integration)

Integration agent, 2026-11-24, about 20:10-20:30 IST (Windows clock). Worktree: `E:\hackathon-wt\integration`, branch `v2/integration` from main (2c0bc3c). Nothing was merged into main and the production API was not touched.

**Important:** the Windows machine **rebooted at about 20:09** (uptime was 11 min at 20:20). The SSH drop was the reboot. Consequences:
- Files that were being written then are NUL-filled: the three data handoff notes in the worktree (restored from the Mac copies before committing), `validate_pooled_visits1-18.log` and `pooled_validation_visits.csv` (rerun, see below).
- `realized_test.csv` did **not** finish: `E:\hackathon-work\v2\eval\realized_test_log.txt` stops at "planned 2000" of 5,300 and has no DONE. It was not relaunched; that is for the eval role at a gate.
- **The production API on port 8000 is down** (`curl /health` fails). Per instructions it was not restarted; restart it with `planner\run.bat`.

## 1. Role branches finished
| Branch | Commit | What was added |
|---|---|---|
| v2/data | 9bbc5e8 | handoff T0.1, T0.4, T2.1 (Mac copies; worktree copies were NUL bytes); 3 LOG lines. `E:\hackathon-work\v2\data\README.txt` STATUS: stop_headways_v2.csv (1,682,554 rows, 576k with n>=20) and stop_headways_v2_1_18.csv (1,623,383 rows) marked FINAL after loading both. |
| v2/planner | 88b3e06, 8818f6b | validate_pooled.py out.csv argument; handoff T4.5, T2.4, T2.2, latency-check; 5 LOG lines; T2.2 visits-index validation rerun result. |
| v2/eval | 93bf79f | No change. realized_test_log.txt does not end with DONE (see above). |

T2.2 visits-index validation (rerun, val 19-24, 400 ODs, seed 2026; any route in the attractive set, n=316): v1 median bias -165 s, MAE 626 s; pooled median bias **-9 s**, mean bias -259 s, MAE 571 s. The pass-based index gave -120 s / 642 s. Details in docs/handoff/T2.2.md.

## 2. Merges (in order)
| Branch | Merge commit | Conflicts and resolution |
|---|---|---|
| fix/criteria-gaps | 38067d1 | None. |
| v2/data | 2477ca6 | docs/LOG.md: all entries kept, new entries sorted by timestamp (script `E:\hackathon-work\v2\integration\merge.py`). |
| v2/eval | 96e0491 | LOG.md as above. docs/handoff/T0.4.md add/add (data part and eval part): both kept, concatenated. |
| v2/ml | a5df1b0 | LOG.md only. |
| v2/realtime | 67f7559 | LOG.md only. |
| v2/planner | cf908b0 (+ d70bdf5, b4b2310 for the T2.2 note) | LOG.md; planner/app.py and planner/router.py, see below. |

- **planner/app.py:** kept `import router` (fix branch, used for `ride_correction`) and added `import router as _router` and `v2ext` (planner). `/health` returns ride_correction **and** models_loaded, admin_enabled and flags. `/journey` takes from=/to= names or lat/lng (fix) **and** arrive_by (planner). It calls `plan(..., quiet_planner=planner, arrive_by=arr)` and still echoes `query.resolved`. The admin token and preload code came over without conflict.
- **planner/router.py:** `plan(..., compare_quiet=True, quiet_planner=None, arrive_by=None)`. On-time ranking (`_rank_on_time`) runs first, then quiet-hour re-timing on the final selected options, so an option added as most_likely_on_time also gets quiet_hour. Then `_skel` is removed.
- **planner/ttmodel.py:** no textual conflict; only v2/data edits it. `V2_WAIT=gaps` loads stop_headways_v2 waits into ttmodel. `V2_WAIT=pooled` loads those **plus** the planner's pooled overlay in time_journey: the data agent made `pooled` load the gaps file on purpose. Both modes run, and v1 (unset) is unchanged. Note: under integration, pooled mode's base and fallback wait (and the RAPTOR search wait) is the gaps wait, not v1. The planner validated pooled against a v1 base, so the combined mode has not been validated on its own.
- **model/v3/predict.py:** only v2/ml edits it (band behind V2_BAND); the planner preload calls it. No conflict, and both work: with V2_BAND=1, models_loaded.v3.band = "quantile+conformal (ml)".
- The history also contains 8cf0ab1, an earlier version of the T2.2 note commit (superseded by 8818f6b; same content apart from a timestamp).

## 3. Tests (all v2 flags at defaults; planner venv; MODEL_DIR=E:\hackathon-project\model)
- pytest: planner/tests, model/v3/v2/tests, realtime/tests, model/scripts/test_headways_v2.py: **35 passed** (planner 16, ml band+journey_band 10, realtime KF 6, data headways 3).
- eval: `python -m unittest test_metrics`: **11 OK** (also 11 passed under pytest).
- **Defaults vs main+fix:** Planner.plan for 5 OD/time pairs, run in a temporary worktree of fix/criteria-gaps (b3a7d32) and on v2/integration. The options (summary, routes, duration, wait, ride, quiet_hour duration) are **identical**; `fc` found no differences. Script: `E:\hackathon-work\v2\integration\cmp.py`. Outputs: `cmp_fix.json`, `cmp_int.json`.

## 4. Validation report (--set val only)
`python eval/report.py --set val --out docs/results/integration` gives `docs/results/integration/v2_2026-11-24_val.md` (22 s; G0 limit is 10 min). It matches the eval branch's report. noise_floor_val.json was regenerated byte-identical.
- Ride error, 300k val rides (days 19-24, 1-18 tables): GTFS 27.53%, **sum of medians 13.05%**, blend 13.21%, xgb_l1_1-18 13.46%, v2 ml band q50 13.80%; noise floor 10.28%.
- v1 band coverage 86.0% (83.2 / 87.9 / 90.9 / 94.2% by predicted bucket); v2 ml band 85.6% (85.1 / 85.4 / 87.2 / 89.1%), width 25.1% vs 30.0%.
- Door-to-door (realized_val, 2,983 scored): median error 26.6%, within +-20% 38.4%, median bias -1,260 s; first-wait bias -462 s (mean real wait 1,410 s); ride bias -6 s; transfer bias -598 s on transfer journeys; v1 d2d band under p50/p85/p90 18.7 / 50.9 / 57.7%.

## 5. Latency (Planner.plan in-process, as /journey calls it incl. quiet-hour; 200 random queries, seed 2026, stops 2-20 km apart, val days 06-22 h)
| Mode | p50 | p95 | max | CPU busy (20 cores) |
|---|---|---|---|---|
| flags off | 312 ms | **403 ms** | 537 ms | 0% before, 5% mean during |
| V2_WAIT=pooled, V2_BAND=1 | 310 ms | **403 ms** | 581 ms | 0% before, 5% mean during |

V2_LIVE=kf was not included: the live layer is not part of `Planner.plan` (it goes through RT.planner_for). On this idle machine (just rebooted), both modes are **under 500 ms**. This is HTTP-free, so /journey adds a few ms. The role agents' p95 of 1,055-1,198 ms (finding (e), "over 500 ms with and without flags") was measured at 76% CPU load. Latency depends mostly on load. Files: `E:\hackathon-work\v2\integration\latency_flags_{off,on}.json`, script `bench_int.py`.

## 6. Key findings and recommendations (from the role results)
a. **Ride model default:** on validation, the plain sum of segment medians (13.05%) beats the blend (13.21%) and XGBoost trained on 1-18 (13.46%; its best is 10 trees). The served `V3_MODE=xgb` default should be reconsidered. The holiday day type (2 and 20 Oct pooled with weekends) improves medians from 13.05% to 12.42% (XGBoost 12.75%).
b. **Door to door:** planned journeys are about 21 min too short (median bias -1,260 s). Almost all of it is waits and transfers: mean real first wait 1,410 s vs 428 s planned, transfer bias -598 s, while ride bias is only -6 s. Waits are the main lever: gaps or pooled-visits waits, then a d2d calibration. Don't serve shifts_d2d.json until that is done.
c. **Ride band:** the conformal band covers 85-89% by predicted bucket, but only 67% for rides whose actual length is over 60 min (89.6 / 84.2 / 79.2 / 66.9% by actual bucket). Humans need to choose which bucketing G1 uses.
d. **Report fix:** the GTFS schedule baseline on the 600k test rides is **27.9%**, not 32.5% (eval measured it on the existing rides; this was not rerun here).
e. **Latency:** the roles found over 500 ms with and without flags at 76% load; on an idle machine integration measures p95 403 ms in both modes (section 5).
f. Pooled waits from the stop-visits index have about zero median bias (-9 s) on val. Consider switching `POOLED_INDEX` to `pooled_index_visits_1-24.npz`.

## 7. Gate checklist (validation evidence, all **for human review**, none marked passed)
- [ ] **G0** (report under 10 min): report.py --set val ran in 22 s on v2/integration. *For human review.*
- [ ] **G1** (band coverage 82-88% in every bucket): v2 ml band 85.1 / 85.4 / 87.2 / 89.1% by predicted bucket (the >60 bucket is above 88); by actual length 89.6 / 84.2 / 79.2 / 66.9%. The 16-hour replay-day criterion was not checked here. *For human review.*
- [ ] **G2** (wait bias +-60 s; d2d under p50/p85/p90 47-53 / 82-88 / 87-93%): realized val mean wait bias -982 s (v1 plan). Pooled-visits validation median bias -9 s, mean -259 s. v1 d2d band 18.7 / 50.9 / 57.7%; v2 d2d band not evaluated on realized journeys. *For human review.*
- [ ] **G3** (next-bus error below v1 live on 28 and 30 Oct; no ride regression 07:30-08:30): realtime role reports KF 104.9 vs 107.4 s (28 Oct) and 105.5 vs 105.6 s (30 Oct). Rides 07:30-08:30 on 28 Oct: KF 9.76% vs historical-only 8.78%. KF parameters at the grid edge. Not rerun here. *For human review.*
- [ ] **G4** (ride error within +0.1 pt of v1; nightly under 15 min; p95 under 500 ms with all flags): p95 403 ms all flags on (idle machine). Ride error: defaults reproduce v1 exactly; V2_BAND changes only the band, not the point times. Nightly job not timed. *For human review.*

## 8. Left undone
- realized_test.csv (lost in the reboot) not rerun; the production API on port 8000 is down and was not restarted.
- The combined V2_WAIT=pooled mode (gaps base + pooled overlay) has not been validated end to end on realized journeys.
- Pending human decisions (from roles): BLEND_S (1200 by the card's rule, at the grid edge; 600 on days 21-24; plain median best), holiday and Saturday day types, ADMIN_TOKEN-unset behaviour (403 disables /realtime/clock; nightly.ps1 needs the header), V3_MODE default.
