# AGENTS.md: Delhi Bus Journey Planner, v2 workload

Operating manual for the AI coding agents on this repo. Every agent reads sections 1–7 **and section 10 (amendments, which override)**, then its own task cards in section 8. If this file disagrees with the code, trust the code, log the mismatch in docs/LOG.md, and fix this file in the same change.

Human-facing plan: "Implementation Plan for the Delhi Bus Journey Planner" (team doc).

## 1. What the system does

Travel-time API for Delhi DTC buses, built from 31 days of bus GPS (October 2026).

GPS pings → map-matched stop passings → segment time tables (median, mean, p85 by hour and day type) → XGBoost correction per bus leg → frequency-based RAPTOR planner → FastAPI + Leaflet UI. A live layer replays a held-out day (28 Oct) to adjust segment times and next-bus ETAs.

v2 goal: measure door-to-door accuracy, calibrate the time band, model waits and transfers properly, improve the live update, and keep the model current. Do not replace the core (lookup tables + XGBoost + RAPTOR).

## 2. Repo and data map
| Path | What it is |
|---|---|
| model/scripts/process_day.py | One day of pings → stop passings (E:\hackathon-work\pass\pass_DD.npz) |
| model/scripts/aggregate.py | segment_times.csv, stop_headways.csv, holdout_trips.csv (built from 1–24 Oct) |
| model/scripts/build_patterns.py | Route patterns, stop distances along shapes |
| model/v3/scripts/build_rides.py | Ride-level dataset with the 4-fold leakage scheme |
| model/v3/scripts/xgb_final.py | XGBoost training (GPU) |
| model/v3/predict.py | Serving-side correction; reads v3_lookup.npz |
| planner/ttmodel.py | Lookup chain, median→mean blend, waits |
| planner/preprocess.py, planner/router.py | Network build; frequency-based RAPTOR, ranking, labels |
| planner/ (rest) | FastAPI app, UI, tests; run.bat / run.sh |
| realtime/ | Replay feed, live ratios, next-bus ETA |
| eval/ | Test pairs, ticket-name matching, Google comparison |
| docs/LOG.md | Decision log |

Data: E:\hackathon-data is raw and read-only. E:\hackathon-work holds v1 intermediates.
Machine: Windows, 20 logical CPUs, 64 GB RAM, RTX 3090.
Settings today: env vars V3_MODE, SEG_MODE, REALTIME, GTFS_DIR, MODEL_DIR, PORT; plus BLEND_S and USE_PAIR (check where these are read).

## 3. Hard rules
- Never write under E:\hackathon-data. Write v2 outputs under E:\hackathon-work\v2\<role>\. Never overwrite v1 files, so v1 numbers stay reproducible.
- Fixed splits:
  - Tables and final ML fit: 1–24 Oct
  - ML train 1–18, validation 19–24
  - Test: 25–31 Oct
  - Live-layer tuning day: 22 Oct; live test days: 28 and 30 Oct
- Nothing is tuned on 25–31 Oct: no hyperparameters, thresholds, shifts, step sizes or time constants. Only the eval agent runs the test set, and only at a gate.
- Every change ships behind a flag (section 5) and defaults to v1 behaviour until its gate passes.
- One docs/LOG.md entry per change: date, task ID, what, why, validation metrics before and after.
- /journey p95 latency stays under 500 ms with all v2 flags on.
- No new heavy dependencies (no deep-learning frameworks). Use numpy, pandas, xgboost, fastapi and what is already installed. If you need anything else, log why first.
- Code must run on Windows: use pathlib, no bash-only steps; scripts in PowerShell or .bat.
- Stay inside your role's directories (section 6). Need a change elsewhere? Put it in your handoff note for the owner.
- No raw data or large binaries in git.

## 4. Shared definitions
- Ride-length buckets: <10, 10–30, 30–60, >60 min.
- Peak: weekdays 08:00–10:59 and 17:00–20:59; everything else is off-peak. Fixed; do not tune.
- Day types: weekday, weekend, holiday. Holidays: 2 Oct (Gandhi Jayanti), 20 Oct (Dussehra), plus any other gazetted Delhi holiday in October. Pool holidays with weekends when building tables.
- Band: p50 = typical (duration_s), p85 = heavy (duration_p85_s). Target: 85% of real journeys finish at or under p85.
- Metrics (all in eval/metrics.py):
  - error: median of |pred − actual| / actual, plus share within ±20%
  - bias: mean and median of pred − actual, in seconds and in %
  - coverage: share with actual ≤ p85
  - width: median of (p85 − p50) / p50
  - pinball loss at 0.85 for the p85 prediction
  - calibration: share of actuals under the predicted 50th, 85th and 90th percentiles
- Noise floor: for (route, board stop, alight stop, hour) cells with ≥10 test-week rides, the median error of each ride against its own cell median.
- Expected wait for a random arrival: mean(H²) / (2 × mean(H)), capped at 1,200 s.
- Log-normal from a band: μ = ln p50, σ = ln(p85 / p50) / 1.0364.

## 5. Flags and contracts
| Flag | Values [default] | Owner |
|---|---|---|
| V2_BAND | 0 / 1 [0] | ml |
| V2_KNN | 0 / 1 [0] | ml |
| V2_WAIT | v1 / gaps / pooled [v1] | data, planner |
| V2_RELIABLE | 0 / 1 [0] | planner |
| V2_LIVE | ratio / kf [ratio] | realtime |
| V2_FALLBACK | 0 / 1 [0] | ml |
| ADMIN_TOKEN | string [unset = admin endpoints disabled] | planner |

Contracts. Owners may refine these, but must log the change and tell every consumer in a handoff note.
1. passing_index (data → eval, planner): E:\hackathon-work\v2\data\passing_index_DD.npz, one per day 1–31. Per stop, sorted arrays of (t_seconds, route_id, trip_id), plus per-route usable-trip share for that day. The planner reads days 1–24 only. (See A1: plus stop_visits_DD.npz.)
2. stop_headways_v2.csv (data → planner): stop_id, route_id, hour, daytype, n, median_headway_s, mean_headway_s, mean_sq_headway_s, wait_p85_s, usable_share.
3. Band artifacts (ml → planner): MODEL_DIR\v2\xgb_quantile.json, conformal_shifts.json keyed by bucket|peak, and shifts_d2d.json. predict.py returns p50_s, p85_s per leg, with p85 ≥ p50 always.
4. Live interface (realtime → planner), Python: `segment_state(seg_id, now, horizon_s) -> (log_ratio, var)`; `next_buses(stop_id, route_id, now) -> [(eta_s, sd_s), ...]`
5. /journey additions: duration_p85_s (calibrated), on_time_probability (when arrive_by is given), label most_reliable, per-leg alternatives and transfer_success_probability (live mode), and band_version.
6. Realized journeys (eval → ml, humans): E:\hackathon-work\v2\eval\realized_{val,test}.csv with od_id, day, depart_s, plan_*, real_wait_s, real_ride_s, real_transfer_s, real_total_s, scored.

## 6. Roles
| Agent | Owns | Human lane |
|---|---|---|
| eval | eval/, docs/results/ | Person A |
| ml | model/v3/ | Person A |
| data | model/scripts/, planner/ttmodel.py, tools/ | Person A |
| realtime | realtime/ | Person B |
| planner | planner/ except ttmodel.py | Person B |

Humans approve gates and merges. Agents never mark a gate as passed themselves.

## 7. Workflow
- One git worktree and branch per agent: v2/eval, v2/ml, v2/data, v2/realtime, v2/planner.
- Merge order: data (T0.1, T0.4) → eval (T0.2–T0.5) → ml → realtime → planner.
- After each merge, run `python eval/report.py --set val` and paste the output into the handoff note.
- One handoff note per task in docs/handoff/<task-id>.md: what changed, flag, files, how to run, validation metrics, open issues.
- Definition of done: behind a flag; pytest unit tests pass on Windows; validation metrics in the handoff note; LOG.md entry; latency rule holds.

## 8. Task cards (format: ID Title · owner · depends on)

### Phase 0: measure first
**T0.1 Passing index · data · none.** Build passing_index_DD.npz (contract 1) from the existing pass files for days 1–31. Done when each day loads in under 5 s and a spot check of three busy stops shows plausible buses per hour.

**T0.2 Realized door-to-door journeys · eval · T0.1.** New eval/realized_journeys.py. Call the planner in-process, not over HTTP. For each OD and departure: walk at the planner's walking speed and detour factor; take the first real bus of the leg's route that passes the boarding stop after you arrive (same day); use that trip's real passing time at the alighting stop; repeat for each transfer. Mark a journey unscored if any leg's route has usable share < 0.8 that day. Sets: the 60 ticket pairs at their departure times, on every weekday; 5,000 random OD-times (seed 2026, stops 2–20 km apart, departures 06:00–22:00). Days 19–24 go to realized_val.csv and days 25–31 to realized_test.csv (contract 6). Done when at least 90% of the 60 pairs, transfer pairs included, are scored on at least 3 days.

**T0.3 Metrics and noise floor · eval · none.** eval/metrics.py with every metric in section 4, by bucket × peak, plus unit tests on toy arrays. Noise floor on the 600k test rides.

**T0.4 Protocol fixes · eval + ml + data · none.** data: holiday day type (section 4) in aggregate.py; build a second table set from 1–18 Oct for validation-time selection. eval: GTFS baseline on the 600k test rides; blend column in the ride-length table. ml: choose BLEND_S from {300, 450, 600, 900, 1200} on 19–24 rides using the 1–18 tables; check the tree count every 10 trees, not every 100.

**T0.5 One-command report · eval · T0.2, T0.3.** `python eval/report.py --set {val,test}` writes docs/results/v2_<date>_<set>.md with: ride error by bucket for GTFS, blend and XGBoost; band metrics; door-to-door error for direct and transfer journeys; the noise floor. Gate G0: runs in under 10 minutes.

### Phase 1: calibrated time band
**T1.1 Quantile model · ml · T0.4.** In xgb_final.py: objective="reg:quantileerror", quantile_alpha=[0.5, 0.85], the same 20 features and log-ratio target, the same split. Early-stop on validation pinball loss. In predict.py behind V2_BAND: p50 comes from the existing model; p85 = base × exp(q85), floored at p50.

**T1.2 Conformal shifts · ml · T1.1, T0.3.** Fit on 1–18. On 19–24 rides compute scores s = log(actual / base) − q85. Per bucket × peak, shift = the ⌈(n + 1) × 0.85⌉-th smallest score. If a cell has n < 500, pool it with its neighbour. Serve the 1–24 model with these shifts. (See A2.)

**T1.3 Online adjustment in replay · realtime · T1.2.** Adaptive conformal inference per bucket: α ← α + γ(0.15 − miss), with γ = 0.005. The band top uses the validation-score quantile at 1 − α. Update after each completed ride in the replay. Report hourly coverage on 28 and 30 Oct.

**T1.4 Nearest-rides band (optional) · ml · T1.2.** Behind V2_KNN, for rides over 60 min: take the 50 nearest historical rides (route, board and alight positions, hour, day type), fit a log-logistic, read p85. Keep it only if it reaches the same validation coverage with a narrower band.

**Gate G1 (eval):** coverage between 82% and 88% in every bucket; on the replay days, at least 12 of 16 hours fall between 80% and 90%. (A4: on validation.)

### Phase 2: waits and transfers
**T2.1 Headway statistics and wait formula · data · T0.1, T0.4.** Build stop_headways_v2.csv (contract 2) from days 1–24. wait_p85_s: simulate random arrivals over the empirical gaps. Pick a gap with probability proportional to its length, then draw the wait uniformly within it. In ttmodel.py with V2_WAIT=gaps: use the expected wait from section 4 when usable_share ≥ 0.8 and n ≥ 20; otherwise fall back to v1. (See A1.)

**T2.2 Pooled parallel routes · planner · T2.1.** Applies when V2_WAIT=pooled. The attractive set is every route that serves board → alight in order with ride time ≤ 1.1 × the best. The wait comes from the gaps between consecutive buses of any route in the set, at the boarding stop (days 1–24, same day type, query hour). Cache per (stop, route set, hour, day type). Return the set as alternatives.

**T2.3 Transfer success probability · planner + realtime · T1.2, T3.4.** Live mode only. P(A + walk < D), where A is log-normal from leg 1's live-adjusted p50/p85 and D ~ Normal(eta_s, sd_s) for the connecting bus. Use 2,000 samples with a fixed seed. If P ≥ 0.7, plan on that bus and show the following bus as the backup. If P < 0.7, plan on the following bus and show the first as "if you make it".

**T2.4 Deadline and on-time probability · planner · T1.2.** Optional arrive_by on /journey. Per option, build the log-normal from its calibrated band (section 4): on_time_probability = Φ((ln D − μ) / σ). When arrive_by is set, rank options by that probability.

**T2.5 Most reliable option · planner · T2.1.** Behind V2_RELIABLE=1: run a second time-optimal RAPTOR pass with p85 segment times and wait_p85_s. Label the result most_reliable if its itinerary differs from the fastest.

**T2.6 Door-to-door calibration · ml + eval · T0.2, T2.1.** Same method as T1.2, applied to realized door-to-door totals from realized_val.csv, writing shifts_d2d.json. The planner applies it to the total p85.

**Gate G2 (eval):** realized wait bias within ±60 s; realized totals fall under the predicted 50th, 85th and 90th percentiles 47–53%, 82–88% and 87–93% of the time. (A4: on validation.)

### Phase 3: live layer
**T3.1 Per-segment Kalman update · realtime · none.** Behind V2_LIVE=kf. State per segment: x = log(observed / historical), with variance P. Measurement noise R = σ², with σ from the segment's p50/p85 (section 4). Between measurements, x relaxes toward its 3×3 km cell: x_c + (x − x_c)·e^(−Δ/τs); the cell relaxes toward 0 with time constant τc; P grows by q·Δ. The forecast at horizon h uses the same relaxation. Keep the existing [0.6, 2.5] clip on exp(x). Fit τs, τc and q on 22 Oct only.

**T3.2 Per-bus running ratio · realtime · T3.1.** For each live bus, b = mean of its last 5 segment log ratios. Remaining time = Σ historical × exp(w·b + (1 − w)·x(h)). Fit w on 22 Oct.

**T3.3 Baselines and replay days · realtime · none.** Build replay_22.npz and replay_30.npz with the same settings as replay_28.npz. Next-bus baselines: half-headway (v1) and GTFS scheduled next arrival. Report mean and median absolute error, and the share within ±2 min.

**T3.4 Live interface · realtime · T3.1, T3.2.** Implement contract 4. Take sd_s from 22 Oct residuals, by horizon bucket.

**Gate G3 (eval):** next-bus error below v1 live on both 28 and 30 Oct, using the same metric as the report's 107 s; no ride-error regression at 07:30–08:30.

### Phase 4: upkeep and API
**T4.1 Paired-route matching · data · none.** Full run with USE_PAIR=1 into E:\hackathon-work\v2\pass\. Report usable share before and after.

**T4.2 Learned fallback for no-data segments · ml · T0.4.** Behind V2_FALLBACK=1: XGBoost on log seconds per km, with features: segment length, hour, day type; route mean speed and schedule speed; speed of other routes' buses in the same 3×3 km cell and hour; distance to Connaught Place. Train on covered segments (1–24 Oct). Validate by masking 10% of covered segments, grouped by segment. It replaces chain levels 3–4 in ttmodel.py; coordinate with data. Done when it beats both schedule × factor and distance ÷ 15 km/h on the masked segments.

**T4.3 Nightly refresh · data · T0.4, T1.2.** tools\nightly.ps1, run from Task Scheduler: process yesterday; rebuild tables over the last 24 days; warm-start XGBoost with 10 trees on the newest 6 days, using features from tables that exclude those days; POST /admin/reload-model with the token. Log to E:\hackathon-work\v2\logs\. Rehearse by refreshing for 25 Oct with data up to 24 Oct. Must finish in under 15 min end to end.

**T4.4 API fields and watch · planner · T2.3, T2.4, T2.5, T3.4.** Add the contract 5 fields. POST /journey/watch returns a watch_id. GET /journey/watch/{id} returns on_track, transfer_at_risk or replanned, plus the new itinerary when replanned. Re-check every 15 s of replay clock. Replan when transfer P < 0.5 or the arrival estimate slips by more than 5 min. Keep watches in memory; they expire after 2 h.

**T4.5 Admin tokens and preload · planner · none.** Require an X-Admin-Token header on POST /admin/reload-model and POST /realtime/clock (401 otherwise; disabled when ADMIN_TOKEN is unset). Load all models at startup; /health reports models_loaded.

**Gate G4 (eval):** ride error no worse than v1 (within +0.1 points); nightly job under 15 min; /journey p95 under 500 ms over 200 queries with all flags on. (A4: on validation.)

### Phase 5: report
**T5.1 Report update · eval, humans approve · G1–G4.** Corrected numbers and tables: GTFS baseline on the 600k rides; blend column per ride-length bucket; n in every table (table 4.2 covers about 37–38 direct pairs, not 60); 35 M vs ~1.5 M/day passings; 2,573 vs 2,574 routes; "always shown" vs the within-20% rule; 8.4 vs 8.5 min. New sections: band calibration; realized door-to-door results (direct and transfer); noise floor; live layer on two test days; paired Google wins. Cite the papers from the literature survey.

## 9. Kickoff prompts
- eval: "You are the eval agent. Read AGENTS.md sections 1–7 and 10, then cards T0.2, T0.3, T0.5 and T5.1. Start with T0.3; begin T0.2 once T0.1 lands."
- ml: "You are the ml agent. Read AGENTS.md sections 1–7 and 10, then cards T0.4 (your part), T1.1, T1.2, T1.4, T2.6 and T4.2. Start with T0.4."
- data: "You are the data agent. Read AGENTS.md sections 1–7 and 10, then cards T0.1, T0.4 (your part), T2.1, T4.1 and T4.3. Start with T0.1."
- realtime: "You are the realtime agent. Read AGENTS.md sections 1–7 and 10, then cards T3.3, T3.1, T3.2, T3.4 and T1.3. Start with T3.3."
- planner: "You are the planner agent. Read AGENTS.md sections 1–7 and 10, then cards T4.5, T2.2, T2.4, T2.5, T2.3 and T4.4. Start with T4.5."

## 10. Amendments (2026-11-24, 19:30 IST). These override sections 1–9 where they conflict

Reviewed before kickoff. Reasons in brackets.

**A1. Waits and headways come from raw stop visits, not map-matched passings.** `pass_DD.npz` holds only the ~68% of trips that map-matched cleanly, so "first real bus after arrival" and headways built from it skip about a third of real buses, which inflates waits. The `usable_share ≥ 0.8` rule would also exclude most routes. Instead, T0.1 builds `stop_visits_DD.npz`: for each (stop, route), the times a bus with that route_id was within 60 m of the stop (first ping in the visit; consecutive pings of one vehicle within 120 s are one visit). It comes from the raw SQLite, read-only, and runs detached and parallel over days. `passing_index` (from pass files) stays the source for **ride times**. Realized journeys (T0.2) use stop visits for boarding and waits, and passings or visits for alighting times. Report the share of journeys scored instead of dropping routes by usable_share. Keep usable_share only as a diagnostic column.

**A2. Conformal scores are out-of-fold.** T1.2 computes scores from the existing 4-fold out-of-fold predictions on days 1–24 (each fold predicted by a model not trained on it, CV+ style). Validation coverage is then checked on days 19–24 using the 1–18 model, and the 1–24 model is served with the out-of-fold shifts. (A refit on more data has smaller residuals, so shifts from 19–24 residuals of a 1–18 model would over-cover.)

**A3. Journey band ≠ sum of leg p85s.** Combine legs in log-normal space (sum of medians; variance from each leg's σ, waits included) and calibrate the **door-to-door** total with T2.6. This is core, not optional.

**A4. Gates are checked on validation (19–24 Oct) only.** Test (25–31) runs once at the end, for the report (T5.1), with no iteration after seeing it. Gate thresholds stay as written but apply to `--set val`.

**A5. Scope for today (v2-today).** Do: T0.1 (with A1), T0.2, T0.3, T0.4, T0.5, T1.1, T1.2 (A2), T2.1, T2.2, T2.4, T2.6, T3.1, T3.3, T3.4, T4.5. **Deferred** (keep the cards; don't start): T1.3, T1.4, T2.3, T2.5, T4.1, T4.2, T4.3, T4.4, T5.1.

**A6. Working setup.** Agents run from a Mac and reach Windows only via `ssh -o ControlPath=~/.ssh/cm-%r@%h:%p -o BatchMode=yes <user>@<windows-host>` (cmd.exe shell). Never log in with a password; if SSH fails, write `BLOCKED:` in your handoff note and stop. Worktrees: `E:\hackathon-wt\<role>` on branch `v2/<role>`, created from main at 68473b9. Edit files there (scp from the Mac, or pipe Python), run code there, and commit on your branch. Gitignored inputs are not in worktrees: use `MODEL_DIR=E:\hackathon-project\model` for the v1 CSVs, and the Python env `E:\hackathon-project\planner\.venv\Scripts\python.exe` (system `python` also has numpy, pandas, xgboost 2.1.4, lightgbm). Long jobs run detached: `python E:\hackathon-work\launch.py <log> <script> ...`. GPU: xgboost 2.1.4, `device="cuda"`. The user is not an admin (no tasklist, taskkill or wmic). Don't touch `E:\hackathon-project` main or the running API on port 8000.

**A7. Time box.** Commit work in progress to your branch at least every ~20 min, and finish with a handoff note `docs/handoff/<task-id>.md` per task, even if partial: state what is done, what isn't, and the numbers. Partial and honest beats complete and unverified.
