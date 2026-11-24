# Model v3: ML refinement of bus ride times (honest result: small gain)

Question: can a learned model beat "sum the segment medians/means from the lookup table" (model/segment_times.csv) for
predicting one bus ride (board stop to alight stop on one route, given a departure time)?

**Short answer: a little, not by much.** On 600k held-out rides (25–31 Oct), the best LightGBM residual model gets
a median error of **12.36%** vs **12.69%** for the plain sum of medians, **12.43%** for a one-line calibration, and
about the same as the median/mean blend the planner already uses. The ML gain over the simple calibration/blend
(~0.1 pt) is **not meaningful**. ML helps clearly only on rides where the lookup has little data
(fallback share > 50%: 22.7% to 19–21%) and on weekends. **Recommendation: keep the planner's median→mean blend
(already live), optionally add the bucket calibration; treat LightGBM as an optional extra** (packaged and
ready, see INTEGRATION.md).

## Setup (no test leakage)

- **Test rides**: 600k rides from held-out days 25–31 Oct (2 random board<alight pairs per GPS trip, same filters as
  `model/holdout_trips.csv`: duration > 0, ≤ 3 h, ≤ 70 km/h). Features use the deployed lookup built from days 1–24.
  The "one ride per trip" subset (397k) matches `holdout_trips.csv`'s sampling and gives the same ranking.
- **Training rides**: 1.2M rides from days 1–24, **out-of-fold**: 4 folds of 6 days; rides of a fold get features
  from a lookup built only on the other 18 days, so the target never leaks into its own features.
  Days 25–31 are never used for training, model choice or tuning.
- **Target**: `log(actual_s / base_med)`; prediction = `base_med × exp(model)`, clipped ×0.5–2.5 in predict.py.
- **Base** (`base_med`, `base_mean`, `base_p85`): time-dependent sum over the ride's segments of the
  (route segment, hour, daytype) cell with n ≥ 3, else the segment's all-hour value for the daytype, else any
  daytype, else distance / network speed for that hour (mimics planner/ttmodel.py; hour advances along the ride).
- **Features** (21): log base_med, skew = base_mean/base_med, n_stops, log distance, base speed, hour (fractional),
  weekend, route length / #stops / observed speed / #observations / coverage / skew, board position along route,
  board & alight lat/lon, share of segments falling back (fb_frac), mean log n of the cells.
  Day-of-week was dropped: with ~3 of each weekday in training it only learns one-off day effects (try 1).

## Results (test rides, days 25–31, n = 600,000)

| predictor | median APE | mean APE | within ±20% | bias | MdAPE <10 min | 10–30 | 30–60 | >60 min |
|---|---|---|---|---|---|---|---|---|
| sum of medians (v2 table) | 12.69% | 18.48% | 67.8% | −120 s | 13.18 | 11.36 | 12.97 | 18.33 |
| sum of means | 13.58% | 20.89% | 64.9% | −34 s | 15.71 | 11.81 | 11.40 | 13.36 |
| median→mean ramp blend (10→40 min) | 12.45% | 18.33% | 68.3% | −74 s | 13.19 | 11.51 | 11.99 | 13.36 |
| bucket calibration (20 buckets of base) | 12.43% | 18.42% | 68.4% | −76 s | 13.47 | 11.18 | 11.50 | 14.59 |
| linear calibration (LAD in log space) | 12.46% | 18.42% | 68.4% | −75 s | 13.60 | 11.15 | 11.41 | 14.52 |
| **LightGBM residual, L1 (try 3, shipped: `lgbm_l1_no_latlon`, 592 trees, picked by grouped CV)** | **12.32%** | **17.91%** | **68.9%** | −71 s | see results.txt | | | |
| LightGBM residual, L1 (try 2) | **12.36%** | **18.19%** | **68.7%** | −75 s | 13.50 | 11.08 | 11.34 | 14.42 |
| LightGBM residual, L2 (try 2) | 12.89% | 18.75% | 67.4% | −57 s | 14.59 | 11.29 | 11.16 | 13.81 |
| XGBoost GPU, abs. error (50 rounds) | **12.32%** | **18.04%** | **68.8%** | −73 s | 13.43 | 11.07 | 11.30 | 14.12 |
| XGBoost GPU, squared error | 12.88% | 18.70% | 67.4% | −56 s | 14.58 | 11.30 | 11.11 | 13.71 |
| LightGBM try 1 (train days 18–24 only, with day-of-week) | 12.94% | 18.55% | 66.9% | −91 s | 13.60 | 11.85 | 12.60 | 16.17 |

XGBoost with an extra p85/median feature gives identical numbers (12.32%), so the p85 sum adds nothing.
Planner blend (median for the first 600 s of a ride, mean after — what planner/ttmodel.py does now) is the
`planner_blend600` row of the next train.py run (test.parquet was rebuilt with that column after try 3 had
loaded it; it is close to the ramp blend, 12.45%). Try 3 (tree count and variant by 4-fold grouped CV on days
1–24): L1 variants 12.30–12.32%, L2 12.86–12.87%; CV mean |log error| 0.1909 vs 0.1960 for no correction.
predict.py self-test: 500 test rides, max relative difference vs LightGBM 0.5% (float16 lookup), 0.6 ms/ride.

By hour band (median APE): AM peak 07–11: sum of medians 10.62, calibration 10.48, LightGBM 10.40;
PM peak 16–21: 15.59 / 14.92 / 14.88. Weekend: 13.32 / 12.89 / 12.78. Rides where > 50% of segments fall back
(n = 18.6k): 22.7 / 23.1 / 20.8 (LightGBM L2: 19.2).

Feature importance (LightGBM L1, try 2, gain): skew (mean/median) 36%, n_stops 20%, hour 18%, fallback share 11%,
base speed 5%, route speed 3%; everything else < 3%. I.e. the model mostly learns "long rides on skewed segments
need the mean, not the median" (which the blend already does) and "trust the table less where it has no data".

## Why the gain is small

Most of the remaining error is ride-to-ride noise that no static feature can see: the same route, stops and hour
differ by ±20–30% between days/buses (signals, dwell, crowding, incidents). Day-level effects dominate (the
per-fold baseline error ranges 12.1–13.05%). The realtime layer (realtime/, live segment ratios) is the right tool
for that and gave a bigger gain (13.0% → 12.3% on 28 Oct) than any static ML here.

## Files

| file | what |
|---|---|
| `predict.py` | `correct(route_id, board_stop, alight_stop, depart_dt, base_median_s, base_mean_s, n_stops, dist_m) -> seconds`. Pure python + numpy (LightGBM not needed: trees are evaluated from the JSON dump). `V3_MODE=lgbm|calib|off`. `python predict.py` self-tests against LightGBM's own predictions. |
| `v3_lookup.npz` | 10 MB: per (route segment, daytype, hour) median/mean (float16, fallbacks filled) and n, route patterns, stop coords, route features (lookup days 1–24) |
| `lgbm_model.json` / `.txt` | LightGBM residual model (JSON dump used by predict.py) |
| `calib.json` | chosen model name, feature order, bucket calibration table, linear calibration coefs |
| `results.txt`, `results_try1.txt`, `results_try2.txt`, `xgb_results.txt`, `importance.txt` | full outputs |
| `check_rides.json` | 500 test rides with LightGBM's predictions, for predict.py's self-test |
| `scripts/` | build_rides.py, train.py, xgb.py, fetch.sh |
| `INTEGRATION.md` | how the planner would call it |

## How to re-run (Windows, in `E:\hackathon-work\v3`)

Inputs: `E:\hackathon-work\pass\pass_DD.npz` (stop passings, from model/scripts/process_day.py),
`E:\hackathon-work\route_patterns.csv`, `E:\hackathon-data\gtfs\stops.txt` (read only). Python: numpy, pandas,
pyarrow, scikit-learn, lightgbm 4.7 (`pip install --user`), xgboost==2.1.4 for xgb.py (not 3.x).
Each step runs detached with `python launch.py <log> <script> <args>` (survives SSH drops); all take < 5 min.

```bat
cd /d E:\hackathon-work\v3
:: 1. out-of-fold training rides (4 folds, ~30 s each, can run in parallel) -> fold1..4.parquet
python build_rides.py 07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 01,02,03,04,05,06 fold1 2 300000
python build_rides.py 01,02,03,04,05,06,13,14,15,16,17,18,19,20,21,22,23,24 07,08,09,10,11,12 fold2 2 300000
python build_rides.py 01,02,03,04,05,06,07,08,09,10,11,12,19,20,21,22,23,24 13,14,15,16,17,18 fold3 2 300000
python build_rides.py 01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18 19,20,21,22,23,24 fold4 2 300000
:: 2. test rides (held-out 25-31, deployed lookup 1-24) -> test.parquet ; --export also writes v3_lookup.npz
python build_rides.py 01,02,...,24 25,26,27,28,29,30,31 test 2 600000 --export
:: 3. compare predictors + train LightGBM -> results.txt/json, lgbm_model.json/txt, calib.json, importance.csv,
::    check_rides.parquet, test_preds.parquet  (~3-8 min)
python train.py
:: 4. optional: XGBoost on GPU -> xgb_results.txt
python xgb.py
```
(write out `01,02,...,24` in full). Then copy `v3_lookup.npz lgbm_model.json lgbm_model.txt calib.json results*.txt
importance.csv` into `model/v3/` (`scripts/fetch.sh` does this from the Mac, and builds check_rides.json).

## Next steps (for whoever continues)

1. Decide: keep the planner's blend (default) or wire in `predict.py` per INTEGRATION.md (V3_MODE=calib is the
   cheap, low-risk choice; lgbm adds ~0.1 pt overall, ~2–3 pt on sparse-data rides).
2. Better ideas than more static ML: combine with the realtime layer (live ratios as a feature); learn the
   correction per leg *with* the realtime ratio; quantile models (p15/p85) for the reliability band.
3. If tuning further: model selection must use the grouped CV on days 1–24, never days 25–31.
