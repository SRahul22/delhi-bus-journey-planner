# Final XGBoost model (main submission)

Split: train = days 1-18, validation = days 19-24 (tree count by median APE + config choice), final refit = days 1-24, test = held-out days 25-31. Nothing was tuned on test.

Chosen config: depth 10, lr 0.03, min_child_weight 50, 100 rounds, objective reg:absoluteerror, GPU (xgboost 2.1.4). Run time 84 s.

| Test rides (days 25-31) | Median error | Mean error | Within ±20% | Bias |
|---|---|---|---|---|
| Sum of segment medians | 12.69% | 18.48% | 67.8% | -120 s |
| **XGBoost final** | 12.28% | 17.94% | 69.0% | -73 s |

| Ride length | XGBoost | Sum of medians |
|---|---|---|
| <10 min | 13.39% | 13.18% |
| 10-30 min | 11.02% | 11.36% |
| 30-60 min | 11.23% | 12.97% |
| >60 min | 13.99% | 18.33% |

Feature importance (gain %): n_stops 15.8, skew 14.7, fb_frac 13.7, hour_f 7.1, weekend 6.0, route_speed 4.0, base_speed 3.9, mean_logn 3.5

Serving: `model/v3/predict.py` with `V3_MODE=xgb` (needs xgboost in the planner venv).
