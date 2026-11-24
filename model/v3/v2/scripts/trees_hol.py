"""T0.4 follow-up (diagnostic, not served): tree count every 10 with the section-4 holiday day type.
Folds rebuilt by build_rides_v2.py with HOLIDAY_DT=1 (fold1_hol..fold3_hol = days 1-18, val19_24_hol = days 19-24),
so `weekend` = weekend or holiday (2, 20 Oct) in both the lookup tables and the feature. Same 3 configs as xgb_final.py."""
import json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
O = r"E:\hackathon-work\v2\ml"
t0 = time.time()
tr = pd.concat([pd.read_parquet(os.path.join(O, f"fold{i}_hol.parquet")) for i in (1, 2, 3)], ignore_index=True)
va = pd.read_parquet(os.path.join(O, "val19_24_hol.parquet"))
FEATS = ["log_base_med", "skew", "n_stops", "log_dist", "base_speed", "hour_f", "weekend", "route_len_m", "route_nstops",
         "route_speed", "log_route_obs", "route_cov", "route_skew", "board_pos", "b_lat", "b_lon", "a_lat", "a_lon", "fb_frac", "mean_logn"]
def feats(d):
    return pd.DataFrame(dict(log_base_med=np.log(d.base_med), skew=d.base_mean / d.base_med, n_stops=d.n_stops, log_dist=np.log1p(d.dist_m),
        base_speed=d.dist_m / d.base_med, hour_f=d.hour_f, weekend=d.dt, route_len_m=d.route_len_m, route_nstops=d.route_nstops,
        route_speed=d.route_speed, log_route_obs=np.log1p(d.route_obs), route_cov=d.route_cov, route_skew=d.route_skew,
        board_pos=d.board_pos, b_lat=d.b_lat, b_lon=d.b_lon, a_lat=d.a_lat, a_lon=d.a_lon, fb_frac=d.fb_frac, mean_logn=d.mean_logn))[FEATS]
ytr = np.log(tr.actual_s / tr.base_med).to_numpy()
dtr = xgb.DMatrix(feats(tr), ytr); dva = xgb.DMatrix(feats(va))
a = va.actual_s.to_numpy(); b = va.base_med.to_numpy(); late = (va.day >= 21).to_numpy()
mape = lambda p, m=slice(None): round(100 * float(np.median(np.abs(p[m] - a[m]) / a[m])), 3)
out = dict(base_all=mape(b), base_21_24=mape(b, late), blend1200_all=mape(va.base_blend1200.to_numpy()), configs=[])
print(out, flush=True)
base_cfg = dict(objective="reg:absoluteerror", device="cuda", tree_method="hist", subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, seed=0)
for g in [dict(learning_rate=0.05, max_depth=7, min_child_weight=200), dict(learning_rate=0.05, max_depth=9, min_child_weight=100),
          dict(learning_rate=0.03, max_depth=10, min_child_weight=50)]:
    m = xgb.train({**base_cfg, **g}, dtr, 1000)
    curve = [(r, mape(b * np.exp(m.predict(dva, iteration_range=(0, r))))) for r in range(10, 1001, 10)]
    r_best, e_best = min(curve, key=lambda x: x[1])
    p = b * np.exp(m.predict(dva, iteration_range=(0, r_best)))
    c = dict(cfg=g, best_rounds=r_best, val_all=e_best, val_21_24=mape(p, late), at100=dict(curve)[100], at500=dict(curve)[500])
    out["configs"].append(c); print(c, f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open(os.path.join(O, "t04_trees_holiday.json"), "w"), indent=1)
print("DONE")
