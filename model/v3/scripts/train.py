"""v3: compare ride-time predictors on held-out days and train the LightGBM residual model.

Inputs (from build_rides.py, in E:\\hackathon-work\\v3):
  fold1..4.parquet : rides of days 1-6 / 7-12 / 13-18 / 19-24, features from a lookup built on the other 18
                     training days (out-of-fold, no target leakage; days 25-31 never used)
  (first try, train.parquet = rides 18-24 with lookup 1-17, overfit day-of-week: see results_try1.txt)
  test.parquet  : rides of days 25-31, features from lookup built on days 1-24 (= the deployed tables)
Target: y = log(actual_s / base_med).
Outputs: results.txt, results.json, importance.csv, lgbm_model.txt, lgbm_model.json, calib.json, test_preds.parquet
"""
import json, os, sys, time
import numpy as np, pandas as pd
import lightgbm as lgb

D = r"E:\hackathon-work\v3"
t0 = time.time()
out_lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out_lines.append(s)

# out-of-fold training rides: 4 folds of 6 days, each with features from a lookup built on the other 18 days
TRAIN = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fold1", "fold2", "fold3", "fold4"]
tr = pd.concat([pd.read_parquet(os.path.join(D, f"{f}.parquet")).assign(fold=i) for i, f in enumerate(TRAIN)], ignore_index=True)
te = pd.read_parquet(os.path.join(D, "test.parquet"))
P(f"train rides {len(tr)} (days {sorted(tr.day.unique())}), test rides {len(te)} (days {sorted(te.day.unique())})")

def feats(d):
    X = pd.DataFrame(index=d.index)
    X["log_base_med"] = np.log(d.base_med)
    X["skew"] = d.base_mean / d.base_med
    X["n_stops"] = d.n_stops
    X["log_dist"] = np.log1p(d.dist_m)
    X["base_speed"] = d.dist_m / d.base_med
    X["hour_f"] = d.hour_f
    X["dow"] = d.dow
    X["weekend"] = d.dt
    X["route_len_m"] = d.route_len_m
    X["route_nstops"] = d.route_nstops
    X["route_speed"] = d.route_speed
    X["log_route_obs"] = np.log1p(d.route_obs)
    X["route_cov"] = d.route_cov
    X["route_skew"] = d.route_skew
    X["board_pos"] = d.board_pos
    X["b_lat"] = d.b_lat; X["b_lon"] = d.b_lon; X["a_lat"] = d.a_lat; X["a_lon"] = d.a_lon
    X["fb_frac"] = d.fb_frac
    X["mean_logn"] = d.mean_logn
    return X
FEATS = list(feats(tr.head()).columns)
Xtr, Xte = feats(tr), feats(te)
ytr = np.log(tr.actual_s / tr.base_med); yte = np.log(te.actual_s / te.base_med)

preds = {}
preds["sum_median"] = te.base_med.to_numpy()
preds["sum_mean"] = te.base_mean.to_numpy()
# planner-style blend (median short, mean long): weight on mean ramps 0->1 between 10 and 40 min of base_med
if "base_blend600" in te: preds["planner_blend600"] = te.base_blend600.to_numpy()   # what planner/ttmodel.py does now
w = np.clip((te.base_med - 600) / 1800, 0, 1); preds["blend_med_mean"] = ((1 - w) * te.base_med + w * te.base_mean).to_numpy()

# ---- calibration 1: multiplicative factor by base_med bucket (median of actual/base on train), log-interpolated
edges = np.quantile(np.log(tr.base_med), np.linspace(0, 1, 21))
cent, fac = [], []
lb = np.log(tr.base_med)
for a, b in zip(edges[:-1], edges[1:]):
    m = (lb >= a) & (lb <= b)
    cent.append(float(lb[m].median())); fac.append(float(np.median(ytr[m])))
calib = {"x_log_base_med": cent, "log_factor": fac}
def apply_calib(base_med):
    return base_med * np.exp(np.interp(np.log(base_med), cent, fac))
preds["calib_bucket"] = apply_calib(te.base_med.to_numpy())

# ---- calibration 2: least-absolute-deviation linear model in log space on a few features
from sklearn.linear_model import QuantileRegressor
def linX(d):
    return np.column_stack([np.log(d.base_med), np.log(d.base_mean / d.base_med), d.fb_frac])
sub = np.random.default_rng(0).choice(len(tr), min(60000, len(tr)), replace=False)
qr = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(linX(tr.iloc[sub]), np.log(tr.actual_s.iloc[sub]))
preds["calib_linear_log"] = np.exp(qr.predict(linX(te)))
calib["linear_log"] = {"coef": qr.coef_.tolist(), "intercept": float(qr.intercept_), "features": ["log_base_med", "log_mean_over_med", "fb_frac"]}
P("linear calib coef", calib["linear_log"])

# ---- LightGBM on the residual. Model/feature choice by a validation fold (days 19-24), never by the test days.
# dow is excluded: with only ~3-4 of each weekday in training it learns one-off day effects (try 1 overfit on it).
VARIANTS = {
    "all": [f for f in FEATS if f != "dow"],
    "no_latlon": [f for f in FEATS if f not in ("dow", "b_lat", "b_lon", "a_lat", "a_lon")],
    "small": ["log_base_med", "skew", "n_stops", "log_dist", "base_speed", "hour_f", "weekend", "fb_frac", "mean_logn",
              "route_speed", "route_skew"],
}
# Number of trees and variant chosen by 4-fold grouped CV (each fold = 6 whole days), metric = mean |log error|.
# (try 2 used a single validation fold: day-level offsets dominated and it stopped at ~20 trees.)
val = (tr.fold == tr.fold.max()).to_numpy()
folds = [(np.flatnonzero(tr.fold != f), np.flatnonzero(tr.fold == f)) for f in sorted(tr.fold.unique())]
models, vscore = {}, {}
for vn, fs in VARIANTS.items():
    for obj in ["l1", "l2"]:
        params = dict(objective={"l1": "regression_l1", "l2": "regression"}[obj], metric="l1", learning_rate=0.05,
                      num_leaves=31, min_data_in_leaf=1000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=10.0, verbose=-1, num_threads=16, seed=1)
        cv = lgb.cv(params, lgb.Dataset(Xtr[fs], ytr), 600, folds=folds, callbacks=[lgb.early_stopping(50, verbose=False)])
        k = [c for c in cv if c.endswith("l1-mean")][0]
        it = int(np.argmin(cv[k])) + 1
        name = f"lgbm_{obj}_{vn}"
        vscore[name] = float(np.min(cv[k]))
        bst = lgb.train(params, lgb.Dataset(Xtr[fs], ytr), it)  # refit on all folds
        models[name] = (bst, fs)
        preds[name] = te.base_med.to_numpy() * np.exp(bst.predict(Xte[fs]))
        P(f"{name}: {it} trees, CV mean|log err| {vscore[name]:.4f}, {time.time()-t0:.0f}s")
P(f"CV mean|log err| of sum_median (zero correction): {float(np.mean(np.abs(ytr))):.4f}")

# ---- metrics
def met(p, a):
    e = p - a; ape = np.abs(e) / a
    return dict(n=int(len(a)), medAPE=round(100 * float(np.median(ape)), 2), MAPE=round(100 * float(np.mean(ape)), 2),
                within20=round(100 * float(np.mean(ape < 0.2)), 1), bias_s=round(float(np.mean(e)), 0),
                med_err_s=round(float(np.median(e)), 0), MAE_s=round(float(np.mean(np.abs(e))), 0))
a = te.actual_s.to_numpy()
res = {"overall": {k: met(v, a) for k, v in preds.items()}}
P("\n== overall (test rides, days 25-31)")
P(pd.DataFrame(res["overall"]).T.to_string())
buckets = [(0, 600, "<10 min"), (600, 1800, "10-30 min"), (1800, 3600, "30-60 min"), (3600, 1e9, ">60 min")]
res["by_length"] = {}
for lo, hi, nm in buckets:
    m = (a >= lo) & (a < hi)
    res["by_length"][nm] = {k: met(v[m], a[m]) for k, v in preds.items()}
    P(f"\n== actual {nm} (n={m.sum()})")
    P(pd.DataFrame(res["by_length"][nm]).T[["medAPE", "MAPE", "within20", "bias_s"]].to_string())
res["by_hour"] = {}
hb = [(0, 7, "00-07"), (7, 11, "07-11 AM peak"), (11, 16, "11-16"), (16, 21, "16-21 PM peak"), (21, 24, "21-24")]
for lo, hi, nm in hb:
    m = ((te.hour_f >= lo) & (te.hour_f < hi)).to_numpy()
    res["by_hour"][nm] = {k: met(v[m], a[m]) for k, v in preds.items()}
P("\n== medAPE by hour band")
P(pd.DataFrame({nm: {k: v["medAPE"] for k, v in d.items()} for nm, d in res["by_hour"].items()}).to_string())
P("\n== within20 by hour band")
P(pd.DataFrame({nm: {k: v["within20"] for k, v in d.items()} for nm, d in res["by_hour"].items()}).to_string())
for nm, m in [("weekday", te.dt == 0), ("weekend", te.dt == 1), ("fb_frac>0.5", te.fb_frac > 0.5)]:
    m = m.to_numpy()
    P(f"\n== {nm} (n={m.sum()}) medAPE: " + ", ".join(f"{k}={met(v[m], a[m])['medAPE']}" for k, v in preds.items()))

# one ride per GPS trip (like model/holdout_trips.csv)
one = ~te.duplicated("trip").to_numpy()
res["one_per_trip"] = {k: met(v[one], a[one]) for k, v in preds.items()}
P(f"\n== one ride per trip (n={one.sum()})")
P(pd.DataFrame(res["one_per_trip"]).T[["medAPE", "MAPE", "within20", "bias_s"]].to_string())

# importance of the best LightGBM
best = min(vscore, key=vscore.get)   # chosen by grouped CV on the training days, not on the test days
bst, FS = models[best]
imp = pd.DataFrame({"feature": FS, "gain": bst.feature_importance("gain"), "splits": bst.feature_importance("split")})
imp["gain_pct"] = (100 * imp.gain / imp.gain.sum()).round(1)
imp = imp.sort_values("gain", ascending=False)
imp.to_csv(os.path.join(D, "importance.csv"), index=False)
P(f"\n== feature importance ({best})"); P(imp[["feature", "gain_pct", "splits"]].to_string(index=False))
bst.save_model(os.path.join(D, "lgbm_model.txt"))
json.dump(bst.dump_model(), open(os.path.join(D, "lgbm_model.json"), "w"))
json.dump({"best": best, "features": FS, "cv_mean_abs_log_err": vscore, "calib": calib}, open(os.path.join(D, "calib.json"), "w"), indent=1)
json.dump(res, open(os.path.join(D, "results.json"), "w"), indent=1)
# a small check set for verifying the pure-python predictor
chk = te.sample(2000, random_state=3)
chk = chk.assign(pred_lgbm=chk.base_med * np.exp(bst.predict(feats(chk)[FS])), pred_calib=apply_calib(chk.base_med.to_numpy()))
chk.to_parquet(os.path.join(D, "check_rides.parquet"), index=False)
pd.DataFrame({k: v for k, v in preds.items()}).assign(actual_s=a, hour_f=te.hour_f.values, n_stops=te.n_stops.values).to_parquet(os.path.join(D, "test_preds.parquet"), index=False)
open(os.path.join(D, "results.txt"), "w").write("\n".join(out_lines))
P(f"DONE {time.time()-t0:.0f}s")
