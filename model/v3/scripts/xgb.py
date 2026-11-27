"""v3: XGBoost (GPU) on the same out-of-fold training rides / target / features as train.py.
Run on Windows: python launch.py E:\\hackathon-work\\v3\\xgb_log.txt E:\\hackathon-work\\v3\\xgb.py
Needs xgboost==2.1.4 (3.x cannot see this driver and silently uses CPU). Output: xgb_results.txt
Number of rounds chosen by early stopping on fold 4 (days 19-24), never on the test days.
"""
import os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
sys.argv = [sys.argv[0]]
D = os.path.join(os.environ.get("WORK_DIR", r"E:\hackathon-work"), "v3")  # env WORK_DIR
t0 = time.time()
tr = pd.concat([pd.read_parquet(os.path.join(D, f"fold{i}.parquet")).assign(fold=i) for i in range(1, 5)], ignore_index=True)
te = pd.read_parquet(os.path.join(D, "test.parquet"))

def feats(d):
    X = pd.DataFrame(index=d.index)
    X["log_base_med"] = np.log(d.base_med); X["skew"] = d.base_mean / d.base_med; X["n_stops"] = d.n_stops
    X["log_dist"] = np.log1p(d.dist_m); X["base_speed"] = d.dist_m / d.base_med; X["hour_f"] = d.hour_f
    X["weekend"] = d.dt; X["route_len_m"] = d.route_len_m; X["route_nstops"] = d.route_nstops
    X["route_speed"] = d.route_speed; X["log_route_obs"] = np.log1p(d.route_obs); X["route_cov"] = d.route_cov
    X["route_skew"] = d.route_skew; X["board_pos"] = d.board_pos
    X["b_lat"] = d.b_lat; X["b_lon"] = d.b_lon; X["a_lat"] = d.a_lat; X["a_lon"] = d.a_lon
    X["fb_frac"] = d.fb_frac; X["mean_logn"] = d.mean_logn
    return X
# "plus_p85": stronger feature set adds the p85 sum (skew of the upper tail); the planner has p85 too
def feats_plus(d):
    X = feats(d); X["p85_ratio"] = d.base_p85 / d.base_med; return X

ytr = np.log(tr.actual_s / tr.base_med).to_numpy()
a = te.actual_s.to_numpy()
val = (tr.fold == 4).to_numpy()
lines = []
def met(p):
    e = p - a; ape = np.abs(e) / a
    return f"medAPE {100*np.median(ape):.2f}%  MAPE {100*np.mean(ape):.2f}%  within20 {100*np.mean(ape<0.2):.1f}%  bias {np.mean(e):+.0f}s"
lines.append(f"sum_median         {met(te.base_med.to_numpy())}")
for fname, ff in [("all", feats), ("plus_p85", feats_plus)]:
    Xtr, Xte = ff(tr), ff(te)
    for obj in ["reg:absoluteerror", "reg:squarederror"]:
        p = dict(objective=obj, device="cuda", tree_method="hist", learning_rate=0.05, max_depth=7, min_child_weight=200,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, eval_metric="mae")
        dtr = xgb.DMatrix(Xtr[~val], ytr[~val]); dva = xgb.DMatrix(Xtr[val], ytr[val])
        b = xgb.train(p, dtr, 1500, evals=[(dva, "val")], early_stopping_rounds=50, verbose_eval=False)
        it = max(b.best_iteration + 1, 50)
        b = xgb.train(p, xgb.DMatrix(Xtr, ytr), it)
        pred = te.base_med.to_numpy() * np.exp(b.predict(xgb.DMatrix(Xte)))
        nm = f"xgb_{obj.split(':')[1][:3]}_{fname}"
        lines.append(f"{nm:18s} {met(pred)}  ({it} rounds)")
        for lo, hi, bn in [(0, 600, "<10"), (600, 1800, "10-30"), (1800, 3600, "30-60"), (3600, 1e9, ">60")]:
            m = (a >= lo) & (a < hi)
            lines.append(f"    {bn:6s} min medAPE {100*np.median(np.abs(pred[m]-a[m])/a[m]):.2f}%  (sum_median {100*np.median(np.abs(te.base_med.to_numpy()[m]-a[m])/a[m]):.2f}%)")
        print(lines[-5], flush=True)
        b.save_model(os.path.join(D, f"{nm}.json"))
out = "XGBoost GPU (test rides days 25-31, n=%d), %.0fs\n" % (len(te), time.time() - t0) + "\n".join(lines)
open(os.path.join(D, "xgb_results.txt"), "w").write(out); print(out); print("DONE")
