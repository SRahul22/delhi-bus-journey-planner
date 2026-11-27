"""FINAL submission model: XGBoost (GPU) ride-time correction.

Split (no leakage, nothing tuned on test):
  train      = out-of-fold rides from days 1-18  (folds 1-3; features from lookup tables built on the other days)
  validation = rides from days 19-24 (fold 4): number of trees (by median APE, every 100) + choice among a few configs
  test       = rides from held-out days 25-31 (features from the 1-24 lookup = what the planner serves)
The final model is refit on train+validation (days 1-24) with the chosen config and number of rounds.

Target: log(actual / sum_of_segment_medians); prediction = base_med * exp(model).
Run on Windows (detached, survives SSH drops):
  python E:\\hackathon-work\\launch.py E:\\hackathon-work\\v3\\xgb_final_log.txt E:\\hackathon-project\\model\\v3\\scripts\\xgb_final.py
Needs xgboost==2.1.4 (3.x cannot see this driver and silently runs on CPU).
Outputs (E:\\hackathon-project\\model\\v3): xgb_final.json (model), xgb_final_meta.json (features, config, metrics),
XGB_FINAL.md (results). With XGB_FINAL_COMMIT=1 it then commits them to git (the original run did).
Env: WORK_DIR (default E:\\hackathon-work; reads WORK_DIR/v3/fold1..4.parquet, test.parquet), XGB_DEVICE (default cuda; cpu without an NVIDIA GPU).
"""
import json, os, subprocess, sys, time
import numpy as np, pandas as pd, xgboost as xgb

D = os.path.join(os.environ.get("WORK_DIR", r"E:\hackathon-work"), "v3")  # env WORK_DIR
REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
OUT = os.path.join(REPO, "model", "v3")
DEVICE = os.environ.get("XGB_DEVICE", "cuda")  # "cpu" on machines without an NVIDIA GPU (results may differ slightly)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
log("xgboost", xgb.__version__)

tr = pd.concat([pd.read_parquet(os.path.join(D, f"fold{i}.parquet")).assign(fold=i) for i in range(1, 5)], ignore_index=True)
te = pd.read_parquet(os.path.join(D, "test.parquet"))
log(f"rides: train+val {len(tr)}, test {len(te)}")

FEATS = ["log_base_med", "skew", "n_stops", "log_dist", "base_speed", "hour_f", "weekend", "route_len_m", "route_nstops",
         "route_speed", "log_route_obs", "route_cov", "route_skew", "board_pos", "b_lat", "b_lon", "a_lat", "a_lon",
         "fb_frac", "mean_logn"]

def feats(d):
    X = pd.DataFrame(index=d.index)
    X["log_base_med"] = np.log(d.base_med); X["skew"] = d.base_mean / d.base_med; X["n_stops"] = d.n_stops
    X["log_dist"] = np.log1p(d.dist_m); X["base_speed"] = d.dist_m / d.base_med; X["hour_f"] = d.hour_f
    X["weekend"] = d.dt; X["route_len_m"] = d.route_len_m; X["route_nstops"] = d.route_nstops
    X["route_speed"] = d.route_speed; X["log_route_obs"] = np.log1p(d.route_obs); X["route_cov"] = d.route_cov
    X["route_skew"] = d.route_skew; X["board_pos"] = d.board_pos
    X["b_lat"] = d.b_lat; X["b_lon"] = d.b_lon; X["a_lat"] = d.a_lat; X["a_lon"] = d.a_lon
    X["fb_frac"] = d.fb_frac; X["mean_logn"] = d.mean_logn
    return X[FEATS]

Xtr, Xte = feats(tr), feats(te)
ytr = np.log(tr.actual_s / tr.base_med).to_numpy()
val = (tr.fold == 4).to_numpy()
a = te.actual_s.to_numpy(); base = te.base_med.to_numpy()

def metrics(p, act):
    e = p - act; ape = np.abs(e) / act
    return dict(median_ape=round(100 * float(np.median(ape)), 2), mean_ape=round(100 * float(np.mean(ape)), 2),
                within20=round(100 * float(np.mean(ape < 0.2)), 1), bias_s=round(float(np.mean(e)), 0), n=int(len(act)))

base_cfg = dict(objective="reg:absoluteerror", device=DEVICE, tree_method="hist", eval_metric="mae",
                subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0)
grid = [dict(learning_rate=0.05, max_depth=7, min_child_weight=200),
        dict(learning_rate=0.05, max_depth=9, min_child_weight=100),
        dict(learning_rate=0.03, max_depth=10, min_child_weight=50)]
dtr, dva = xgb.DMatrix(Xtr[~val], ytr[~val]), xgb.DMatrix(Xtr[val], ytr[val])
yva_act = tr.actual_s.to_numpy()[val]; bva = tr.base_med.to_numpy()[val]
best = None
for g in grid:
    p = {**base_cfg, **g}
    b = xgb.train(p, dtr, 2000, verbose_eval=False)
    # pick the number of trees by validation median APE (the metric we report), checked every 100 trees
    for r in range(100, 2001, 100):
        mv = metrics(bva * np.exp(b.predict(dva, iteration_range=(0, r))), yva_act)
        if best is None or mv["median_ape"] < best[2]["median_ape"]:
            best = (p, r, mv)
    log("config", g, "best so far", best[1], best[2])

p, rounds, mval = best
log("chosen", p, rounds)
final = xgb.train(p, xgb.DMatrix(Xtr, ytr), rounds)          # refit on days 1-24 (train + validation)
pred = base * np.exp(final.predict(xgb.DMatrix(Xte)))
res = {"test_xgb_final": metrics(pred, a), "test_sum_median": metrics(base, a), "validation_chosen": mval}
buckets = {}
for lo, hi, nm in [(0, 600, "<10 min"), (600, 1800, "10-30 min"), (1800, 3600, "30-60 min"), (3600, 1e9, ">60 min")]:
    m = (a >= lo) & (a < hi)
    buckets[nm] = {"xgb": metrics(pred[m], a[m]), "sum_median": metrics(base[m], a[m])}
hours = {}
for h in [8, 9, 13, 18, 19]:
    m = te.hour_f.to_numpy().astype(int) == h
    if m.any():
        hours[h] = {"xgb": metrics(pred[m], a[m])["median_ape"], "sum_median": metrics(base[m], a[m])["median_ape"]}
imp = final.get_score(importance_type="gain"); tot = sum(imp.values()) or 1
imp = {k: round(100 * v / tot, 1) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
log("TEST", res["test_xgb_final"], "baseline", res["test_sum_median"])

os.makedirs(OUT, exist_ok=True)
final.save_model(os.path.join(OUT, "xgb_final.json"))
meta = dict(features=FEATS, target="log(actual_s / sum_of_segment_medians)", clip=[0.5, 2.5],
            params={k: v for k, v in p.items()}, rounds=rounds, split=dict(train="days 1-18 (folds 1-3, out-of-fold)",
            validation="days 19-24 (fold 4)", final_fit="days 1-24", test="days 25-31"),
            results=res, by_length=buckets, by_hour=hours, importance_gain_pct=imp, xgboost=xgb.__version__,
            train_seconds=round(time.time() - t0))
json.dump(meta, open(os.path.join(OUT, "xgb_final_meta.json"), "w"), indent=1)

L = ["# Final XGBoost model (main submission)", "",
     "Split: train = days 1-18, validation = days 19-24 (tree count by median APE + config choice), final refit = days 1-24, "
     "test = held-out days 25-31. Nothing was tuned on test.", "",
     f"Chosen config: depth {p['max_depth']}, lr {p['learning_rate']}, min_child_weight {p['min_child_weight']}, {rounds} rounds, "
     f"objective {p['objective']}, GPU (xgboost {xgb.__version__}). Run time {round(time.time()-t0)} s.", "",
     "| Test rides (days 25-31) | Median error | Mean error | Within ±20% | Bias |", "|---|---|---|---|---|"]
for nm, r in [("Sum of segment medians", res["test_sum_median"]), ("**XGBoost final**", res["test_xgb_final"])]:
    L.append(f"| {nm} | {r['median_ape']}% | {r['mean_ape']}% | {r['within20']}% | {r['bias_s']:+.0f} s |")
L += ["", "| Ride length | XGBoost | Sum of medians |", "|---|---|---|"]
for nm, r in buckets.items():
    L.append(f"| {nm} | {r['xgb']['median_ape']}% | {r['sum_median']['median_ape']}% |")
L += ["", "Feature importance (gain %): " + ", ".join(f"{k} {v}" for k, v in list(imp.items())[:8]),
      "", "Serving: `model/v3/predict.py` with `V3_MODE=xgb` (needs xgboost in the planner venv)."]
open(os.path.join(OUT, "XGB_FINAL.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
log("wrote", OUT)

if os.environ.get("XGB_FINAL_COMMIT", "0") != "1":
    log("not committing (set XGB_FINAL_COMMIT=1 to git-commit the outputs as the original run did)")
else:
  try:
    subprocess.run(["git", "-C", REPO, "add", "model/v3/xgb_final.json", "model/v3/xgb_final_meta.json", "model/v3/XGB_FINAL.md"], check=True)
    subprocess.run(["git", "-C", REPO, "commit", "-q", "-m", "v3: final XGBoost model (train 1-18, val 19-24, refit 1-24, test 25-31)",
                    "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"], check=True)
    log("committed")
  except Exception as e:
    log("git commit failed:", e)
log("DONE")
