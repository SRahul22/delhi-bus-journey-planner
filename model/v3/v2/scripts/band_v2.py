"""v2 ml: T0.4 (tree count every 10), T1.1 (quantile model), T1.2 (out-of-fold conformal shifts, amendment A2).

Data: E:\\hackathon-work\\v3\\fold1..4.parquet (rides of days 1-6, 7-12, 13-18, 19-24; features of each fold come
from lookup tables built on the other 18 days). test.parquet is NOT read.

Split (AGENTS.md section 3): ML train = days 1-18 (folds 1-3), validation = days 19-24 (fold 4).
  T0.4  L1 XGBoost (same 3 configs as xgb_final.py), tree count checked every 10 trees on fold 4 by median APE.
  T1.1  reg:quantileerror, quantile_alpha=[0.5, 0.85], same 20 features, target log(actual/base_med), early stopping
        on fold-4 pinball loss. Final model refit on days 1-24 with the same number of rounds.
  T1.2  scores s = log(actual/base) - q85 from OUT-OF-FOLD predictions (A2):
          shifts_24: 4 models, each trained on 3 of the 4 folds, predicting the 4th   -> served with the 1-24 model
          shifts_18: 3 models, each trained on 2 of folds 1-3, predicting the 3rd     -> clean validation check:
                     1-18 model + shifts_18, coverage measured on fold 4 (days 19-24)
        per bucket x peak: shift = ceil((n+1)*0.85)-th smallest score; cells with n < 500 pooled with a neighbour.
        Bucket = predicted p50 (base_med * exp(q50 of the quantile model)), so serving can key it without the actual.
Run (detached):
  python E:\\hackathon-work\\launch.py E:\\hackathon-work\\v2\\ml\\band_log.txt E:\\hackathon-wt\\ml\\model\\v3\\v2\\scripts\\band_v2.py
"""
import json, math, os, shutil, time
import numpy as np, pandas as pd, xgboost as xgb

D = r"E:\hackathon-work\v3"
OUT = r"E:\hackathon-work\v2\ml"
REPO_OUT = r"E:\hackathon-wt\ml\model\v3\v2"
os.makedirs(OUT, exist_ok=True); os.makedirs(REPO_OUT, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
log("xgboost", xgb.__version__)

FEATS = ["log_base_med", "skew", "n_stops", "log_dist", "base_speed", "hour_f", "weekend", "route_len_m", "route_nstops",
         "route_speed", "log_route_obs", "route_cov", "route_skew", "board_pos", "b_lat", "b_lon", "a_lat", "a_lon",
         "fb_frac", "mean_logn"]
HOLIDAYS = {2, 20}                       # Oct 2026: Gandhi Jayanti, Dussehra (section 4)
BUCKETS = [(0, 600, "<10"), (600, 1800, "10-30"), (1800, 3600, "30-60"), (3600, 1e12, ">60")]
ALPHA = 0.85
MIN_CELL = 500


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


tr = pd.concat([pd.read_parquet(os.path.join(D, f"fold{i}.parquet")).assign(fold=i) for i in range(1, 5)], ignore_index=True)
log("rides days 1-24:", len(tr), tr.groupby("fold").day.agg(["min", "max", "size"]).to_dict("index"))
X = feats(tr).to_numpy(np.float32)
y = np.log(tr.actual_s / tr.base_med).to_numpy()
act = tr.actual_s.to_numpy(); base = tr.base_med.to_numpy(); fold = tr.fold.to_numpy()
hour = tr.hour_f.to_numpy()
weekday = (tr.dow.to_numpy() < 5) & ~np.isin(tr.day.to_numpy(), list(HOLIDAYS))
peak = weekday & (((hour >= 8) & (hour < 11)) | ((hour >= 17) & (hour < 21)))
val = fold == 4; trn = ~val


def bucket_of(sec):
    b = np.zeros(len(sec), np.int64)
    for k, (lo, hi, _) in enumerate(BUCKETS):
        b[(sec >= lo) & (sec < hi)] = k
    return b


def ape_metrics(p, a):
    e = p - a; ape = np.abs(e) / a
    return dict(median_ape=round(100 * float(np.median(ape)), 3), within20=round(100 * float(np.mean(ape < 0.2)), 2),
                bias_s=round(float(np.mean(e)), 1), n=int(len(a)))


def pinball(q, a, alpha=ALPHA):
    d = a - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


# ------------------------------------------------------------------ T0.4: tree count every 10 (L1 model, p50)
base_cfg = dict(objective="reg:absoluteerror", device="cuda", tree_method="hist", eval_metric="mae",
                subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, seed=0)
grid = [dict(learning_rate=0.05, max_depth=7, min_child_weight=200),
        dict(learning_rate=0.05, max_depth=9, min_child_weight=100),
        dict(learning_rate=0.03, max_depth=10, min_child_weight=50)]
dtr = xgb.DMatrix(X[trn], y[trn]); dva = xgb.DMatrix(X[val], y[val])
MAXR = 1000
curve, best, best100 = [], None, None
for g in grid:
    p = {**base_cfg, **g}
    b = xgb.train(p, dtr, MAXR, verbose_eval=False)
    for r in range(10, MAXR + 1, 10):
        mv = ape_metrics(base[val] * np.exp(b.predict(dva, iteration_range=(0, r))), act[val])
        curve.append(dict(cfg=g, rounds=r, **mv))
        if best is None or mv["median_ape"] < best[2]["median_ape"]:
            best = (p, r, mv, b)
        if r % 100 == 0 and (best100 is None or mv["median_ape"] < best100[2]["median_ape"]):
            best100 = (p, r, mv)
    log("T0.4 config", g, "best so far", best[1], best[2])
p50cfg, p50_rounds, p50_val, m18_p50 = best
log("T0.4 chosen (every 10):", p50cfg, p50_rounds, p50_val, "| every-100 pick:", best100[1], best100[2])
p50_18_val = base[val] * np.clip(np.exp(m18_p50.predict(dva, iteration_range=(0, p50_rounds))), 0.5, 2.5)
pd.DataFrame(curve).assign(cfg=lambda d: d.cfg.astype(str)).to_csv(os.path.join(OUT, "t04_tree_curve.csv"), index=False)
fin_p50 = xgb.train(p50cfg, xgb.DMatrix(X, y), p50_rounds)
fin_p50.save_model(os.path.join(OUT, "xgb_final_r10.json"))
t04 = dict(chosen_every10=dict(cfg={k: v for k, v in p50cfg.items()}, rounds=p50_rounds, val=p50_val),
           chosen_every100=dict(cfg=best100[0], rounds=best100[1], val=best100[2]),
           v1_served=dict(rounds=100, note="xgb_final.json: depth 10, lr 0.03, mcw 50, 100 rounds (grid every 100)"),
           val_sum_median=ape_metrics(base[val], act[val]))
json.dump(t04, open(os.path.join(OUT, "t04_trees.json"), "w"), indent=1, default=str)
log("T0.4 written")

# ------------------------------------------------------------------ T1.1: quantile model
qcfg = dict(objective="reg:quantileerror", quantile_alpha=np.array([0.5, ALPHA]), device="cuda", tree_method="hist",
            eval_metric="quantile", learning_rate=0.05, max_depth=8, min_child_weight=100,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, seed=0)
ev = {}
m18_q = xgb.train(qcfg, dtr, 3000, evals=[(dva, "val")], early_stopping_rounds=50, evals_result=ev, verbose_eval=False)
q_rounds = m18_q.best_iteration + 1
log("T1.1 quantile: early-stopped at", q_rounds, "val pinball(avg of 0.5,0.85, log space)", round(float(m18_q.best_score), 5))
Q18 = m18_q.predict(dva, iteration_range=(0, q_rounds))           # (n, 2) log-ratio quantiles


def fit_q(mask):
    return xgb.train(qcfg, xgb.DMatrix(X[mask], y[mask]), q_rounds)


fin_q = fit_q(np.ones(len(y), bool))
fin_q.save_model(os.path.join(OUT, "xgb_quantile.json"))
log("T1.1 final quantile model (days 1-24) saved")

# ------------------------------------------------------------------ T1.2: out-of-fold scores (A2)
oof24 = np.full((len(y), 2), np.nan)
for k in range(1, 5):
    m = fit_q(fold != k); te = fold == k
    oof24[te] = m.predict(xgb.DMatrix(X[te]))
    log("oof24 fold", k)
oof18 = np.full((len(y), 2), np.nan)
for k in range(1, 4):
    m = fit_q((fold != k) & trn); te = fold == k
    oof18[te] = m.predict(xgb.DMatrix(X[te]))
    log("oof18 fold", k)


def cell_keys(q50log, b_, pk):
    return bucket_of(b_ * np.exp(q50log)), pk.astype(np.int64)


def conformal(scores, bk, pk):
    """shift per bucket|peak; cells with n < MIN_CELL pooled with the neighbouring bucket (same peak flag)."""
    out, ns = {}, {}
    for pf in (0, 1):
        groups = [[k] for k in range(len(BUCKETS))]
        cnt = lambda gr: int(np.sum(np.isin(bk, gr) & (pk == pf)))
        changed = True
        while changed and len(groups) > 1:
            changed = False
            for i, gr in enumerate(groups):
                if cnt(gr) < MIN_CELL:
                    j = i - 1 if i == len(groups) - 1 else i + 1
                    groups[min(i, j)] = sorted(groups[i] + groups[j]); del groups[max(i, j)]
                    changed = True; break
        for gr in groups:
            s = np.sort(scores[np.isin(bk, gr) & (pk == pf)])
            n = len(s)
            sh = float(s[min(n, math.ceil((n + 1) * ALPHA)) - 1]) if n else 0.0
            for k in gr:
                key = f"{BUCKETS[k][2]}|{'peak' if pf else 'offpeak'}"
                out[key] = round(sh, 5); ns[key] = n
    return out, ns


def apply_shift(shifts, bk, pk):
    return np.array([shifts[f"{BUCKETS[b][2]}|{'peak' if p else 'offpeak'}"] for b, p in zip(bk, pk)])


s24 = y - oof24[:, 1]
bk24, pk24 = cell_keys(oof24[:, 0], base, peak)
shifts24, n24 = conformal(s24, bk24, pk24)
s18 = y[trn] - oof18[trn, 1]
bk18, pk18 = cell_keys(oof18[trn, 0], base[trn], peak[trn])
shifts18, n18 = conformal(s18, bk18, pk18)
# fallback for legs predict.py cannot locate: conformal quantile of log(actual / caller's base median), bucket by base
fb, fbn = conformal(y, bucket_of(base), peak.astype(np.int64))
log("shifts24", shifts24); log("shifts18", shifts18)

# ------------------------------------------------------------------ validation on days 19-24 (1-18 models)
a_v = act[val]; b_v = base[val]; pk_v = peak[val]
bkv, pkv = cell_keys(Q18[:, 0], b_v, pk_v)


def band_metrics(p50, p85, a, mask=None):
    if mask is not None:
        p50, p85, a = p50[mask], p85[mask], a[mask]
    if len(a) == 0:
        return {}
    return dict(n=int(len(a)), coverage85=round(100 * float(np.mean(a <= p85)), 2),
                under_p50=round(100 * float(np.mean(a <= p50)), 2),
                width_med=round(float(np.median((p85 - p50) / p50)), 4),
                pinball85_s=round(pinball(p85, a), 1), p50_median_ape=round(100 * float(np.median(np.abs(p50 - a) / a)), 3))


variants = {}
p85_raw = np.maximum(b_v * np.exp(Q18[:, 1]), p50_18_val)
p85_c18 = np.maximum(b_v * np.exp(Q18[:, 1] + apply_shift(shifts18, bkv, pkv)), p50_18_val)
p85_c24 = np.maximum(b_v * np.exp(Q18[:, 1] + apply_shift(shifts24, bkv, pkv)), p50_18_val)
v1_p85 = np.maximum(tr.base_p85.to_numpy()[val], b_v)
variants["v1_sum_segment_p85 (p50=sum of medians)"] = (b_v, v1_p85)
variants["quantile_raw (no shift)"] = (p50_18_val, p85_raw)
variants["quantile+shifts18 (clean)"] = (p50_18_val, p85_c18)
variants["quantile+shifts24 (served shifts; fold 4 in calibration)"] = (p50_18_val, p85_c24)
res = {}
for nm, (p50v, p85v) in variants.items():
    r = {"all": band_metrics(p50v, p85v, a_v)}
    for k, (_, _, bn) in enumerate(BUCKETS):          # gate buckets are by ACTUAL ride length (section 4)
        for pf in (0, 1):
            m = (bucket_of(a_v) == k) & (pk_v == pf)
            r[f"actual {bn}|{'peak' if pf else 'offpeak'}"] = band_metrics(p50v, p85v, a_v, m)
        r[f"actual {bn}"] = band_metrics(p50v, p85v, a_v, bucket_of(a_v) == k)
        r[f"pred {bn}"] = band_metrics(p50v, p85v, a_v, bkv == k)
    dv = tr.day.to_numpy()[val]
    for dd in range(19, 25):
        r[f"day {dd}"] = band_metrics(p50v, p85v, a_v, dv == dd)
    r["days 21-24"] = band_metrics(p50v, p85v, a_v, dv >= 21)
    res[nm] = r
    log("VAL", nm, r["all"], "| days 21-24", r["days 21-24"])

meta = dict(version="v2-band-1", features=FEATS, target="log(actual_s / sum_of_segment_medians)",
            quantile_alpha=[0.5, ALPHA], quantile_rounds=q_rounds, quantile_params={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in qcfg.items()},
            serve="p85 = max(p50, base_med * exp(q85 + shift[bucket(base_med*exp(q50))|peak]))",
            buckets_s=[[lo, hi, nm] for lo, hi, nm in BUCKETS], peak="weekday (not 2 or 20 Oct) and hour in [8,11) or [17,21)",
            holidays_oct=sorted(HOLIDAYS), min_cell=MIN_CELL, alpha=ALPHA,
            shifts=shifts24, n=n24, shifts_18=shifts18, n_18=n18, fallback_log_ratio=fb, fallback_n=fbn,
            fallback="legs not in v3 patterns: p85 = max(p50, base_median_s * exp(fallback_log_ratio[bucket(base)|peak]))",
            scores="out-of-fold (A2): shifts from 4 models each trained on 3 of 4 six-day folds of days 1-24")
json.dump(meta, open(os.path.join(OUT, "conformal_shifts.json"), "w"), indent=1)
json.dump(dict(t04=t04, quantile_val_best_score=float(m18_q.best_score), val=res), open(os.path.join(OUT, "band_val.json"), "w"), indent=1, default=str)
for f in ("xgb_quantile.json", "conformal_shifts.json", "band_val.json", "t04_trees.json"):
    shutil.copy(os.path.join(OUT, f), os.path.join(REPO_OUT, f))
pd.DataFrame(dict(day=tr.day.to_numpy()[val], actual_s=a_v, base_med=b_v, peak=pk_v, p50_l1_18=p50_18_val,
                  q50_18=Q18[:, 0], q85_18=Q18[:, 1], p85_c18=p85_c18, v1_p85=v1_p85)).to_parquet(os.path.join(OUT, "band_val_preds.parquet"))
log("DONE")
