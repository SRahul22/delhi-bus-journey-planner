"""v2 live-layer evaluation on one replay day (T3.3 baselines, T3.1 KF fit, T3.4 sd_s table).

Usage (Windows, planner venv, MODEL_DIR=E:\\hackathon-project\\model):
    python realtime/v2_live_eval.py 22 --fit      # fit KF (tau_s, tau_c, q) + next-bus sd_s table on 22 Oct ONLY
    python realtime/v2_live_eval.py 28            # evaluate with the 22-Oct parameters (no tuning)
    python realtime/v2_live_eval.py 30

Same protocol as realtime/validate.py (v1, the "107 s" next-bus metric): replay 06:00-22:00 in 5-min steps,
evaluation times every 30 min 07:00-21:00, 400 random (route, stop) pairs with a live bus approaching per
evaluation time, actual wait = next observed passing - T. Predictors:
    half_headway  historical headway / 2 (v1 planner default)
    gtfs_sched    next GTFS scheduled arrival of that route at that stop after T
    live_ratio    v1 live ETA (V2_LIVE=ratio)
    live_kf       live ETA with Kalman horizon forecasts (V2_LIVE=kf)
Rides: random board/alight pairs from the day's projected passings; historical vs ratio vs kf, lead 0-30 and
30-60 min, and the 07:30-08:30 boarding window.
Segment forecasts: for snapshots every 10 min, every later segment traversal starting within 60 min:
weighted MSE of log(observed/historical) for ratio vs kf (this is the KF fitting objective on 22 Oct).

Outputs (E:\\hackathon-work\\v2\\realtime\\, override RT_V2_OUT): results_DD.json; with --fit also
kf_params.json, kf_grid_22.csv, next_bus_sd.json.
"""
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(REPO, "planner"), HERE]

import kf as kfmod  # noqa: E402
from feed import ReplayFeed, parse_ist  # noqa: E402
from live_state import LiveState, Snapshot, sod  # noqa: E402
from rt_model import RealtimeTravelTimeModel  # noqa: E402
from ttmodel import TravelTimeModel  # noqa: E402
from paths import gtfs_dir  # noqa: E402

OUT = os.environ.get("RT_V2_OUT", r"E:\hackathon-work\v2\realtime")
H_BUCKETS = [0, 300, 600, 1200, 1800, 3600]          # next-bus ETA buckets for sd_s (s)
SEG_HB = [0, 600, 1200, 1800, 2700, 3600]           # segment-forecast horizon buckets (s)
GRID = dict(tau_s=[300, 900, 1800, 3600], tau_c=[1800, 3600, 7200, 14400], q=[2e-6, 2e-5, 1e-4])


def find(*c):
    for p in c:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(c)


def errstats(pred, act):
    e = pred - act
    a = np.abs(e)
    return dict(n=int(len(a)), mean_abs_err_s=round(float(a.mean()), 1), median_abs_err_s=round(float(np.median(a)), 1),
                within_2min_pct=round(float((a <= 120).mean()) * 100, 1), median_bias_s=round(float(np.median(e)), 1))


def ride_stats(r):
    ape = np.abs(r[:, 1] - r[:, 0]) / r[:, 0]
    return dict(n=int(len(r)), median_ape=round(float(np.median(ape)) * 100, 2), within_20pct=round(float((ape <= .2).mean()) * 100, 1),
                median_bias_pct=round(float(np.median((r[:, 1] - r[:, 0]) / r[:, 0])) * 100, 2))


def gtfs_next(net, queries):
    """{(ri, k): sorted scheduled arrival sod array} for the queried (route, stop) pairs."""
    stop_ids = net["stops"].stop_id.astype(str).to_numpy()
    want = {}
    for _, ri, k in queries:
        r = net["routes"][ri]
        want.setdefault((str(r["route_id"]), stop_ids[r["stops"][k]]), []).append((ri, k))
    g = gtfs_dir()
    trips = pd.read_csv(os.path.join(g, "trips.txt"), usecols=["route_id", "trip_id"], dtype=str)
    routes = {a for a, _ in want}
    trips = trips[trips.route_id.isin(routes)]
    st = pd.read_csv(os.path.join(g, "stop_times.txt"), usecols=["trip_id", "arrival_time", "stop_id"], dtype=str)
    st = st.merge(trips, on="trip_id")
    st["key"] = st.route_id + "|" + st.stop_id
    keys = {f"{a}|{b}" for a, b in want}
    st = st[st.key.isin(keys)]
    hms = st.arrival_time.str.split(":", expand=True).astype(float)
    st["s"] = hms[0] * 3600 + hms[1] * 60 + hms[2]
    out = {}
    for key, grp in st.groupby("key"):
        a, b = key.split("|")
        arr = np.sort(grp.s.to_numpy())
        for rk in want[(a, b)]:
            out[rk] = arr
    return out


def main():
    t00 = time.time()
    DD = sys.argv[1]
    fit = "--fit" in sys.argv
    if fit and DD != "22":
        sys.exit("fitting is allowed on 22 Oct only")
    DAY = f"2026-10-{DD}"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    net = pickle.load(open(find(os.environ.get("RT_NETWORK"), os.path.join(REPO, "planner", "cache", "network.pkl"),
                                os.path.join(OUT, "network.pkl")), "rb"))
    replay = find(os.path.join(OUT, f"replay_{DD}.npz"), rf"E:\hackathon-work\realtime\replay_{DD}.npz")
    base = TravelTimeModel(net)
    ls = LiveState(net, base, mode="ratio")
    ls.keep_history = True
    rt = RealtimeTravelTimeModel(base, ls)
    feed = ReplayFeed(replay)
    start, end = parse_ist(f"{DAY}T06:00"), parse_ist(f"{DAY}T22:00")
    evals = [parse_ist(f"{DAY}T07:00") + 1800 * i for i in range(29)]
    nobs_at, veh_at, wait_q = {}, {}, []
    t = start
    while t < end:
        ls.ingest(feed.window(t, t + 300))
        t += 300
        nobs_at[t] = ls.stats["obs"]
        if t in evals:
            snap = ls.refresh(t)
            veh_at[t] = snap.vehicles
            cand = [(ri, k) for ri, vs in snap.vehicles.items() for k in range(1, len(ls.G[int(net["routes"][ri]["route_id"])][8]))
                    if any(nk <= k for _, _, nk, _, _ in vs)]
            for ri, k in [cand[i] for i in rng.choice(len(cand), min(400, len(cand)), replace=False)]:
                wait_q.append((t, ri, k))
    print(f"replayed in {time.time()-t00:.0f}s", ls.stats, flush=True)
    obs = ls.observations()
    sigma = kfmod.seg_sigma(base)

    # ---------------------------------------------------------------- segment forecast pairs (10-min snapshots)
    fit_times = [parse_ist(f"{DAY}T07:00") + 600 * i for i in range(85)]  # 07:00 .. 21:00
    t1o = obs["t1"]; t0o = obs["t0"]
    PT, PF, PH, PZ, PW = [], [], [], [], []
    for ti, T in enumerate(fit_times):
        n = nobs_at[T]
        idx = np.flatnonzero((t0o > T) & (t0o <= T + 3600))
        idx = idx[idx >= n]
        idx = idx if len(idx) <= 4000 else np.sort(rng.choice(idx, 4000, replace=False))
        PT.append(np.full(len(idx), ti)); PF.append(obs["f"][idx]); PH.append(t0o[idx] - T); PZ.append(obs["logr"][idx]); PW.append(obs["w"][idx])
    PT, PF, PH, PZ, PW = map(np.concatenate, (PT, PF, PH, PZ, PW))
    ratio_logr = {T: ls.compute(T, n_max=nobs_at[T])[0] for T in sorted(set(fit_times) | set(evals))}
    wfade = np.vectorize(RealtimeTravelTimeModel.weight)(PH)
    pred_ratio = np.array([ratio_logr[fit_times[ti]][f] for ti, f in zip(PT, PF)]) * wfade

    def seg_scores(pred, var=None):
        e = PZ - pred
        out = dict(wmse=round(float((PW * e * e).sum() / PW.sum()), 5), n=int(len(e)))
        hb = np.digitize(PH, SEG_HB) - 1
        out["wmse_by_h"] = {f"{SEG_HB[b]}-{SEG_HB[b+1]}": round(float((PW[hb == b] * e[hb == b] ** 2).sum() / PW[hb == b].sum()), 5)
                            for b in range(len(SEG_HB) - 1) if (hb == b).any()}
        if var is not None:
            out["mean_var"] = round(float(var.mean()), 5)
        return out

    def run_kf(params, times):
        k = kfmod.SegmentKF(ls.n_flat, ls.gx, ls.gy, sigma, **params)
        st = {}
        for T in sorted(times):
            k.advance(obs, T, n_max=nobs_at[T])
            st[T] = k.state()
        return st

    def kf_pred(states):
        x = np.empty(len(PF)); v = np.empty(len(PF))
        for ti, T in enumerate(fit_times):
            m = PT == ti
            x[m], v[m] = kfmod.forecast(states[T], PH[m], PF[m])
        return x, v

    res = {"replay_day": DAY, "replay": replay, "model_source": base.source, "stats": ls.stats}
    if fit:
        rows = []
        for ts_, tc_, q_ in itertools.product(GRID["tau_s"], GRID["tau_c"], GRID["q"]):
            p = dict(tau_s=ts_, tau_c=tc_, q=q_)
            x, v = kf_pred(run_kf(p, fit_times))
            e = PZ - x
            # Gaussian NLL of the observation: var = P(h) + R(segment, hour)
            rows.append(dict(p, wmse=float((PW * e * e).sum() / PW.sum())))
            print(rows[-1], flush=True)
        grid = pd.DataFrame(rows).sort_values("wmse")
        grid.to_csv(os.path.join(OUT, "kf_grid_22.csv"), index=False)
        best = {k: float(grid.iloc[0][k]) for k in ("tau_s", "tau_c", "q")}
        params = dict(kfmod.DEFAULTS); params.update(best)
        with open(os.path.join(OUT, "kf_params.json"), "w") as f:
            json.dump(dict(fitted_on="2026-10-22", objective="weighted MSE of segment log-ratio forecasts, horizons 0-60 min, "
                           "snapshots every 10 min 07:00-21:00 (weights = min(hist,300)/60)", params=params,
                           best_wmse=float(grid.iloc[0].wmse)), f, indent=1)
        res["kf_grid_top5"] = grid.head(5).to_dict("records")
    kfp = os.path.join(OUT, "kf_params.json")
    while not fit and not os.path.exists(kfp):   # test days wait for the 22-Oct fit (run in parallel)
        print("waiting for", kfp, flush=True)
        time.sleep(30)
    params = kfmod.load_params(kfp)
    res["kf_params"] = params
    kf_states = run_kf(params, set(fit_times) | set(evals))
    x_kf, v_kf = kf_pred(kf_states)
    res["segment_forecast"] = dict(zero=seg_scores(np.zeros(len(PZ))), ratio=seg_scores(pred_ratio), kf=seg_scores(x_kf, v_kf))
    print("segment forecast", res["segment_forecast"], flush=True)

    # ---------------------------------------------------------------- truth from passings
    P = ls.passings()
    order = np.lexsort((P["t"], P["trk"]))
    ri_, k_, t_, trk = P["ri"][order], P["k"][order], P["t"][order], P["trk"][order]
    cuts = np.flatnonzero(np.diff(trk)) + 1
    rides = []
    for a, b in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(trk)]])):
        if b - a < 4:
            continue
        for _ in range(3):
            i, j = sorted(rng.choice(b - a, 2, replace=False))
            if k_[a + j] - k_[a + i] < 3:
                continue
            dur = t_[a + j] - t_[a + i]
            if dur < 120 or dur > 3 * 3600:
                continue
            rides.append((int(ri_[a]), int(k_[a + i]), int(k_[a + j]), float(t_[a + i]), float(dur)))
    rides = np.array(rides)
    pk = P["ri"].astype(np.int64) * 1000 + P["k"]
    o2 = np.lexsort((P["t"], pk)); pk_s, pt_s = pk[o2], P["t"][o2]

    def next_pass(ri, k, T):
        key = ri * 1000 + k
        lo, hi = np.searchsorted(pk_s, key, "left"), np.searchsorted(pk_s, key, "right")
        j = lo + np.searchsorted(pt_s[lo:hi], T, "right")
        return pt_s[j] if j < hi else None

    def snap_for(T, mode):
        logr = ratio_logr[T] if mode == "ratio" else kfmod.forecast(kf_states[T], 0.0)[0].astype(np.float32)
        return Snapshot(T, logr, None, None, (1.0, 0.0), 0, veh_at.get(T, {}), kf=kf_states[T] if mode == "kf" else None)

    # ---------------------------------------------------------------- rides
    for lead_name, lo, hi in (("lead_0_30min", 0, 1800), ("lead_30_60min", 1800, 3600)):
        rows = []
        for T in evals:
            m = (rides[:, 3] > T + lo) & (rides[:, 3] <= T + hi)
            preds = {}
            for mode in ("ratio", "kf"):
                ls.snapshot = snap_for(T, mode)
                preds[mode] = [rt.ride(int(ri), int(i), int(j), sod(tb), 0)[0] for ri, i, j, tb, dur in rides[m]]
            for n_, (ri, i, j, tb, dur) in enumerate(rides[m]):
                rows.append((T, tb, dur, base.ride(int(ri), int(i), int(j), sod(tb), 0)[0], preds["ratio"][n_], preds["kf"][n_]))
        r = np.array(rows)
        d = {}
        for name, col in (("historical", 3), ("live_ratio", 4), ("live_kf", 5)):
            d[name] = ride_stats(r[:, [2, col]])
            sb = (r[:, 1] + 19800) % 86400
            w = (sb > 7.5 * 3600) & (sb <= 8.5 * 3600)
            if w.any():
                d[name + "_board_0730_0830"] = ride_stats(r[w][:, [2, col]])
        res[lead_name] = d
        print(lead_name, d, flush=True)

    # ---------------------------------------------------------------- next bus
    sched = gtfs_next(net, wait_q)
    W = []
    for T, ri, k in wait_q:
        nxt = next_pass(ri, k, T)
        if nxt is None or nxt - T > 3600:
            continue
        etas = []
        for mode in ("ratio", "kf"):
            arr = ls.arrivals(ri, k, snap_for(T, mode))
            etas.append(arr[0][0] - T if arr else np.nan)
        s = sched.get((ri, k))
        g = np.nan
        if s is not None and len(s):
            j = np.searchsorted(s, sod(T), "right")
            if j < len(s) and s[j] - sod(T) <= 3 * 3600:
                g = s[j] - sod(T)
        W.append((T, nxt - T, base.wait_s(ri, k, sod(T), 0), g, etas[0], etas[1]))
    W = np.array(W)
    ok = ~np.isnan(W[:, 4]) & ~np.isnan(W[:, 5])
    Wa = W[ok]
    nb = dict(n=int(len(Wa)), actual_median_s=round(float(np.median(Wa[:, 1]))),
              half_headway=errstats(Wa[:, 2], Wa[:, 1]), live_ratio=errstats(Wa[:, 4], Wa[:, 1]), live_kf=errstats(Wa[:, 5], Wa[:, 1]))
    g = ~np.isnan(Wa[:, 3])
    nb["gtfs_sched_subset"] = dict(coverage_pct=round(float(g.mean()) * 100, 1), gtfs_sched=errstats(Wa[g][:, 3], Wa[g][:, 1]),
                                   live_ratio=errstats(Wa[g][:, 4], Wa[g][:, 1]), live_kf=errstats(Wa[g][:, 5], Wa[g][:, 1]),
                                   half_headway=errstats(Wa[g][:, 2], Wa[g][:, 1]))
    res["next_bus"] = nb
    print("next bus", json.dumps(nb), flush=True)

    # ---------------------------------------------------------------- sd_s table (22 Oct only)
    if fit:
        tab = {}
        for name, col in (("ratio", 4), ("kf", 5)):
            eta, act = Wa[:, col], Wa[:, 1]
            b = np.digitize(eta, H_BUCKETS) - 1
            rows = []
            for i in range(len(H_BUCKETS) - 1):
                m = b == i
                if m.sum() < 20:
                    continue
                e = act[m] - eta[m]
                rows.append(dict(lo_s=H_BUCKETS[i], hi_s=H_BUCKETS[i + 1], n=int(m.sum()), median_resid_s=round(float(np.median(e)), 1),
                                 sd_s=round(float(1.4826 * np.median(np.abs(e - np.median(e)))), 1), std_s=round(float(e.std()), 1),
                                 p15_s=round(float(np.percentile(e, 15)), 1), p85_s=round(float(np.percentile(e, 85)), 1)))
            tab[name] = rows
        # predictive variance of segment log ratios by horizon (for segment_state in ratio mode)
        hb = np.digitize(PH, SEG_HB) - 1
        R = sigma[PF, 0, 7].astype(np.float64) ** 2  # rough measurement noise (weekday 07h); subtracted below
        segvar = []
        for i in range(len(SEG_HB) - 1):
            m = hb == i
            if m.any():
                e = PZ[m] - pred_ratio[m]
                segvar.append(dict(lo_s=SEG_HB[i], hi_s=SEG_HB[i + 1], n=int(m.sum()),
                                   resid_var=round(float(np.average(e * e, weights=PW[m])), 5),
                                   median_R=round(float(np.median(R[m])), 5)))
        with open(os.path.join(OUT, "next_bus_sd.json"), "w") as f:
            json.dump(dict(fitted_on="2026-10-22", definition="residual = actual wait - predicted ETA of the first live bus; "
                           "sd_s = 1.4826*MAD within the ETA bucket [lo_s, hi_s)", next_bus=tab, ratio_segment_var=segvar), f, indent=1)
        res["next_bus_sd"] = tab
    res["runtime_s"] = round(time.time() - t00)
    with open(os.path.join(OUT, f"results_{DD}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("done", res["runtime_s"], "s")


if __name__ == "__main__":
    main()
