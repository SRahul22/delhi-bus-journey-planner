"""v2 T2.1: stop_headways_v2.csv from raw stop visits (amendment A1).
Usage: python build_headways_v2.py <days comma> <out_csv> [--gaps-out file.npz]
Gaps: per day, (stop, route) visits sorted by departure time (t_last = last ping within 60 m, so terminal layovers
count as dwell, not as a bus), revisits by the same vehicle within MERGE_S are merged, gap = difference of consecutive
departures, kept if 0 < gap <= 3 h. A gap is assigned to the IST hour of its midpoint (the hour a random arrival
inside it falls in). Daytype: weekday | weekend, holidays (HOLIDAYS env, default 02,20) pooled with weekend.
Columns: stop_id, route_id, hour, daytype, n, median_headway_s, mean_headway_s, mean_sq_headway_s,
  expected_wait_s (= mean(H^2) / (2 mean(H)), uncapped), wait_p85_s, usable_share (diagnostic: map-matched / raw trips of the
  route over the same days, from passing_index files; NaN if unknown).
wait_p85_s is the exact 85th percentile of the wait of a random arrival (gap picked with prob. proportional to length,
wait uniform within it): solves sum(max(H - w, 0)) = 0.15 * sum(H). Equals the Monte-Carlo simulation in the limit.
"""
import sys, os, time
import numpy as np, pandas as pd
W = os.path.join(os.environ.get("WORK_DIR", r"E:\hackathon-work"), "v2", "data")  # env WORK_DIR
IST = 19800; MERGE_S = 600; MAXGAP = 3 * 3600
HOLIDAYS = {int(x) for x in os.environ.get("HOLIDAYS", "02,20").split(",") if x.strip()}
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

def wait_quantile_sorted(H, gid, q=0.85):
    """Exact q-quantile of the wait of a random arrival, per group. H: gaps sorted descending within each group,
    groups contiguous (gid non-decreasing run labels). A random arrival lands in gap i with prob. H_i / sum(H) and
    waits U(0, H_i), so P(W > w) = sum(max(H - w, 0)) / sum(H); solve P(W > w) = 1 - q.
    Returns (group start offsets, quantile per group)."""
    H = np.asarray(H, float); gid = np.asarray(gid)
    start = np.r_[0, np.flatnonzero(np.diff(gid)) + 1]
    cnt = np.diff(np.r_[start, len(H)])
    cs = np.cumsum(H); S = cs - np.repeat(np.r_[0, cs[start[1:] - 1]], cnt)  # within-group cumsum S_k
    k = np.arange(len(H)) - np.repeat(start, cnt) + 1
    last = np.r_[start[1:] - 1, len(H) - 1]
    nxt = np.r_[H[1:], 0.0]; nxt[last] = 0.0
    T = (1 - q) * np.repeat(S[last], cnt)
    ok = S - k * nxt >= T - 1e-9  # f(w) = sum(max(H-w,0)) at w = next gap; first k where it reaches T
    first = pd.Series(np.where(ok, np.arange(len(H)), len(H))).groupby(np.repeat(np.arange(len(start)), cnt)).min().to_numpy()
    return start, (S[first] - T[first]) / k[first]

def day_gaps(DD):
    z = np.load(os.path.join(W, f"stop_visits_{DD}.npz"))
    st, r, v, t = z["stop"].astype(np.int64), z["route"].astype(np.int64), z["veh"], z["t_last"].astype(np.int64)
    o = np.lexsort((t, r, st)); st, r, v, t = st[o], r[o], v[o], t[o]
    same_grp = np.r_[False, (st[1:] == st[:-1]) & (r[1:] == r[:-1])]
    # merge revisits of the same vehicle (keep the later departure)
    for _ in range(3):
        dup = np.r_[(same_grp[1:]) & (v[1:] == v[:-1]) & (t[1:] - t[:-1] < MERGE_S), False]  # row i is superseded by i+1
        if not dup.any(): break
        k = ~dup; st, r, v, t = st[k], r[k], v[k], t[k]
        same_grp = np.r_[False, (st[1:] == st[:-1]) & (r[1:] == r[:-1])]
    g = (t[1:] - t[:-1]).astype(np.float64); m = same_grp[1:] & (g > 0) & (g <= MAXGAP)
    mid = (t[:-1][m] + g[m] / 2 + IST)
    lt = pd.to_datetime(mid, unit="s")
    wk = np.asarray((lt.day_of_week >= 5) | np.isin(lt.day, list(HOLIDAYS)))
    return pd.DataFrame({"stop_id": st[1:][m].astype(np.int32), "route_id": r[1:][m].astype(np.int32),
                         "hour": np.asarray(lt.hour, np.int8), "wkend": wk, "H": g[m]})

def build(days):
    G = pd.concat([day_gaps(d) for d in days], ignore_index=True)
    log("gaps", len(G))
    G["H2"] = G.H ** 2
    keys = ["stop_id", "route_id", "hour", "wkend"]
    a = G.groupby(keys).agg(n=("H", "size"), median_headway_s=("H", "median"), mean_headway_s=("H", "mean"),
                            mean_sq_headway_s=("H2", "mean"), sumH=("H", "sum")).reset_index()
    a["expected_wait_s"] = a.mean_sq_headway_s / (2 * a.mean_headway_s)
    # exact p85 wait
    G = G.sort_values(keys + ["H"], ascending=[True] * 4 + [False], kind="stable").reset_index(drop=True)
    gid = G.groupby(keys, sort=False).ngroup().to_numpy()
    start, w85 = wait_quantile_sorted(G.H.to_numpy(), gid, 0.85)
    gk = G.iloc[start][keys].reset_index(drop=True); gk["wait_p85_s"] = w85
    a = a.merge(gk, on=keys)
    a["daytype"] = np.where(a.wkend, "weekend", "weekday")
    # usable share (diagnostic)
    mt, rt = {}, {}
    for d in days:
        fn = os.path.join(W, f"passing_index_{d}.npz")
        if not os.path.exists(fn): continue
        z = np.load(fn)
        for rr, mm, raw in zip(z["us_route"], z["us_matched"], z["us_raw"]):
            if raw > 0: mt[rr] = mt.get(rr, 0) + mm; rt[rr] = rt.get(rr, 0) + raw
    us = pd.Series({k_: mt[k_] / rt[k_] for k_ in rt}, dtype=float)
    a["usable_share"] = a.route_id.map(us).round(3)
    for c in ("median_headway_s", "mean_headway_s", "expected_wait_s", "wait_p85_s"): a[c] = a[c].round(1)
    a["mean_sq_headway_s"] = a.mean_sq_headway_s.round(0)
    return a[["stop_id", "route_id", "hour", "daytype", "n", "median_headway_s", "mean_headway_s", "mean_sq_headway_s",
              "expected_wait_s", "wait_p85_s", "usable_share"]]

if __name__ == "__main__":
    days = [d.zfill(2) for d in sys.argv[1].split(",")]
    out = build(days)
    out.to_csv(sys.argv[2], index=False)
    log("rows", len(out), "n>=20 rows", int((out.n >= 20).sum()), "->", sys.argv[2])
