"""Aggregate per-day passing files into model tables.
Usage: python aggregate.py <train_days comma> <holdout_days comma or -> <outdir>
"""
import sys, os, glob, json
import numpy as np, pandas as pd

W = r"E:\hackathon-work"
train = [d for d in sys.argv[1].split(",") if d]
hold = [d for d in sys.argv[2].split(",") if d and d != "-"]
OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
IST = 19800

pat = pd.read_csv(os.path.join(W, "route_patterns.csv"))
dist_lookup = {(r, s): d for r, s, d in zip(pat.route_id, pat.stop_sequence, pat.dist_m)}

def load(days):
    dfs = []
    for i, d in enumerate(days):
        fn = os.path.join(W, "pass", f"pass_{d}.npz")
        if not os.path.exists(fn) or os.path.getsize(fn) < 1000:
            print("missing", fn); continue
        z = np.load(fn)
        df = pd.DataFrame({k: z[k] for k in z.files})
        df["trip"] = df["trip"].astype(np.int64) + i * 10_000_000
        df["day"] = int(d)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    pk = pat.set_index(["route_id", "stop_sequence"]).dist_m
    df["dist"] = pk.reindex(pd.MultiIndex.from_arrays([df.route, df.seq])).to_numpy()
    lt = pd.to_datetime(df.t + IST, unit="s")
    df["hour"] = lt.dt.hour.astype(np.int8)
    df["daytype"] = np.where(lt.dt.dayofweek >= 5, "weekend", "weekday")
    return df.sort_values(["trip", "seq"], kind="stable").reset_index(drop=True)

def segments(df):
    nx = df.shift(-1)
    m = (nx.trip == df.trip) & (nx.seq == df.seq + 1)
    s = pd.DataFrame({"route_id": df.route[m], "from_stop": df.stop[m], "to_stop": nx.stop[m].astype(np.int64),
                      "hour": df.hour[m], "daytype": df.daytype[m],
                      "dur": (nx.t[m] - df.t[m]), "dd": (nx.dist[m] - df.dist[m])})
    good = (s.dur >= 0) & ((s.dur > 0) | (s.dd < 30)) & (s.dd <= 19.4 * s.dur + 20) & (s.dur <= np.maximum(1800, s.dd / 1.0))
    print("segment obs", len(s), "kept", int(good.sum()))
    return s[good]

stats = {}
if train:
    df = load(train)
    print("train passings", len(df))
    s = segments(df)
    g = s.groupby(["route_id", "from_stop", "to_stop", "hour", "daytype"]).dur
    seg = pd.concat([g.median().rename("median_s"), g.quantile(0.85).rename("p85_s"), g.size().rename("n"), g.mean().rename("mean_s")], axis=1).reset_index()
    # mean_s (extra column): sums of medians under-predict long rides because segment times are right-skewed
    seg["median_s"] = seg.median_s.round(1); seg["p85_s"] = seg.p85_s.round(1); seg["mean_s"] = seg.mean_s.round(1)
    seg.to_csv(os.path.join(OUT, "segment_times.csv"), index=False)
    gtfs_segs = len(pat) - pat.route_id.nunique()
    cov = seg[["route_id", "from_stop", "to_stop"]].drop_duplicates()
    stats.update(train_days=train, passings=len(df), segment_obs=len(s), segment_rows=len(seg),
                 gtfs_segments=gtfs_segs, segments_covered=len(cov), seg_cov_pct=round(100 * len(cov) / gtfs_segs, 1),
                 routes_covered=int(seg.route_id.nunique()), routes_total=int(pat.route_id.nunique()),
                 seg_n_ge5=int(((s.groupby(["route_id", "from_stop", "to_stop"]).size()) >= 5).sum()),
                 median_seg_s=float(s.dur.median()), median_seg_speed_kmh=float((s.dd.sum() / s.dur.sum()) * 3.6))
    # headways
    h = df[["route", "stop", "seq", "veh", "t", "day", "hour", "daytype"]].sort_values(["day", "route", "seq", "t"])
    prev = h.shift(1)
    same = (prev.day == h.day) & (prev.route == h.route) & (prev.seq == h.seq) & (prev.veh != h.veh)
    hw = (h.t - prev.t)[same]
    hd = h[same].assign(hw=hw)
    hd = hd[(hd.hw >= 20) & (hd.hw <= 3 * 3600)]
    g = hd.groupby(["stop", "route", "hour", "daytype"]).hw
    hwt = pd.concat([g.median().round(1).rename("median_headway_s"), g.size().rename("n")], axis=1).reset_index()
    hwt = hwt.rename(columns={"stop": "stop_id", "route": "route_id"})
    hwt.to_csv(os.path.join(OUT, "stop_headways.csv"), index=False)
    stats.update(headway_rows=len(hwt), headway_obs=len(hd))
    del df

if hold:
    df = load(hold)
    rng = np.random.default_rng(42)
    trips = df.groupby("trip")
    starts = trips.indices
    keys = list(starts.keys())
    rng.shuffle(keys)
    rows = []
    for k in keys:
        idx = starts[k]
        if len(idx) < 3: continue
        i, j = sorted(rng.choice(len(idx), 2, replace=False))
        a, b = df.iloc[idx[i]], df.iloc[idx[j]]
        dur = b.t - a.t
        dd = b.dist - a.dist
        if dur <= 0 or dd / dur > 19.4 or dur > max(3 * 3600, 0): continue
        rows.append((int(a.route), int(a.stop), int(b.stop), int(a.seq), int(b.seq),
                     pd.Timestamp(a.t + IST, unit="s").strftime("%Y-%m-%d %H:%M:%S"), round(float(dur), 1)))
        if len(rows) >= 20000: break
    ho = pd.DataFrame(rows, columns=["route_id", "board_stop", "alight_stop", "board_seq", "alight_seq", "depart_ts_ist", "actual_s"])
    ho.to_csv(os.path.join(OUT, "holdout_trips.csv"), index=False)
    stats.update(holdout_days=hold, holdout_rows=len(ho), holdout_passings=len(df))

json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1, default=str)
print(json.dumps(stats, indent=1, default=str))
