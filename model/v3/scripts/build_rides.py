"""v3: build a ride-level dataset with lookup-table features, without target leakage.

Usage (Windows, E:\\hackathon-work\\v3):
  python build_rides.py <lookup_days> <ride_days> <out_prefix> <pairs_per_trip> [max_rides] [--export]

  lookup_days : comma list of days whose stop passings build the segment lookup (median/mean/p85/n per
                route segment x hour x daytype), exactly like model/scripts/aggregate.py.
  ride_days   : comma list of days from which rides (random board<alight pairs of one GPS trip) are sampled.
                Must not overlap lookup_days.
  --export    : also write the deploy artefact pieces (per-segment n array, route features) for model/v3.

Base predictions mimic the planner (planner/ttmodel.py): a segment cell is used if n >= 3, else the
segment's n-weighted all-hour value for that daytype, then any daytype, then distance / hourly network speed.
Hour advances along the ride using the cumulative median time (like model/scripts/eval_holdout.py).
Output: <out_prefix>.parquet (one row per ride) + <out_prefix>_log.txt
"""
import sys, os, time, json
import numpy as np, pandas as pd

W = r"E:\hackathon-work"
OUTD = os.path.join(W, "v3")
IST = 19800
MIN_N = 3
t0 = time.time()
def log(*a):
    print(f"{time.time()-t0:7.1f}s", *a, flush=True)

lookup_days = [d for d in sys.argv[1].split(",") if d]
ride_days = [d for d in sys.argv[2].split(",") if d]
prefix = sys.argv[3]
PER_TRIP = int(sys.argv[4])
MAXR = int(sys.argv[5]) if len(sys.argv) > 5 and not sys.argv[5].startswith("--") else 10**9
EXPORT = "--export" in sys.argv
assert not set(lookup_days) & set(ride_days), "lookup and ride days overlap"

pat = pd.read_csv(os.path.join(W, "route_patterns.csv")).sort_values(["route_id", "stop_sequence"]).reset_index(drop=True)
routes = pat.route_id.unique()
rsize = pat.groupby("route_id").size().reindex(routes).to_numpy()
assert (pat.groupby("route_id").stop_sequence.max().reindex(routes).to_numpy() == rsize - 1).all()
# flat segment index: route r, seg k (seq k -> k+1) = soff[r] + k ; stop position index = poff[r] + seq
nseg_r = rsize - 1
soff = np.concatenate([[0], np.cumsum(nseg_r)])[:-1]
poff = np.concatenate([[0], np.cumsum(rsize)])[:-1]
NSEG = int(nseg_r.sum())
ridx = pd.Series(np.arange(len(routes)), index=routes)
pdist = pat.dist_m.to_numpy()
seg_route = np.repeat(np.arange(len(routes)), nseg_r)
seg_k = np.concatenate([np.arange(n) for n in nseg_r])
seg_len = pdist[poff[seg_route] + seg_k + 1] - pdist[poff[seg_route] + seg_k]
seg_len = np.maximum(seg_len, 0)

def load(days):
    dfs = []
    for i, d in enumerate(days):
        z = np.load(os.path.join(W, "pass", f"pass_{d}.npz"))
        df = pd.DataFrame({k: z[k] for k in z.files})
        df["trip"] = df["trip"].astype(np.int64) + i * 10_000_000
        df["day"] = np.int8(int(d))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df.route.isin(routes)]
    df["ri"] = ridx.reindex(df.route).to_numpy()
    df["dist"] = pdist[poff[df.ri.to_numpy()] + df.seq.to_numpy()]
    lt = pd.to_datetime(df.t + IST, unit="s")
    df["hour"] = lt.dt.hour.astype(np.int8)
    df["dow"] = lt.dt.dayofweek.astype(np.int8)
    df["dt"] = (df.dow >= 5).astype(np.int8)
    return df.sort_values(["trip", "seq"], kind="stable").reset_index(drop=True)

# ------------------------------------------------------------------ lookup tables
df = load(lookup_days)
log("lookup passings", len(df))
nx = df.shift(-1)
m = ((nx.trip == df.trip) & (nx.seq == df.seq + 1)).to_numpy()
s = pd.DataFrame({"g": (soff[df.ri.to_numpy()[m]] + df.seq.to_numpy()[m]).astype(np.int64),
                  "dt": df.dt.to_numpy()[m], "hour": df.hour.to_numpy()[m],
                  "dur": (nx.t.to_numpy()[m] - df.t.to_numpy()[m]), "dd": (nx.dist.to_numpy()[m] - df.dist.to_numpy()[m])})
good = (s.dur >= 0) & ((s.dur > 0) | (s.dd < 30)) & (s.dd <= 19.4 * s.dur + 20) & (s.dur <= np.maximum(1800, s.dd / 1.0))
s = s[good]
log("segment obs", len(s))
del df, nx
gb = s.groupby(["g", "dt", "hour"]).dur
agg = pd.concat([gb.median().rename("med"), gb.mean().rename("mean"), gb.quantile(0.85).rename("p85"), gb.size().rename("n")], axis=1).reset_index()
shape = (NSEG, 2, 24)
MED = np.full(shape, np.nan, np.float32); MEAN = MED.copy(); P85 = MED.copy(); N = np.zeros(shape, np.int32)
ix = (agg.g.to_numpy(), agg.dt.to_numpy(), agg.hour.to_numpy())
N[ix] = agg.n.to_numpy()
ok = agg.n.to_numpy() >= MIN_N
oix = tuple(a[ok] for a in ix)
MED[oix] = agg.med.to_numpy()[ok]; MEAN[oix] = agg["mean"].to_numpy()[ok]; P85[oix] = agg.p85.to_numpy()[ok]
# network speed per (dt, hour) for last-resort fallback
sp = s.groupby(["dt", "hour"])[["dd", "dur"]].sum()
SPEED = np.full((2, 24), 17 / 3.6, np.float32)
for (dt_, h_), r_ in sp.iterrows():
    if r_.dur > 0: SPEED[int(dt_), int(h_)] = np.clip(r_.dd / r_.dur, 2, 15)
# fallbacks: n-weighted all-hour value per daytype, then any daytype
FB = np.zeros(shape, bool)
for A, col in ((MED, "med"), (MEAN, "mean"), (P85, "p85")):
    w = agg.assign(w=agg[col] * agg.n)
    gd = w.groupby(["g", "dt"])[["w", "n"]].sum(); vd = np.full((NSEG, 2), np.nan, np.float32)
    vd[gd.index.get_level_values(0), gd.index.get_level_values(1)] = (gd.w / gd.n).to_numpy()
    ga = w.groupby("g")[["w", "n"]].sum(); va = np.full(NSEG, np.nan, np.float32)
    va[ga.index.to_numpy()] = (ga.w / ga.n).to_numpy()
    miss = np.isnan(A)
    FB |= miss
    fill = np.where(np.isfinite(vd)[:, :, None], vd[:, :, None], va[:, None, None])
    fill = np.where(np.isfinite(fill), fill, seg_len[:, None, None] / SPEED[None, :, :])
    A[miss] = np.broadcast_to(fill, shape)[miss]
log("lookup built; cells with n>=3:", int((~FB).sum()), "of", FB.size)

# route-level features (from lookup days only)
segn = N.sum(axis=(1, 2))
tot_dur = np.bincount(s.g, weights=s.dur, minlength=NSEG)
tot_dd = np.bincount(s.g, weights=s.dd, minlength=NSEG)
r_dur = np.bincount(seg_route, weights=tot_dur, minlength=len(routes))
r_dd = np.bincount(seg_route, weights=tot_dd, minlength=len(routes))
route_feat = pd.DataFrame({"route_id": routes, "route_len_m": pdist[poff + rsize - 1], "route_nstops": rsize,
                           "route_speed": np.where(r_dur > 0, r_dd / np.maximum(r_dur, 1), np.nan),
                           "route_obs": np.bincount(seg_route, weights=segn, minlength=len(routes)),
                           "route_cov": np.bincount(seg_route, weights=(segn > 0), minlength=len(routes)) / np.maximum(nseg_r, 1)})
# skew of the route: mean/median ratio over all cells with data
okc = ~FB
rs = np.bincount(seg_route, weights=np.where(okc, MEAN, 0).sum(axis=(1, 2)), minlength=len(routes))
rm = np.bincount(seg_route, weights=np.where(okc, MED, 0).sum(axis=(1, 2)), minlength=len(routes))
route_feat["route_skew"] = np.where(rm > 0, rs / np.maximum(rm, 1e-6), np.nan)
del s

if EXPORT:
    stops_ = pd.read_csv(r"E:\hackathon-data\gtfs\stops.txt", usecols=["stop_id", "stop_lat", "stop_lon"])
    stops_ = stops_[stops_.stop_id.isin(pat.stop_id)]
    rf = route_feat.drop(columns="route_id")
    # everything predict.py needs, one file (median/mean already have the n<3 fallbacks filled in)
    np.savez_compressed(os.path.join(OUTD, "v3_lookup.npz"), N=np.minimum(N, 65535).astype(np.uint16),
                        MED=MED.astype(np.float16), MEAN=MEAN.astype(np.float16),
                        soff=soff, poff=poff, routes=routes, rsize=rsize, speed=SPEED,
                        pat_stop=pat.stop_id.to_numpy(), pat_dist=pdist.astype(np.float32),
                        stop_id=stops_.stop_id.to_numpy(), stop_lat=stops_.stop_lat.to_numpy(), stop_lon=stops_.stop_lon.to_numpy(),
                        rf_cols=np.array(rf.columns), rf=rf.to_numpy(np.float64), lookup_days=np.array(lookup_days))
    route_feat.to_csv(os.path.join(OUTD, f"{prefix}_route_feat.csv"), index=False)
    log("exported deploy pieces")

# ------------------------------------------------------------------ rides
df = load(ride_days)
log("ride passings", len(df))
trip = df.trip.to_numpy()
starts = np.flatnonzero(np.r_[True, trip[1:] != trip[:-1]])
lens = np.diff(np.r_[starts, len(trip)])
keep = lens >= 3
starts, lens = starts[keep], lens[keep]
rng = np.random.default_rng(int(ride_days[0]) * 7 + PER_TRIP)
st = np.repeat(starts, PER_TRIP); L = np.repeat(lens, PER_TRIP)
u1 = (rng.random(len(st)) * L).astype(np.int64); u2 = (rng.random(len(st)) * (L - 1)).astype(np.int64)
u2 = np.where(u2 >= u1, u2 + 1, u2)
i = st + np.minimum(u1, u2); j = st + np.maximum(u1, u2)
t = df.t.to_numpy(); dist = df.dist.to_numpy(); seq = df.seq.to_numpy()
dur = t[j] - t[i]; dd = dist[j] - dist[i]
ok = (dur > 0) & (dd / np.maximum(dur, 1e-6) <= 19.4) & (dur <= 3 * 3600)
i, j = i[ok], j[ok]
# de-duplicate identical pairs drawn twice from one trip
ij = np.unique(np.stack([i, j], 1), axis=0); i, j = ij[:, 0], ij[:, 1]
if len(i) > MAXR:
    sel = rng.choice(len(i), MAXR, replace=False); i, j = i[sel], j[sel]
log("rides", len(i))
R = pd.DataFrame({"route_id": df.route.to_numpy()[i], "board_stop": df.stop.to_numpy()[i], "alight_stop": df.stop.to_numpy()[j],
                  "board_seq": seq[i], "alight_seq": seq[j], "day": df.day.to_numpy()[i], "trip": trip[i],
                  "t0": t[i], "actual_s": t[j] - t[i], "dist_m": dist[j] - dist[i]})
ri = df.ri.to_numpy()[i]
del df
lt = pd.to_datetime(R.t0 + IST, unit="s")
R["hour_f"] = (lt.dt.hour + lt.dt.minute / 60).astype(np.float32)
R["dow"] = lt.dt.dayofweek.astype(np.int8)
R["dt"] = (R.dow >= 5).astype(np.int8)
R["n_stops"] = R.alight_seq - R.board_seq
# time-dependent sums
g = soff[ri] + R.board_seq.to_numpy()
steps = R.n_stops.to_numpy()
dtv = R.dt.to_numpy()
tsec = (lt.dt.hour * 3600 + lt.dt.minute * 60 + lt.dt.second).to_numpy().astype(np.float64)
sbl = np.zeros(len(R)); smed = np.zeros(len(R)); smean = np.zeros(len(R)); sp85 = np.zeros(len(R)); nfb = np.zeros(len(R)); slogn = np.zeros(len(R))
for k in range(int(steps.max())):
    a = np.flatnonzero(steps > k)
    if len(a) == 0: break
    gg = g[a] + k; h = ((tsec[a] + smed[a]) // 3600 % 24).astype(np.int64); d_ = dtv[a]
    # planner/ttmodel.py SEG_MODE=blend: median for the first 600 s of the ride, mean after (hour follows its own clock)
    hb = ((tsec[a] + sbl[a]) // 3600 % 24).astype(np.int64)
    sbl[a] += np.where(sbl[a] >= 600, MEAN[gg, d_, hb], MED[gg, d_, hb])
    smed[a] += MED[gg, d_, h]; smean[a] += MEAN[gg, d_, h]; sp85[a] += P85[gg, d_, h]
    nfb[a] += FB[gg, d_, h]; slogn[a] += np.log1p(N[gg, d_, h])
R["base_med"] = smed; R["base_mean"] = smean; R["base_p85"] = sp85; R["base_blend600"] = sbl
R["fb_frac"] = nfb / steps; R["mean_logn"] = slogn / steps
R = R.merge(route_feat, on="route_id", how="left")
R["board_pos"] = R.board_seq / (R.route_nstops - 1)
stops = pd.read_csv(r"E:\hackathon-data\gtfs\stops.txt", usecols=["stop_id", "stop_lat", "stop_lon"])
ll = stops.set_index("stop_id")
R["b_lat"] = ll.stop_lat.reindex(R.board_stop).to_numpy(); R["b_lon"] = ll.stop_lon.reindex(R.board_stop).to_numpy()
R["a_lat"] = ll.stop_lat.reindex(R.alight_stop).to_numpy(); R["a_lon"] = ll.stop_lon.reindex(R.alight_stop).to_numpy()
R.to_parquet(os.path.join(OUTD, f"{prefix}.parquet"), index=False)
ape = np.abs(R.base_med - R.actual_s) / R.actual_s
log("written", len(R), "base_med medAPE", round(100 * float(ape.median()), 2), "fb_frac", round(float(R.fb_frac.mean()), 3))
log("DONE")
