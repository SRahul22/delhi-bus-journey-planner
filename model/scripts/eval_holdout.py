"""Evaluate segment_times.csv on holdout_trips.csv (pure model, no planner).
Prediction = sum over consecutive pattern segments of median_s for (hour of departure, daytype), falling back to
the segment's all-hour median for that daytype, then any daytype, then dist / 17 km/h.
Usage: python3 eval_holdout.py [model_dir] [median_s|mean_s]
"""
import sys
import numpy as np, pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else "."
COL = sys.argv[2] if len(sys.argv) > 2 else "median_s"   # or mean_s
seg = pd.read_csv(f"{D}/segment_times.csv")
pat = pd.read_csv(f"{D}/route_patterns.csv")
ho = pd.read_csv(f"{D}/holdout_trips.csv")
k3 = ["route_id", "from_stop", "to_stop"]
by_h = {k: v for k, v in zip(zip(seg.route_id, seg.from_stop, seg.to_stop, seg.hour, seg.daytype), seg[COL])}
w = seg.assign(w=seg[COL] * seg.n)
g = w.groupby(k3 + ["daytype"])[["w", "n"]].sum(); by_d = (g.w / g.n).to_dict()
g = w.groupby(k3)[["w", "n"]].sum(); by_a = (g.w / g.n).to_dict()
P = {r: (g.stop_id.to_numpy(), g.dist_m.to_numpy()) for r, g in pat.groupby("route_id")}
ts = pd.to_datetime(ho.depart_ts_ist)
ho["hour"] = ts.dt.hour; ho["daytype"] = np.where(ts.dt.weekday >= 5, "weekend", "weekday")
pred, src = [], []
for r, bs, as_, hr, dt in zip(ho.route_id, ho.board_seq, ho.alight_seq, ho.hour, ho.daytype):
    stops, dist = P[r]
    t = 0.0; fb = 0
    for k in range(bs, as_):
        a, b = stops[k], stops[k + 1]
        h = (hr + int(t // 3600)) % 24
        v = by_h.get((r, a, b, h, dt))
        if v is None:
            fb += 1
            v = by_d.get((r, a, b, dt), by_a.get((r, a, b)))
            if v is None: v = (dist[k + 1] - dist[k]) / (17 / 3.6)
        t += v
    pred.append(t); src.append(fb / max(1, as_ - bs))
ho["pred_s"] = pred; ho["fallback_frac"] = src
e = ho.pred_s - ho.actual_s
ape = (e.abs() / ho.actual_s)
print(f"n={len(ho)}  MAE={e.abs().mean():.0f}s  median AE={e.abs().median():.0f}s  bias={e.mean():+.0f}s  "
      f"MAPE={100*ape.mean():.1f}%  median APE={100*ape.median():.1f}%  within 20%: {100*(ape<0.2).mean():.1f}%")
for lo, hi in [(0, 600), (600, 1800), (1800, 3600), (3600, 99999)]:
    m = (ho.actual_s >= lo) & (ho.actual_s < hi)
    print(f"  actual {lo:>5}-{hi:<5}s n={m.sum():6d} medAPE={100*ape[m].median():5.1f}%  bias={e[m].mean():+5.0f}s")
print(f"mean fraction of segments needing hour fallback: {100*ho.fallback_frac.mean():.1f}%")
