"""T2.2 validation on 19-24 Oct (no test days). In-process planner.
For random ODs (seed 2026, stops 2-20 km apart, departures 06:00-22:00) on validation days, take the first bus
leg of the recommended option and compare predicted waits with the realized wait on that day:
  realized_any   = first bus of ANY route in the attractive set at the boarding stop after arrival
  realized_single= first bus of the planned route only
Predictions: v1 (headway/2 table) and pooled (index built from days 1-18 only).
Ground truth: data agent's stop_visits_DD.npz (A1) when present, else v1 pass files (68% of trips -> waits inflated).
Usage: python validate_pooled.py [N] [index] [out.csv]
"""
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["V2_WAIT"] = "v1"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
os.environ["POOLED_INDEX"] = sys.argv[2] if len(sys.argv) > 2 else r"E:\hackathon-work\v2\planner\pooled_index_1-18.npz"
from datetime import datetime  # noqa: E402

import router  # noqa: E402
import v2ext  # noqa: E402

P = router.Planner()
PW = v2ext.pooled()
sidx = {s: k for k, s in enumerate(P.stop_id)}
DAYS = [19, 21, 22, 23, 24]  # 20 Oct (holiday) skipped: v1 planner treats it as a weekday
truth = {}


def day_times(d):
    if d not in truth:
        fv = rf"E:\hackathon-work\v2\data\stop_visits_{d:02d}.npz"
        src = "visits" if os.path.exists(fv) else "pass"
        z = np.load(fv if src == "visits" else rf"E:\hackathon-work\pass\pass_{d:02d}.npz")
        day0 = np.datetime64(f"2026-10-{d:02d}").astype("datetime64[s]").astype(np.int64)
        sec = (z["t"].astype(np.float64) + 19800 - day0)
        o = np.lexsort((sec, z["stop"]))
        st, rt, sec = z["stop"][o].astype(np.int64), z["route"][o].astype(np.int64), sec[o]
        u, start = np.unique(st, return_index=True)
        truth[d] = (src, u, np.append(start, len(st)), rt, sec)
    return truth[d]


def first_after(d, stop, routes, t):
    src, u, ptr, rt, sec = day_times(d)
    k = np.searchsorted(u, stop)
    if k >= len(u) or u[k] != stop:
        return None
    a, b = ptr[k], ptr[k + 1]
    m = np.isin(rt[a:b], routes) & (sec[a:b] >= t)
    if not m.any():
        return None
    return float(sec[a:b][m].min() - t)


rng = random.Random(2026)
rows = []
t0 = time.time()
lat, lon = P.stop_lat, P.stop_lon
tries = 0
while len(rows) < N and tries < N * 4:
    tries += 1
    a, b = rng.randrange(P.N), rng.randrange(P.N)
    dkm = router._dist_m(lat[a], lon[a], lat[b], lon[b]) / 1000
    if not 2 <= dkm <= 20:
        continue
    d = rng.choice(DAYS)
    dep = datetime(2026, 10, d, 6, 0) + __import__("datetime").timedelta(seconds=rng.randrange(16 * 3600))
    res = P.plan(lat[a], lon[a], lat[b], lon[b], dep, max_options=1)
    if not res["options"]:
        continue
    leg = next((l for l in res["options"][0]["legs"] if l["mode"] == "bus"), None)
    if leg is None:
        continue
    ri = P.route_by_id[leg["route_id"]]
    s0, s1 = sidx[leg["from_stop"]["stop_id"]], sidx[leg["to_stop"]["stop_id"]]
    ss = P.route_stops[ri]
    i = ss.index(s0); j = ss.index(s1, i + 1)
    t_arr = (datetime.fromisoformat(leg["arrive_at_stop"]).replace(tzinfo=None) - datetime(2026, 10, d)).total_seconds()
    dt = 1 if dep.weekday() >= 5 else 0
    pw, extra = P.pooled_wait(ri, i, j, t_arr, dt, leg["wait_s"])
    aset = [int(P.routes[rj]["route_id"]) for rj, *_ in P.attractive_set(ri, i, j, t_arr + leg["wait_s"], dt)]
    stop = int(P.stop_id[s0])
    ra = first_after(d, stop, aset, t_arr)
    rs = first_after(d, stop, [int(leg["route_id"])], t_arr)
    rows.append(dict(day=d, hour=int(t_arr // 3600), n_set=len(aset), src=extra["wait_source"], v1=leg["wait_s"],
                     pooled=pw, real_any=ra, real_single=rs))
print(f"{len(rows)} legs in {time.time()-t0:.0f}s; truth source:", {d: truth[d][0] for d in truth})
import pandas as pd  # noqa: E402

df = pd.DataFrame(rows)
out = sys.argv[3] if len(sys.argv) > 3 else r"E:\hackathon-work\v2\planner\pooled_validation.csv"
df.to_csv(out, index=False)


def summ(pred, real, mask=None):
    m = df[real].notna() & (df[real] < 3600)
    if mask is not None:
        m &= mask
    e = df.loc[m, pred] - df.loc[m, real]
    return f"n={m.sum():4d} bias mean {e.mean():+6.0f}s median {e.median():+6.0f}s  MAE {e.abs().mean():5.0f}s  medAE {e.abs().median():5.0f}s"


print("share of legs with pooled estimate:", round((df.src == "pooled").mean(), 3), " multi-route sets:", round((df.n_set > 1).mean(), 3))
print("realized wait for ANY route in set (what a rider takes):")
print("  v1     ", summ("v1", "real_any"))
print("  pooled ", summ("pooled", "real_any"))
print("  among legs with set size > 1:")
print("  v1     ", summ("v1", "real_any", df.n_set > 1))
print("  pooled ", summ("pooled", "real_any", df.n_set > 1))
print("realized wait for the planned route only:")
print("  v1     ", summ("v1", "real_single"))
print("mean waits: v1 %.0f pooled %.0f real_any %.0f real_single %.0f" % (df.v1.mean(), df.pooled.mean(), df.real_any.mean(), df.real_single.mean()))
print("cache:", PW.hits, "hits", PW.misses, "misses")
