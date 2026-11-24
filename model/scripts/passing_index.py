"""v2 T0.1: passing index per day from map-matched pass files (contract 1).
Usage: python passing_index.py DD[,DD...]   -> E:\\hackathon-work\\v2\\data\\passing_index_DD.npz
Arrays sorted by (stop, t): stop int32, t float64 (unix s), route int32, trip int32 (pass-file trip id, unique within day),
  seq int16, veh int32.  Per-stop index: g_stop int32, g_start int64, g_len int64.
Usable share per route that day: us_route int32, us_matched int32 (map-matched trips), us_raw int32
  (distinct raw (vehicle, trip_id) with that route_id, from route_trips_DD.csv), us_share float32 (=matched/raw; NaN if raw unknown).
"""
import sys, os, time
import numpy as np, pandas as pd
W = r"E:\hackathon-work"; OUT = os.path.join(W, "v2", "data")
for DD in sys.argv[1].split(","):
    t0 = time.time()
    z = np.load(os.path.join(W, "pass", f"pass_{DD}.npz"))
    stop = z["stop"].astype(np.int32); t = z["t"].astype(np.float64)
    o = np.lexsort((t, stop))
    a = {k: z[k][o] for k in ("route", "trip", "seq", "veh")}
    stop, t = stop[o], t[o]
    us, gs, gl = np.unique(stop, return_index=True, return_counts=True)
    m = pd.DataFrame({"route": a["route"], "trip": a["trip"]}).drop_duplicates().groupby("route").size()
    rt_fn = os.path.join(OUT, f"route_trips_{DD}.csv")
    raw = pd.read_csv(rt_fn).set_index("route_id").raw_trips if os.path.exists(rt_fn) else pd.Series(dtype=float)
    rr = raw.reindex(m.index)
    np.savez(os.path.join(OUT, f"passing_index_{DD}.npz"), stop=stop, t=t, route=a["route"].astype(np.int32), trip=a["trip"].astype(np.int32),
             seq=a["seq"].astype(np.int16), veh=a["veh"].astype(np.int32), g_stop=us.astype(np.int32), g_start=gs.astype(np.int64), g_len=gl.astype(np.int64),
             us_route=m.index.to_numpy().astype(np.int32), us_matched=m.to_numpy().astype(np.int32),
             us_raw=rr.fillna(-1).to_numpy().astype(np.int32), us_share=(m / rr).to_numpy().astype(np.float32))
    tot = (m.sum() / rr.sum()) if rr.notna().any() else float("nan")
    print(f"day {DD}: passings {len(t)} stops {len(us)} routes {len(m)} overall usable share {tot:.3f} raw_known={rr.notna().all()} {time.time()-t0:.1f}s", flush=True)
