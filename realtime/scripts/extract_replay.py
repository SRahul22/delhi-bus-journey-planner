"""Extract a compact replay file of one day's GPS feed (runs on Windows, 1 process, read-only).
Usage: python extract_replay.py DD [step_s=30] [h0=6] [h1=22]
Output: E:\\hackathon-work\\realtime\\replay_DD.npz with arrays
  ts (int64 unix UTC), veh (int32), route (int32 GTFS route_id), trip (int32), lat5/lng5 (int32 = deg*1e5),
  veh_names, trip_names (str)
Keeps 1 ping per vehicle every >= step_s seconds (repeated/stale timestamps dropped), h0..h1 IST only.
"""
import sys, sqlite3, time, os
from array import array
from datetime import datetime, timezone, timedelta
import numpy as np
DD = sys.argv[1]
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 30
H0 = int(sys.argv[3]) if len(sys.argv) > 3 else 6
H1 = int(sys.argv[4]) if len(sys.argv) > 4 else 22
IST = timezone(timedelta(hours=5, minutes=30))
day0 = int(datetime(2026, 10, int(DD), tzinfo=IST).timestamp())
T0, T1 = day0 + H0 * 3600, day0 + H1 * 3600
DB = rf"E:\hackathon-data\bus-gps\bus_movements_2026_10_{DD}.db"
t00 = time.time()
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
last = {}; vidx = {}; tidx = {}
cols = {k: array("q") for k in ("ts", "veh", "route", "trip", "lat5", "lng5")}
n = 0
for vid, tid, rid, ts, la, lo in con.execute("select vehicle_id, trip_id, route_id, timestamp, lat, lng from vehicle_feed"):
    n += 1
    if n % 5_000_000 == 0: print(n, time.time() - t00, flush=True)
    try:
        t = int(ts)
    except (TypeError, ValueError):
        continue
    if t < T0 or t >= T1 or rid is None or la is None: continue
    if t < last.get(vid, -10**9) + STEP: continue
    try:
        a = float(la); b = float(lo)
    except ValueError:
        continue
    if not (28.0 < a < 29.2 and 76.5 < b < 77.8): continue
    last[vid] = t
    v = vidx.setdefault(vid, len(vidx)); k = tidx.setdefault(tid, len(tidx))
    cols["ts"].append(t); cols["veh"].append(v); cols["route"].append(int(rid)); cols["trip"].append(k)
    cols["lat5"].append(int(round(a * 1e5))); cols["lng5"].append(int(round(b * 1e5)))
con.close()
arr = {k: np.frombuffer(v, dtype=np.int64) for k, v in cols.items()}
o = np.argsort(arr["ts"], kind="stable")
out = dict(ts=arr["ts"][o], veh=arr["veh"][o].astype(np.int32), route=arr["route"][o].astype(np.int32),
           trip=arr["trip"][o].astype(np.int32), lat5=arr["lat5"][o].astype(np.int32), lng5=arr["lng5"][o].astype(np.int32),
           veh_names=np.array(list(vidx), dtype=str), trip_names=np.array([str(x) for x in tidx], dtype=str))
os.makedirs(r"E:\hackathon-work\realtime", exist_ok=True)
fn = rf"E:\hackathon-work\realtime\replay_{DD}.npz"
np.savez_compressed(fn, **out)
print("rows read", n, "kept", len(o), "veh", len(vidx), "trips", len(tidx), "MB", round(os.path.getsize(fn) / 1e6, 1), "s", round(time.time() - t00))
