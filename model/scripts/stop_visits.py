"""v2 T0.1 (amendment A1): raw stop visits per (stop, route) for one day.
Usage: python stop_visits.py DD [limit]     (env VISITS_DENSIFY=0 disables track densification)
 -> E:\\hackathon-work\\v2\\data\\stop_visits_DD.npz  and route_trips_DD.csv
A visit = a run of pings of one vehicle (with that route_id) within RADIUS m of a stop on that
route's pattern; a new visit starts when the gap between in-range pings exceeds GAP s.
Arrays (all same length, sorted by stop, route, t):
  stop int32, route int32, veh int32 (index into veh_names), t int64 (first in-range point, unix s),
  t_last int64 (last in-range point; ~departure, use it at terminals), Points = raw pings plus points
  interpolated every 40 m between consecutive pings (<=300 s apart), so t/t_last are interpolated times. dmin float32 (closest ping distance, m), npings int16
Group index: g_stop, g_route int32, g_start int64 (offset), g_len int64 -- one row per (stop, route).
"""
import sys, os, sqlite3, time
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

DD = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DB = rf"E:\hackathon-data\bus-gps\bus_movements_2026_10_{DD}.db"
GTFS = os.environ.get("GTFS_DIR", r"E:\hackathon-data\gtfs")
PAT = r"E:\hackathon-work\route_patterns.csv"
OUT = r"E:\hackathon-work\v2\data"
RADIUS = 60.0
STEP = 40.0
DENSIFY = os.environ.get("VISITS_DENSIFY", "1") == "1"
MAXDT = float(os.environ.get("VISITS_MAXDT", 300))  # max ping gap (s) to interpolate across
GAP = 120
LAT0 = 28.6
KX = 111320.0 * np.cos(np.radians(LAT0)); KY = 110540.0
t0 = time.time()
def log(*a): print(f"[{DD} {time.time()-t0:7.1f}s]", *a, flush=True)

stops = pd.read_csv(os.path.join(GTFS, "stops.txt"), usecols=["stop_id", "stop_lat", "stop_lon"])
sx = (stops.stop_lon.to_numpy() - 77.0) * KX; sy = (stops.stop_lat.to_numpy() - LAT0) * KY
sid = stops.stop_id.to_numpy().astype(np.int64)
tree = cKDTree(np.c_[sx, sy])
pat = pd.read_csv(PAT)
patkey = np.unique(pat.route_id.to_numpy().astype(np.int64) * 1_000_000 + pat.stop_id.to_numpy().astype(np.int64))

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
q = ("select vehicle_id, trip_id, cast(route_id as integer) r, cast(timestamp as integer) t, "
     "cast(lat as real) la, cast(lng as real) lo from vehicle_feed")
if LIMIT: q += f" limit {LIMIT}"
df = pd.read_sql_query(q, con)
con.close()
log("rows", len(df))
df = df.dropna(subset=["r", "t", "la", "lo"])
df = df[(df.la > 27.5) & (df.la < 29.5) & (df.lo > 76.0) & (df.lo < 78.5)]
veh_codes, veh_names = pd.factorize(df.vehicle_id)
route = df.r.to_numpy().astype(np.int64)
# raw trip counts per route (denominator of usable share)
tr = pd.DataFrame({"route_id": route, "veh": veh_codes, "trip": pd.factorize(df.trip_id)[0]}).drop_duplicates()
tr.groupby("route_id").size().rename("raw_trips").reset_index().to_csv(os.path.join(OUT, f"route_trips_{DD}{'_test' if LIMIT else ''}.csv"), index=False)
del tr
ts = df.t.to_numpy().astype(np.int64)
px = (df.lo.to_numpy() - 77.0) * KX; py = (df.la.to_numpy() - LAT0) * KY
del df
log("clean pings", len(ts), "vehicles", len(veh_names), "densify", DENSIFY, "step", STEP, "maxdt", MAXDT)
# densify: pings are ~10-15 s apart, so a bus at speed can cross a 60 m circle without pinging inside it
# (79% of map-matched passings had a raw visit without this). Insert points every STEP m along the straight
# line between consecutive pings of the same vehicle and route (gap <= MAXDT = 300 s, jump <= 2.5 km), with linear time.
if DENSIFY:
    o = np.lexsort((ts, veh_codes)); vc = veh_codes[o]; route, ts, px, py = route[o], ts[o], px[o], py[o]
    dx = np.diff(px); dy = np.diff(py); dt_ = np.diff(ts); L = np.hypot(dx, dy)
    ok = (vc[1:] == vc[:-1]) & (route[1:] == route[:-1]) & (dt_ > 0) & (dt_ <= MAXDT) & (L <= 2500)
    nadd = np.where(ok, np.floor(L / STEP).astype(np.int64), 0)
    j = np.repeat(np.arange(len(nadd)), nadd)
    f = (np.arange(len(j)) - np.repeat(np.cumsum(nadd) - nadd, nadd) + 1) / np.repeat(nadd + 1, nadd)
    veh_codes = np.r_[vc, vc[j]]; route = np.r_[route, route[j]]
    ts = np.r_[ts, ts[j] + np.round(f * dt_[j]).astype(np.int64)]
    px = np.r_[px, px[j] + f * dx[j]]; py = np.r_[py, py[j] + f * dy[j]]
    log("densified points", len(ts))

K = 8
parts = []
CH = 2_000_000
for s in range(0, len(ts), CH):
    d, i = tree.query(np.c_[px[s:s+CH], py[s:s+CH]], k=K, distance_upper_bound=RADIUS, workers=2)
    ok = np.isfinite(d)
    rows, cols = np.nonzero(ok)
    pi = rows + s
    st = sid[i[rows, cols]]
    m = np.isin(route[pi] * 1_000_000 + st, patkey)
    parts.append((pi[m], st[m], d[rows, cols][m].astype(np.float32)))
pi = np.concatenate([p[0] for p in parts]); st = np.concatenate([p[1] for p in parts]); dd = np.concatenate([p[2] for p in parts])
log("in-range (ping, stop) pairs", len(pi))
v = veh_codes[pi].astype(np.int64); r = route[pi]; t = ts[pi]
o = np.lexsort((t, v, r, st))
v, r, t, st, dd = v[o], r[o], t[o], st[o], dd[o]
new = np.ones(len(t), bool)
new[1:] = (st[1:] != st[:-1]) | (r[1:] != r[:-1]) | (v[1:] != v[:-1]) | (t[1:] - t[:-1] > GAP)
start = np.flatnonzero(new)
end = np.r_[start[1:], len(t)]
vis = pd.DataFrame({"stop": st[start].astype(np.int32), "route": r[start].astype(np.int32), "veh": v[start].astype(np.int32),
                    "t": t[start], "t_last": t[end - 1], "dmin": np.minimum.reduceat(dd, start),
                    "npings": np.minimum(end - start, 32767).astype(np.int16)})
vis = vis.sort_values(["stop", "route", "t"], kind="stable").reset_index(drop=True)
g = vis.groupby(["stop", "route"], sort=False).size()
gs = np.r_[0, np.cumsum(g.to_numpy())[:-1]]
fn = os.path.join(OUT, f"stop_visits_{DD}{'_test' + os.environ.get('VISITS_TAG', '') if LIMIT else ''}.npz")
tmp = fn[:-4] + "_tmp.npz"  # write then rename, so readers never see a half-written file
np.savez(tmp, **{c: vis[c].to_numpy() for c in vis.columns},
         g_stop=g.index.get_level_values(0).to_numpy().astype(np.int32), g_route=g.index.get_level_values(1).to_numpy().astype(np.int32),
         g_start=gs.astype(np.int64), g_len=g.to_numpy().astype(np.int64), veh_names=np.asarray(veh_names, dtype=str),
         params=np.array([RADIUS, GAP, STEP if DENSIFY else 0.0, MAXDT]))  # version stamp: radius m, visit gap s, densify step m, max interp gap s
os.replace(tmp, fn)
log("done visits", len(vis), "groups", len(g), "->", fn)
