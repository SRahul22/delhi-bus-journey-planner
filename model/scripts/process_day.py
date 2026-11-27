"""Process one day of DTC GPS into per-trip stop passing times.
Usage: python process_day.py DD   -> $WORK_DIR/pass/pass_DD.npz (default WORK_DIR=E:\\hackathon-work, DATA_DIR=E:\\hackathon-data)
Arrays saved: route(int32), trip(int32), veh(int32), seq(int16), stop(int32), t(float64 unix s)
"""
import sys, sqlite3, time, os, collections
from array import array
import numpy as np

W = os.environ.get("WORK_DIR", r"E:\hackathon-work")  # env WORK_DIR (route_shapes.csv, route_patterns.csv in; pass/ out)
DD = sys.argv[1]
DB = os.path.join(os.environ.get("DATA_DIR", r"E:\hackathon-data"), "bus-gps", f"bus_movements_2026_10_{DD}.db")  # env DATA_DIR
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
LAT0 = 28.6
KX = 111320.0 * np.cos(np.radians(LAT0)); KY = 110540.0
MAXD = 150.0      # max distance from shape (m)
VMAX = 25.0       # m/s max plausible progress speed
MAXGAP = 300.0    # max time gap to interpolate across (s)
USE_PAIR = os.environ.get("USE_PAIR") == "1"  # also match trips against the opposite-direction route (tested: +5% trips, 2x slower; off for the production run)

t0 = time.time()
def log(*a):
    print(f"[{DD} {time.time()-t0:7.1f}s]", *a, flush=True)

# ---- load route geometry
shapes = collections.defaultdict(list)
with open(os.path.join(W, "route_shapes.csv")) as f:
    next(f)
    for line in f:
        r, k, la, lo, d = line.split(",")
        shapes[int(r)].append((float(la), float(lo), float(d)))
pats = collections.defaultdict(list)
with open(os.path.join(W, "route_patterns.csv")) as f:
    next(f)
    for line in f:
        r, k, s, d = line.split(",")
        pats[int(r)].append((int(k), int(s), float(d)))
G = {}
for r, pts in shapes.items():
    a = np.array(pts)
    x = (a[:, 1] - 77.0) * KX; y = (a[:, 0] - LAT0) * KY
    if len(x) < 2: continue
    dx, dy = np.diff(x), np.diff(y)
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    p = np.array(pats[r])
    G[r] = (x[:-1], y[:-1], dx, dy, L2, np.sqrt(L2), a[:-1, 2], p[:, 0].astype(np.int16), p[:, 1].astype(np.int32), p[:, 2])
# direction pairs from GTFS short names (xxxUP <-> xxxDOWN)
PAIR = {}
import csv
by = collections.defaultdict(dict)
RT = os.path.join(W, "routes.txt")  # the v1 run used a copy of GTFS routes.txt in WORK_DIR
if not os.path.exists(RT):
    RT = os.path.join(os.environ.get("GTFS_DIR") or os.path.join(os.environ.get("DATA_DIR", r"E:\hackathon-data"), "gtfs"), "routes.txt")
with open(RT, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sn = row["route_short_name"].strip().upper()
        for suf in ("UP", "DOWN"):
            if sn.endswith(suf):
                by[sn[: -len(suf)].rstrip("_- ")][suf] = int(row["route_id"])
for b, d in by.items():
    if "UP" in d and "DOWN" in d:
        PAIR[d["UP"]] = d["DOWN"]; PAIR[d["DOWN"]] = d["UP"]
log("routes loaded", len(G), "pairs", len(PAIR) // 2)

# ---- stream GPS
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
q = "select vehicle_id, trip_id, route_id, timestamp, lat, lng from vehicle_feed"
if LIMIT: q += f" limit {LIMIT}"
trips = {}
vidx = {}
n = 0
for vid, tid, rid, ts, la, lo in con.execute(q):
    n += 1
    if rid is None or ts is None or la is None: continue
    key = (vid, tid)
    rec = trips.get(key)
    if rec is None:
        rec = trips[key] = (int(rid), array("q"), array("f"), array("f"))
    try:
        rec[1].append(int(ts)); rec[2].append(float(la)); rec[3].append(float(lo))
    except ValueError:
        continue
    if vid not in vidx: vidx[vid] = len(vidx)
con.close()
log("rows", n, "trips", len(trips))

# ---- per trip
def match(rid, ts, la, lo, stats):
    g = G.get(rid)
    if g is None: stats["no_route"] += 1; return None
    ax, ay, dx, dy, L2, L, cum, pseq, pstop, pdist = g
    px = (lo - 77.0) * KX; py = (la - LAT0) * KY
    along = np.empty(len(px)); dist = np.empty(len(px))
    CH = max(1, 2_000_000 // len(ax))
    for s in range(0, len(px), CH):
        X = px[s:s + CH, None]; Y = py[s:s + CH, None]
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1)
        d = np.hypot(ax + t * dx - X, ay + t * dy - Y)
        i = np.argmin(d, axis=1)
        r_ = np.arange(len(i))
        dist[s:s + CH] = d[r_, i]; along[s:s + CH] = cum[i] + t[r_, i] * L[i]
    ok = dist < MAXD
    if ok.sum() < 5: stats["off_shape"] += 1; return None
    ts, along = ts[ok], along[ok]
    best = None
    for st in range(min(5, len(ts))):
        acc = [st]; last = along[st]; lt = ts[st]
        for j in range(st + 1, len(ts)):
            a = along[j]
            if a < last - 100: continue
            if a - last > VMAX * (ts[j] - lt) + 100: continue
            acc.append(j)
            if a > last: last = a; lt = ts[j]
        if best is None or len(acc) > len(best): best = acc
        if len(acc) > 0.9 * len(ts): break
    if len(best) < 5: stats["filtered"] += 1; return None
    T = ts[best].astype(float); A = np.maximum.accumulate(along[best])
    if A[-1] - A[0] < 300: stats["no_progress"] += 1; return None
    idx = np.searchsorted(A, pdist, side="left")
    tpass = np.full(len(pdist), np.nan)
    inner = (idx > 0) & (idx < len(A))
    ii = idx[inner]
    gap = T[ii] - T[ii - 1]
    frac = (pdist[inner] - A[ii - 1]) / np.maximum(A[ii] - A[ii - 1], 1e-6)
    tp = T[ii - 1] + frac * gap
    tp[gap > MAXGAP] = np.nan
    tpass[inner] = tp
    if A[0] <= pdist[0] + 50 or idx[0] == 0:
        j = np.searchsorted(A, pdist[0] + 50, side="right") - 1
        if 0 <= j < len(A) - 1 and T[j + 1] - T[j] <= MAXGAP:
            tpass[0] = T[j]
    m = ~np.isnan(tpass)
    if m.sum() < 2: stats["few_stops"] += 1; return None
    return pseq[m], pstop[m], tpass[m]

out = {k: [] for k in ("route", "trip", "veh", "seq", "stop", "t")}
stats = collections.Counter(); stats2 = collections.Counter()
for ti, ((vid, tid), (rid, ts, la, lo)) in enumerate(trips.items()):
    ts = np.frombuffer(ts, dtype=np.int64); la = np.frombuffer(la, dtype=np.float32).astype(float); lo = np.frombuffer(lo, dtype=np.float32).astype(float)
    o = np.argsort(ts, kind="stable"); ts, la, lo = ts[o], la[o], lo[o]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, la, lo = ts[keep], la[keep], lo[keep]
    if len(ts) < 5: stats["short"] += 1; continue
    for k_, (r, st_) in enumerate(((rid, stats), (PAIR.get(rid) if USE_PAIR else None, stats2))):
        if r is None: continue
        res = match(r, ts, la, lo, st_)
        if res is None: continue
        st_["ok"] += 1
        sq, sp, tpz = res; n_ = len(sq)
        out["route"].append(np.full(n_, r, np.int32)); out["trip"].append(np.full(n_, 2 * ti + k_, np.int32))
        out["veh"].append(np.full(n_, vidx[vid], np.int32)); out["seq"].append(sq); out["stop"].append(sp); out["t"].append(tpz)
    if ti % 10000 == 0: log("trip", ti, dict(stats), "pair:", dict(stats2))

os.makedirs(os.path.join(W, "pass"), exist_ok=True)
fn = os.path.join(W, "pass", f"pass_{DD}{'_test' if LIMIT else ''}.npz")
np.savez(fn, **{k: np.concatenate(v) for k, v in out.items()})
log("done", dict(stats), "pair:", dict(stats2), "passings", sum(len(v) for v in out["t"]))
