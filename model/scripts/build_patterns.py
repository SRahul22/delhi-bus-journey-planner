"""Build per-route stop pattern + shape (local metric coords) from GTFS.
Outputs: route_patterns.csv (route_id,stop_sequence,stop_id,dist_m) and route_shapes.csv
(route_id,pt_seq,lat,lon,dist_m) used by the GPS processor.
Usage: python build_patterns.py <gtfs_dir> <out_dir>
"""
import sys, collections
import numpy as np, pandas as pd

gtfs, out = sys.argv[1], sys.argv[2]
LAT0 = 28.6
KX = 111320.0 * np.cos(np.radians(LAT0)); KY = 110540.0

def xy(lat, lon):
    return (np.asarray(lon) - 77.0) * KX, (np.asarray(lat) - LAT0) * KY

trips = pd.read_csv(f"{gtfs}/trips.txt", dtype={"shape_id": str})
st = pd.read_csv(f"{gtfs}/stop_times.txt", usecols=["trip_id", "stop_id", "stop_sequence"])
st = st.sort_values(["trip_id", "stop_sequence"])
seqs = st.groupby("trip_id").stop_id.apply(tuple)
trips["seq"] = trips.trip_id.map(seqs)
trips = trips.dropna(subset=["seq"])
stops = pd.read_csv(f"{gtfs}/stops.txt").set_index("stop_id")
shapes = pd.read_csv(f"{gtfs}/shapes.txt", dtype={"shape_id": str}).sort_values(["shape_id", "shape_pt_sequence"])
shp = {k: g[["shape_pt_lat", "shape_pt_lon"]].to_numpy() for k, g in shapes.groupby("shape_id")}

pat_rows, shp_rows, bad = [], [], 0
for rid, g in trips.groupby("route_id"):
    seq = collections.Counter(g.seq).most_common(1)[0][0]
    sids = collections.Counter(g[g.seq == seq].shape_id.dropna()).most_common(1)
    if not sids or sids[0][0] not in shp:
        # fallback: build shape from stop coordinates
        ll = stops.loc[list(seq), ["stop_lat", "stop_lon"]].to_numpy()
    else:
        ll = shp[sids[0][0]]
    x, y = xy(ll[:, 0], ll[:, 1])
    seglen = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0], np.cumsum(seglen)])
    # project stops sequentially (monotone)
    sx, sy = xy(stops.loc[list(seq), "stop_lat"].to_numpy(), stops.loc[list(seq), "stop_lon"].to_numpy())
    ax, ay, dx, dy = x[:-1], y[:-1], np.diff(x), np.diff(y)
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    prev = -1e9; dists = []
    for px, py in zip(sx, sy):
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
        d = np.hypot(ax + t * dx - px, ay + t * dy - py)
        along = cum[:-1] + t * np.sqrt(L2)
        d2 = np.where(along >= prev - 30, d, np.inf)
        if np.isinf(d2).all(): d2 = d
        i = int(np.argmin(d2))
        if d2[i] > 300: bad += 1
        a = max(float(along[i]), prev if prev > 0 else 0.0)
        dists.append(a); prev = a
    for k, (s, dm) in enumerate(zip(seq, dists)):
        pat_rows.append((rid, k, s, round(dm, 1)))
    for k in range(len(ll)):
        shp_rows.append((rid, k, ll[k, 0], ll[k, 1], round(float(cum[k]), 1)))

pd.DataFrame(pat_rows, columns=["route_id", "stop_sequence", "stop_id", "dist_m"]).to_csv(f"{out}/route_patterns.csv", index=False)
pd.DataFrame(shp_rows, columns=["route_id", "pt_seq", "lat", "lon", "dist_m"]).to_csv(f"{out}/route_shapes.csv", index=False)
print("routes", trips.route_id.nunique(), "pattern rows", len(pat_rows), "stops far from shape", bad)
