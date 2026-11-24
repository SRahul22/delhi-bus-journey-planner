"""Preprocess the static GTFS feed into a compact pickle used by the router.

Output: planner/cache/network.pkl with
  stops: DataFrame(stop_id, stop_name, lat, lon)  (index = internal stop idx)
  routes: list of dicts per internal route idx:
      route_id, short_name, desc, headsign, agency,
      stops (np.int32 internal stop idx, canonical pattern),
      sched_seg (float32 [nseg, 24]  scheduled seconds per segment by trip-start hour),
      dist (float32 [nseg] metres between consecutive stops),
      offset (float32 [nstops] scheduled seconds from first stop, all-day median),
      trips_per_hour (int16 [24])  (all trips of the route_id, by first departure hour),
      shape (float32 [npts,2] lat/lon), shape_idx (int32 [nstops]) index into shape per stop
  footpaths: dict stop_idx -> list[(stop_idx, walk_seconds, dist_m)]
Run: .venv/bin/python preprocess.py
"""
import os
import pickle
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from paths import gtfs_dir

GTFS = gtfs_dir()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "network.pkl")
WALK_SPEED = 1.2  # m/s
DETOUR = 1.3
FOOTPATH_RADIUS = 400.0
LAT0 = 28.6
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(LAT0))


def xy(lat, lon):
    return np.column_stack([np.asarray(lat) * M_PER_DEG_LAT, np.asarray(lon) * M_PER_DEG_LON])


def hms_to_s(s: pd.Series) -> np.ndarray:
    parts = s.str.split(":", expand=True).astype(np.int32)
    return (parts[0] * 3600 + parts[1] * 60 + parts[2]).to_numpy()


def main():
    t0 = time.time()
    stops = pd.read_csv(f"{GTFS}/stops.txt", dtype={"stop_id": str, "stop_code": str})
    routes = pd.read_csv(f"{GTFS}/routes.txt", dtype={"route_id": str})
    trips = pd.read_csv(f"{GTFS}/trips.txt", dtype={"route_id": str, "trip_id": str, "shape_id": str})
    st = pd.read_csv(f"{GTFS}/stop_times.txt", usecols=["trip_id", "arrival_time", "stop_id", "stop_sequence"],
                     dtype={"trip_id": str, "stop_id": str, "stop_sequence": np.int32, "arrival_time": str})
    print(f"loaded in {time.time()-t0:.1f}s; stop_times={len(st)}")

    stops = stops.reset_index(drop=True)
    stop_idx = {sid: i for i, sid in enumerate(stops.stop_id)}
    st["s"] = st.stop_id.map(stop_idx)
    st = st.dropna(subset=["s"])
    st["s"] = st.s.astype(np.int32)
    st["t"] = hms_to_s(st.arrival_time)
    st = st.sort_values(["trip_id", "stop_sequence"], kind="stable")

    # Pattern per trip
    g = st.groupby("trip_id", sort=False)
    trip_pat = g.s.agg(tuple)
    trip_times = g.t.agg(lambda x: np.asarray(x, dtype=np.int32))
    trip_start = g.t.first()
    trips = trips.set_index("trip_id")
    trips = trips.loc[trips.index.intersection(trip_pat.index)]
    trips["pat"] = trip_pat
    trips["start"] = trip_start
    print(f"patterns built in {time.time()-t0:.1f}s")

    # Shapes
    shapes = pd.read_csv(f"{GTFS}/shapes.txt", dtype={"shape_id": str})
    shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"])
    shape_pts = {sid: grp[["shape_pt_lat", "shape_pt_lon"]].to_numpy(np.float64)
                 for sid, grp in shapes.groupby("shape_id", sort=False)}

    sxy = xy(stops.stop_lat, stops.stop_lon)
    rinfo = routes.set_index("route_id")
    out_routes = []
    for rid, grp in trips.groupby("route_id"):
        if rid not in rinfo.index:
            continue
        cnt = Counter(grp.pat)
        pat, _ = cnt.most_common(1)[0]
        if len(pat) < 2:
            continue
        pat_arr = np.array(pat, dtype=np.int32)
        canon = grp[grp.pat == pat]
        tt = np.stack([trip_times[tid] for tid in canon.index]).astype(np.float64)  # ntrips x nstops
        seg = np.diff(tt, axis=1)
        seg = np.clip(seg, 0, None)
        hours = ((canon.start.to_numpy() // 3600) % 24).astype(int)
        allday = np.median(seg, axis=0)
        sched = np.empty((len(pat) - 1, 24), dtype=np.float32)
        for h in range(24):
            m = hours == h
            sched[:, h] = np.median(seg[m], axis=0) if m.any() else allday
        offset = np.concatenate([[0.0], np.cumsum(allday)]).astype(np.float32)
        tph = np.bincount(((grp.start.to_numpy() // 3600) % 24).astype(int), minlength=24).astype(np.int16)
        p = sxy[pat_arr]
        dist = np.hypot(*(np.diff(p, axis=0).T)).astype(np.float32)

        # shape projection
        shape_id = canon.shape_id.mode().iloc[0] if canon.shape_id.notna().any() else None
        shp = shape_pts.get(shape_id) if shape_id else None
        if shp is not None and len(shp) >= 2:
            shxy = xy(shp[:, 0], shp[:, 1])
            d = np.hypot(p[:, None, 0] - shxy[None, :, 0], p[:, None, 1] - shxy[None, :, 1])
            idx = np.empty(len(pat), dtype=np.int32)
            prev = 0
            for i in range(len(pat)):
                j = prev + int(np.argmin(d[i, prev:]))
                idx[i] = j
                prev = j
            # sanity: if projection is poor, drop shape
            if np.median(d[np.arange(len(pat)), idx]) > 150:
                shp, idx = None, None
            else:
                # along-shape distance is better than straight line
                cum = np.concatenate([[0], np.cumsum(np.hypot(*(np.diff(shxy, axis=0).T)))])
                sd = np.diff(cum[idx])
                good = sd > 0
                dist = np.where(good, np.maximum(sd, dist), dist).astype(np.float32)
        else:
            shp, idx = None, None
        r = rinfo.loc[rid]
        out_routes.append(dict(
            route_id=rid, short_name=str(r.route_short_name), desc=str(r.route_desc),
            agency=str(r.agency_id), headsign=str(canon.trip_headsign.mode().iloc[0]) if canon.trip_headsign.notna().any() else "",
            stops=pat_arr, sched_seg=sched, dist=dist, offset=offset, trips_per_hour=tph,
            shape=None if shp is None else shp.astype(np.float32), shape_idx=idx,
        ))
    print(f"{len(out_routes)} routes in {time.time()-t0:.1f}s")

    # Footpaths
    tree = cKDTree(sxy)
    pairs = tree.query_pairs(FOOTPATH_RADIUS, output_type="ndarray")
    fp = defaultdict(list)
    for a, b in pairs:
        dm = float(np.hypot(*(sxy[a] - sxy[b]))) * DETOUR
        w = dm / WALK_SPEED
        fp[int(a)].append((int(b), w, dm))
        fp[int(b)].append((int(a), w, dm))
    print(f"{len(pairs)} footpath pairs")

    out_stops = pd.DataFrame(dict(stop_id=stops.stop_id, stop_name=stops.stop_name,
                                  lat=stops.stop_lat, lon=stops.stop_lon))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(dict(stops=out_stops, routes=out_routes, footpaths=dict(fp)), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
