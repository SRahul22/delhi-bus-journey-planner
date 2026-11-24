"""T2.2: build the stop-visit index used for pooled waits (V2_WAIT=pooled).

Source (default --source pass, fallback, see docs/handoff/T2.2.md): v1 map-matched stop passings E:\\hackathon-work\\pass\\pass_DD.npz.
Preferred source once the data agent publishes it: E:\\hackathon-work\\v2\\data\\stop_visits_DD.npz (A1);
pass --source visits to use it (format assumed per contract; checked at load).

Usage: python build_pooled_index.py [--days 1-24] [--out E:\\hackathon-work\\v2\\planner\\pooled_index_1-24.npz]
Output arrays: stops (int64 sorted unique GTFS stop_id), ptr (int64 CSR into rows), day (int8, Oct day),
               t (int32 IST seconds since midnight of that day), route (int32 GTFS route_id)
Rows are sorted by (stop, day, t). One vehicle passing the same stop twice within 120 s counts once.
"""
import argparse
import os
import time

import numpy as np

PASS = r"E:\hackathon-work\pass"
IST = 19800

ap = argparse.ArgumentParser()
ap.add_argument("--days", default="1-24")
ap.add_argument("--out", default=None)
ap.add_argument("--source", choices=["pass", "visits"], default="pass")
a = ap.parse_args()
d0, d1 = map(int, a.days.split("-"))
out = a.out or rf"E:\hackathon-work\v2\planner\pooled_index_{'visits_' if a.source == 'visits' else ''}{d0}-{d1}.npz"
VISITS = r"E:\hackathon-work\v2\data"
t0 = time.time()
S, R, V, D, T = [], [], [], [], []
for dd in range(d0, d1 + 1):
    z = np.load(os.path.join(PASS, f"pass_{dd:02d}.npz") if a.source == "pass"
                else os.path.join(VISITS, f"stop_visits_{dd:02d}.npz"))
    tl = z["t"] + IST
    day0 = np.datetime64(f"2026-10-{dd:02d}").astype("datetime64[s]").astype(np.int64)
    sec = (tl - day0).astype(np.int64)
    keep = (sec >= 0) & (sec < 86400 + 3 * 3600)  # keep the early hours after midnight with their service day
    S.append(z["stop"][keep].astype(np.int64)); R.append(z["route"][keep]); V.append(z["veh"][keep])
    T.append(sec[keep].astype(np.int32)); D.append(np.full(int(keep.sum()), dd, np.int8))
    print(f"day {dd}: {keep.sum()} passings ({time.time()-t0:.0f}s)", flush=True)
S = np.concatenate(S); R = np.concatenate(R); V = np.concatenate(V); D = np.concatenate(D); T = np.concatenate(T)
# dedupe: same stop, day, vehicle within 120 s
o = np.lexsort((T, V, D, S))
S, R, V, D, T = S[o], R[o], V[o], D[o], T[o]
dup = np.zeros(len(S), bool)
dup[1:] = (S[1:] == S[:-1]) & (D[1:] == D[:-1]) & (V[1:] == V[:-1]) & (T[1:] - T[:-1] < 120)
print("duplicates dropped:", int(dup.sum()), "of", len(S))
S, R, D, T = S[~dup], R[~dup], D[~dup], T[~dup]
o = np.lexsort((T, D, S))
S, R, D, T = S[o], R[o], D[o], T[o]
stops, start = np.unique(S, return_index=True)
ptr = np.append(start, len(S)).astype(np.int64)
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, stops=stops, ptr=ptr, day=D, t=T, route=R.astype(np.int32))
print(f"wrote {out}: {len(S)} rows, {len(stops)} stops, {time.time()-t0:.0f}s")
