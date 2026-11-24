"""Latency check: p95 of GET /journey over 200 queries, in-process (FastAPI TestClient, no port).
Flags come from the environment (e.g. V2_WAIT=pooled V2_BAND=1). arrive_by is always set (T2.4 path).
Usage: python bench_latency.py [N] [out.json]
"""
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("REALTIME", "0")
t_import = time.time()
import app as appmod  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

startup_s = time.time() - t_import
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
P = appmod.planner
c = TestClient(appmod.app)
rng = random.Random(7)
qs = []
while len(qs) < N:
    a, b = rng.randrange(P.N), rng.randrange(P.N)
    d = ((P.stop_lat[a] - P.stop_lat[b]) * 111320) ** 2 + ((P.stop_lon[a] - P.stop_lon[b]) * 97700) ** 2
    if not 2000 ** 2 <= d <= 20000 ** 2:
        continue
    dep = datetime(2026, 10, 21, 6) + timedelta(seconds=rng.randrange(16 * 3600))
    qs.append(dict(from_lat=float(P.stop_lat[a]), from_lng=float(P.stop_lon[a]), to_lat=float(P.stop_lat[b]),
                   to_lng=float(P.stop_lon[b]), depart=dep.isoformat(), arrive_by=(dep + timedelta(minutes=75)).isoformat()))
lat, comp, errs = [], [], 0
for q in qs:
    t = time.perf_counter()
    r = c.get("/journey", params=q)
    lat.append((time.perf_counter() - t) * 1000)
    if r.status_code != 200:
        errs += 1
    else:
        comp.append(r.json().get("compute_ms", 0))
lat_s = sorted(lat)


def pct(p):
    return round(lat_s[min(len(lat_s) - 1, int(p * len(lat_s)))], 1)


out = dict(n=N, errors=errs, startup_s=round(startup_s, 1), p50_ms=pct(0.5), p95_ms=pct(0.95), max_ms=round(max(lat), 1),
           mean_ms=round(sum(lat) / len(lat), 1),
           flags={k: os.environ.get(k) for k in ("V2_WAIT", "V2_BAND", "V3_MODE", "POOLED_INDEX", "V3_PREDICT_DIR")},
           models_loaded=appmod.MODELS_LOADED)
print(json.dumps(out, indent=1, default=str))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
