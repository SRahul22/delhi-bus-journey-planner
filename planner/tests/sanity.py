"""Sanity test: plan a few known Delhi pairs and print options.  Run from planner/: .venv/bin/python tests/sanity.py"""
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router import Planner  # noqa: E402

PAIRS = {
    "KashmereGate->CP": (28.6675, 77.2290, 28.6315, 77.2167),
    "Rohini22->Azadpur": (28.7180, 77.0639, 28.7075, 77.1755),
    "AnandVihar->NehruPlace": (28.6470, 77.3160, 28.5485, 77.2520),
    "Dwarka->AIIMS": (28.5921, 77.0460, 28.5672, 77.2100),
}
REALTIME = "--realtime" in sys.argv
args = [a for a in sys.argv[1:] if a != "--realtime"]

P = Planner()
if REALTIME:  # live-replay planner (realtime/service.py); departures = replay clock "now"
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "realtime"))
    from service import RealtimeService  # noqa: E402
    RT = RealtimeService(P, start=args[0] if args else None)
    RT.stop()
    P = RT.rt_planner
    DEPARTS = [RT.now_dt().replace(tzinfo=None).isoformat(timespec="minutes")]
    print("realtime replay clock", RT.feed.now_iso(), RT.status().get("city_ratio"))
else:
    DEPARTS = args or ["2026-11-25T09:00", "2026-11-25T14:00", "2026-11-25T18:00", "2026-11-29T11:00"]
lat = []
for nm, (a, b, c, d) in PAIRS.items():
    for dep in DEPARTS:
        t = time.time()
        r = P.plan(a, b, c, d, datetime.fromisoformat(dep))
        lat.append(time.time() - t)
        print(f"== {nm} {dep} {lat[-1]*1000:.0f}ms crow={r['query']['straight_line_m']}m")
        for o in r["options"]:
            bl = [l for l in o["legs"] if l["mode"] == "bus"]
            print(f"  {','.join(o['labels'])}: {o['duration_s']/60:.0f}min (p85 {o['duration_p85_s']/60:.0f}) "
                  f"walk={o['walk_m']}m wait={o['wait_s']/60:.0f}m tr={o['transfers']} | " +
                  " | ".join(f"{l['route']} {l['from_stop']['name'][:22]}->{l['to_stop']['name'][:22]} "
                             f"{l['num_stops']}st {l['ride_s']/60:.0f}m {l['speed_kmh']}kmh" for l in bl))
print(f"latency ms: median {sorted(lat)[len(lat)//2]*1000:.0f}, max {max(lat)*1000:.0f}")
