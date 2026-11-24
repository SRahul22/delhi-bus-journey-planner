import os, sys, re, numpy as np, pandas as pd
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "planner"))
from router import Planner
p = Planner()
d = pd.read_csv(os.path.join(os.path.dirname(__file__), "google_pairs_todo.csv"))
rows = []
for r in d.itertuples(index=False):
    dep = datetime.fromisoformat(r.depart)
    res = p.plan(r.from_lat, r.from_lng, r.to_lat, r.to_lng, dep)
    o = min(res["options"], key=lambda o: o["duration_s"])
    bus = [l for l in o["legs"] if l.get("mode") == "bus"]
    first_wait = bus[0].get("wait_s", 0) if bus else 0
    m = re.search(r"dep (\d+):(\d+) arr", r.google_notes)
    g_wait = (int(m.group(1))*60 + int(m.group(2)) - (dep.hour*60 + dep.minute))*60 if m else 0
    metro = "Line" in r.google_notes.split("fastest")[1].split(";")[0] if "fastest" in r.google_notes else False
    rows.append(dict(pair=r.pair_id, hour=dep.hour, google=r.google_s, g_wait=g_wait, ours=o["duration_s"],
                     first_wait=first_wait, walk=o["walk_m"], ride=o["ride_s"], transfers=o["transfers"],
                     ours_no_wait=o["duration_s"]-first_wait, routes="+".join(o["routes"])[:30]))
df = pd.DataFrame(rows)
def m(a, b):
    e = (a - b) / b
    return f"median err {np.median(abs(e))*100:.1f}%  within20 {np.mean(abs(e)<=.2)*100:.0f}%  median bias {np.median(a-b):+.0f}s"
print("as-is (ours incl. first wait vs google from its own departure):", m(df.ours, df.google))
print("like-for-like A (drop our first wait)                         :", m(df.ours_no_wait, df.google))
print("like-for-like B (add google's wait from requested time)       :", m(df.ours, df.google + df.g_wait))
for h, g in df.groupby("hour"): print(f"  hour {h}: A {m(g.ours_no_wait, g.google)}")
print("direct-only pairs (ours 0 transfers):", m(df[df.transfers==0].ours_no_wait, df[df.transfers==0].google), "n=", (df.transfers==0).sum())
print(df.to_string(index=False))
df.to_csv(os.path.join(os.path.dirname(__file__), "google_comparison.csv"), index=False)
