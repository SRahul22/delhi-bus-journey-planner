"""Benchmark: our planner (config C = main, F = teammate waits) vs Google Maps BUS-ONLY vs actual GPS rides.
Inputs: google_bus_only_todo.csv (filled), pred_C.csv / pred_F.csv (from bench_pred.py), comparison.csv (actual ride
medians on the ticketed route, held-out weekdays 26-30 Oct). Like for like: Google's time runs from its own first
departure to arrival (no initial wait), so ours = total - first wait. Actual = in-bus ride only on the ticket route."""
import pandas as pd, numpy as np, re, sys
g = pd.read_csv("google_bus_only_todo.csv"); g = g[g.google_bus_s.notna()]
act = pd.read_csv("comparison.csv")[["pair_id", "actual_median_s", "n_obs"]]
def norm(r): return set(re.sub(r"(UP|DOWN)\d*$|\(.*?\)|STL|^0+", "", x.strip().upper()) for x in str(r).split("+") if x.strip())
def met(x, y):
    e = (x - y) / y
    return f"median diff {np.median(np.abs(e))*100:5.1f}%  within ±20% {np.mean(np.abs(e)<=.2)*100:4.0f}%  median bias {np.median(e)*100:+5.1f}%"
rows = []
for cfg in ["C", "F"]:
    p = pd.read_csv(f"pred_{cfg}.csv")
    m = g.merge(p, on="pair_id").merge(act, on="pair_id", how="left")
    m["ours_nowait"] = m.total_s - m.first_wait_s
    m["same_route"] = [len(norm(a) & norm(b)) > 0 for a, b in zip(m.routes, m.google_bus_routes)]
    print(f"\n=== config {cfg} (n={len(m)} pairs with Google bus-only times)")
    print(" ours (door-to-door, no first wait) vs Google bus-only:", met(m.ours_nowait, m.google_bus_s))
    print(" same first-choice route as Google:", f"{m.same_route.sum()}/{len(m)}")
    for nm, sub in [("same route", m[m.same_route]), ("different route", m[~m.same_route])]:
        if len(sub): print(f"   {nm:16s} n={len(sub):2d}", met(sub.ours_nowait, sub.google_bus_s))
    h = pd.to_datetime(m.depart).dt.hour
    for hh in sorted(h.unique()): print(f"   depart {hh:02d}:00 n={int((h==hh).sum()):2d}", met(m.ours_nowait[h==hh], m.google_bus_s[h==hh]))
    a = m.dropna(subset=["actual_median_s"])
    print(f" vs ACTUAL bus ride (n={len(a)}, ride only on the ticketed route):")
    print("   ours ride     ", met(a.ride_s, a.actual_median_s))
    print("   Google (d2d)  ", met(a.google_bus_s, a.actual_median_s))
    rows.append(m.assign(cfg=cfg))
pd.concat(rows).to_csv("google_bus_bench_out.csv", index=False)
