"""Model-only check on eval pairs: ride time on the ticket routes between the exact ticket stops
(no option selection involved) vs held-out actual median. Run from planner/."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router import Planner
from ttmodel import TravelTimeModel

E = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "eval")
P = Planner()
base = TravelTimeModel(P.net, model_dir="/nonexistent")
T = pd.read_csv(os.path.join(E, "test_pairs.csv"), dtype={"from_stop_id": str, "to_stop_id": str})
A = pd.read_csv(os.path.join(E, "actual_rides.csv"))
sidx = {s: i for i, s in enumerate(P.stop_id)}
rows = []
for r in T[T.kind == "direct"].itertuples():
    a, b = sidx.get(r.from_stop_id), sidx.get(r.to_stop_id)
    ts = pd.Timestamp(r.depart); t = ts.hour * 3600 + ts.minute * 60
    preds, preds_b = [], []
    for name in str(r.routes).split("|"):
        for ri, rt in enumerate(P.routes):
            if rt["short_name"] != name:
                continue
            ss = P.route_stops[ri]
            if a in ss and b in ss[ss.index(a) + 1:]:
                i = ss.index(a); j = ss.index(b, i + 1)
                preds.append(P.model.ride(ri, i, j, t, 0)[0]); preds_b.append(base.ride(ri, i, j, t, 0)[0])
    rows.append(dict(pair_id=r.pair_id, depart=r.depart, pred=np.median(preds) if preds else np.nan,
                     sched=np.median(preds_b) if preds_b else np.nan))
D = pd.DataFrame(rows).merge(A, on="pair_id")
D = D[D.n_obs >= 3]
for c in ["pred", "sched"]:
    e = (D[c] - D.actual_median_s) / D.actual_median_s * 100
    D[c + "_err"] = e.round(1)
    print(f"{c}: n={e.notna().sum()} median APE {e.abs().median():.1f}%  within20 {(e.abs() <= 20).mean():.0%}  bias {e.median():+.1f}%")
print(D.groupby(D.depart.str[11:13]).pred_err.agg(["count", "median", lambda x: x.abs().median()]).round(1))
print(D.sort_values("pred_err")[["pair_id", "depart", "actual_median_s", "pred", "pred_err"]].to_string(index=False))
