"""Join planner_results.csv with actual_rides.csv (held-out GPS) and print accuracy.

  ../planner/.venv/bin/python compare.py      -> eval/comparison.csv
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(HERE, "planner_results.csv"))
A = pd.read_csv(os.path.join(HERE, "actual_rides.csv"))
D = R.merge(A, on="pair_id", how="left")
D = D[D.n_obs >= 3].copy()
# predicted ride on the ticket route if the planner offered it, else the fastest option's ride
D["pred_ride_s"] = D.ticket_route_ride_s.fillna(D.ride_s)
D["pred_source"] = np.where(D.ticket_route_ride_s.notna(), "ticket_route_option", "fastest_option")
for c in ["pred_ride_s", "ticket_sched_ride_s"]:
    D[c[:-2] + "_err_pct"] = ((D[c] - D.actual_median_s) / D.actual_median_s * 100).round(1)
D["within_iqr"] = D.pred_ride_s.between(D.actual_p25_s, D.actual_p75_s)
cols = ["pair_id", "band", "depart", "from_name", "to_name", "dist_km", "n_obs", "actual_median_s", "actual_p25_s",
        "actual_p75_s", "pred_ride_s", "pred_source", "ticket_sched_ride_s", "pred_ride_err_pct",
        "ticket_sched_ride_err_pct", "within_iqr"]
D[cols].to_csv(os.path.join(HERE, "comparison.csv"), index=False)


def m(err):
    e = err.dropna().abs()
    return f"n={len(e)} median APE {e.median():.1f}%  within 20%: {(e <= 20).mean():.0%}  bias {err.dropna().median():+.1f}%"


print("planner (model) ride vs held-out GPS actual:", m(D.pred_ride_err_pct))
print("GTFS schedule ride vs held-out GPS actual:  ", m(D.ticket_sched_ride_err_pct))
print("planner within actual IQR:", f"{D.within_iqr.mean():.0%}")
print(D.groupby("band").apply(lambda g: pd.Series(dict(n=len(g), model_mape=g.pred_ride_err_pct.abs().median(),
      sched_mape=g.ticket_sched_ride_err_pct.abs().median()))).round(1))
print(D[cols].sort_values("pred_ride_err_pct").to_string(index=False))
