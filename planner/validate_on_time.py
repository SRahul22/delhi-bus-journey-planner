"""T2.4 validation (ride level, 19-24 Oct, no test days): is on_time_probability calibrated?
Rides: ml's E:\\hackathon-work\\v2\\ml\\val19_24.parquet (300k validation rides, tables from days 1-18).
Band: v1 (p50 = base_blend600 = planner point estimate without XGBoost, p85 = base_p85 = sum of segment p85),
floored at p50. Each ride gets a random deadline D = p50 * U(0.6, 1.8) (seed 2026); predicted P = Phi((ln D - mu)/sigma).
Reports a reliability table (10 bins), Brier score and expected calibration error (ECE), by ride-length bucket.
Usage: python validate_on_time.py [out.json]
"""
import json
import math
import sys

import numpy as np
import pandas as pd

df = pd.read_parquet(r"E:\hackathon-work\v2\ml\val19_24.parquet", columns=["actual_s", "base_blend600", "base_p85", "base_med", "hour_f", "dow"])
df = df[(df.actual_s > 0) & (df.base_blend600 > 0)]
p50 = df.base_blend600.to_numpy()
p85 = np.maximum(df.base_p85.to_numpy(), p50)
rng = np.random.default_rng(2026)
D = p50 * rng.uniform(0.6, 1.8, len(df))
sig = np.maximum(np.log(p85 / p50) / 1.0364, 1e-6)
z = (np.log(D) - np.log(p50)) / sig
from math import erf  # noqa: E402
P = 0.5 * (1 + np.vectorize(erf)(z / math.sqrt(2)))
y = (df.actual_s.to_numpy() <= D).astype(float)
m = df.actual_s.to_numpy() / 60
bk = np.select([m < 10, m < 30, m < 60], ["<10", "10-30", "30-60"], ">60")


def summary(mask):
    p, t = P[mask], y[mask]
    bins = np.clip((p * 10).astype(int), 0, 9)
    rel = []
    ece = 0.0
    for b in range(10):
        s = bins == b
        if s.sum():
            rel.append(dict(bin=f"{b/10:.1f}-{(b+1)/10:.1f}", n=int(s.sum()), pred=round(float(p[s].mean()), 3),
                            obs=round(float(t[s].mean()), 3)))
            ece += s.sum() / len(p) * abs(p[s].mean() - t[s].mean())
    return dict(n=int(mask.sum()), brier=round(float(np.mean((p - t) ** 2)), 4), ece=round(float(ece), 4),
                mean_pred=round(float(p.mean()), 3), mean_obs=round(float(t.mean()), 3), reliability=rel)


out = {"all": summary(np.ones(len(P), bool))}
for b in ["<10", "10-30", "30-60", ">60"]:
    out[f"actual {b}"] = summary(bk == b)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "reliability"} for k, v in out.items()}, indent=1))
print("reliability (all):")
for r in out["all"]["reliability"]:
    print(f"  {r['bin']}: n={r['n']:6d} predicted {r['pred']:.3f} observed {r['obs']:.3f}")
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
