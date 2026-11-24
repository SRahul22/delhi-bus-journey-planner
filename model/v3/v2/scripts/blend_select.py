"""T0.4 (ml): choose BLEND_S from {300,450,600,900,1200} on days 19-24 rides with 1-18 tables (val19_24.parquet).
Criterion: overall median APE (ties -> within +-20%). Writes E:\\hackathon-work\\v2\\ml\\blend_select.json."""
import json, os, numpy as np, pandas as pd
OUT = r"E:\hackathon-work\v2\ml"
import sys
NAME = sys.argv[1] if len(sys.argv) > 1 else "val19_24"
R = pd.read_parquet(os.path.join(OUT, NAME + ".parquet"))
f4 = pd.read_parquet(r"E:\hackathon-work\v3\fold4.parquet", columns=["actual_s", "base_med"])
same = bool(np.array_equal(f4.actual_s.to_numpy(), R.actual_s.to_numpy()))
a = R.actual_s.to_numpy()
B = [(0, 600, "<10"), (600, 1800, "10-30"), (1800, 3600, "30-60"), (3600, 1e12, ">60")]
def met(p, m=None):
    if m is not None: p, aa = p[m], a[m]
    else: aa = a
    e = p - aa; ape = np.abs(e) / aa
    return dict(n=int(len(aa)), median_ape=round(100*float(np.median(ape)), 3), within20=round(100*float(np.mean(ape < .2)), 2),
                mean_bias_pct=round(100*float(np.mean(e/aa)), 2), median_bias_s=round(float(np.median(e)), 1))
cols = {"median": "base_med", "mean": "base_mean", **{f"blend{s}": f"base_blend{s}" for s in (300, 450, 600, 900, 1200)}}
res = {}
for nm, c in cols.items():
    p = R[c].to_numpy(); r = {"all": met(p)}
    for lo, hi, bn in B:
        r[bn] = met(p, (a >= lo) & (a < hi))
    res[nm] = r
    print(nm, r["all"], {bn: r[bn]["median_ape"] for _, _, bn in B})
cand = [k for k in res if k.startswith("blend")]
best = min(cand, key=lambda k: (res[k]["all"]["median_ape"], -res[k]["all"]["within20"]))
print("chosen", best, "same rides as fold4:", same)
json.dump(dict(chosen_BLEND_S=int(best[5:]), criterion="min overall median APE on days 19-24 rides, 1-18 tables",
               rides=f"{NAME}.parquet = fold4 rides (identical rides: {same})", results=res), open(os.path.join(OUT, "blend_select.json" if NAME == "val19_24" else f"blend_select_{NAME}.json"), "w"), indent=1)
