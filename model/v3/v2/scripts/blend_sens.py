import numpy as np, pandas as pd
R = pd.read_parquet(r"E:\hackathon-work\v2\ml\val19_24.parquet")
for nm, m in [("days 21-24 (no Dussehra/eve)", R.day >= 21), ("weekdays 21-23", R.day.between(21, 23)), ("19-20", R.day <= 20)]:
    d = R[m]; a = d.actual_s
    print(nm, len(d), {c: round(100*float((abs(d[c]-a)/a).median()), 3) for c in ["base_med","base_blend300","base_blend600","base_blend900","base_blend1200","base_mean"]})
