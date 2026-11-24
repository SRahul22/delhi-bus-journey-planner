import numpy as np, pandas as pd
R = pd.read_csv(r"E:\hackathon-work\v2\ml\realized_val_snapshot.csv"); R = R[R.scored.astype(bool)]
print("scored", len(R), R["sample"].value_counts().to_dict())
for a, b in [("plan_wait_s", "real_wait_s"), ("plan_ride_s", "real_ride_s"), ("plan_transfer_s", "real_transfer_s"), ("plan_total_s", "real_total_s")]:
    d = R[b] - R[a]
    print(f"{a:16s} plan med {R[a].median():7.0f}  real med {R[b].median():7.0f}  median diff {d.median():7.0f}  mean diff {d.mean():7.0f}  median ratio {np.median(R[b]/R[a].clip(lower=1)):.2f}")
for k, g in R.groupby("transfers"):
    print("transfers", k, len(g), "median real/plan total", round(float(np.median(g.real_total_s / g.plan_total_s)), 3), "under p85", round(100*float(np.mean(g.real_total_s <= g.plan_p85_s)), 1))
ride_ape = (R.plan_ride_s - R.real_ride_s).abs() / R.real_ride_s
print("ride-only median APE", round(100*float(ride_ape.median()), 1), "ride under plan_ride_p85", round(100*float(np.mean(R.real_ride_s <= R.plan_ride_p85_s)), 1))
print("wait > 20 min share real", round(100*float(np.mean(R.real_wait_s > 1200)), 1), "plan", round(100*float(np.mean(R.plan_wait_s > 1200)), 1))
