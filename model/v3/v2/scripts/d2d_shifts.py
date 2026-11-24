"""T2.6: door-to-door conformal shifts from eval's realized journeys on days 19-24 (realized_val.csv, contract 6).

  python d2d_shifts.py [realized_val.csv] [--p50 COL] [--p85 COL] [--a3]
  --a3: rebuild the plan P85 with journey_band.combine (A3) from plan_walk_s, plan_wait_s, plan_ride_s,
        plan_ride_p85_s, plan_transfer_s (waits: p85 = 1.7 x p50); writes shifts_d2d_a3.json AND shifts_d2d.json.
  without --a3: calibrates eval's plan_p85_s (v1 duration_p85_s) -> shifts_d2d_v1p85.json.

Same method as T1.2 on journey totals: score s = ln(real_total_s) - ln(plan P85); per bucket(plan P50) x peak,
shift = ceil((n+1)*0.85)-th smallest score; cells with n < MIN_CELL pooled with the neighbouring bucket.
Only scored rows are used. Honest check: leave-one-day-out (shifts from the other 5 days, coverage on the held-out day).
Writes E:\\hackathon-work\\v2\\ml\\shifts_d2d.json (+ d2d_val.json) and copies them to model/v3/v2/.
The planner multiplies its door-to-door P85 (journey_band.combine, A3) by exp(shift); journey_band.journey_band does it.
NOTE: the shifts are only valid for the P85 definition that produced plan P85 in realized_val.csv; if eval's plan P85
is the v1 duration_p85_s rather than the A3 combination, re-run after eval re-plans with the A3 band.
"""
import json, math, os, shutil, sys
import numpy as np, pandas as pd

OUT = os.environ.get("ML_OUT", r"E:\hackathon-work\v2\ml")
REPO_OUT = os.environ.get("ML_REPO_OUT", r"E:\hackathon-wt\ml\model\v3\v2")
ALPHA, MIN_CELL, Z85 = 0.85, 500, 1.0364
BUCKETS = [(0, 600, "<10"), (600, 1800, "10-30"), (1800, 3600, "30-60"), (3600, 1e12, ">60")]   # section 4
HOLI = {2, 20}

args = [a for a in sys.argv[1:]]
def opt(name):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return None
c50, c85 = opt("--p50"), opt("--p85")
A3 = "--a3" in args
if A3:
    args.remove("--a3")
WAIT_K = 1.7   # --a3: wait p85/p50 for a random arrival on a regular headway (uniform wait: 0.85H / 0.5H)
path = args[0] if args else r"E:\hackathon-work\v2\eval\realized_val.csv"
R = pd.read_csv(path)
print("rows", len(R), "columns", list(R.columns))
pick = lambda cands: next((c for c in cands if c in R.columns), None)
c50 = c50 or pick(["plan_p50_s", "plan_total_s", "plan_duration_s", "plan_total_p50_s"])
c85 = c85 or pick(["plan_p85_s", "plan_duration_p85_s", "plan_total_p85_s"])
assert c50 and c85, f"need plan P50/P85 columns, got {list(R.columns)}"
if "scored" in R.columns:
    R = R[R.scored.astype(bool)]
R = R[(R.real_total_s > 0) & (R[c50] > 0) & (R[c85] > 0)].copy()
assert R.day.between(19, 24).all(), "realized_val must be days 19-24 only"
if A3:
    # A3: rebuild the journey P85 in log-normal space from eval's plan components instead of v1's duration_p85_s
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.environ["V2_D2D_SHIFTS"] = os.path.join(OUT, "__none__.json")   # combine() only, no old shifts
    from journey_band import combine
    g = lambda c: R[c].fillna(0).to_numpy(float) if c in R.columns else np.zeros(len(R))
    walk, wait, ride, ride85, xfer = g("plan_walk_s"), g("plan_wait_s"), g("plan_ride_s"), g("plan_ride_p85_s"), g("plan_transfer_s")
    a3 = np.array([combine([(w, w), (wt, wt * WAIT_K), (r, max(r85, r)), (x, x * WAIT_K)])[:2]
                   for w, wt, r, r85, x in zip(walk, wait, ride, ride85, xfer)])
    R["a3_p50_s"], R["a3_p85_s"] = a3[:, 0], a3[:, 1]
    print("A3 P50 vs plan_total_s median ratio", round(float(np.median(R.a3_p50_s / R[c50])), 4))
    c50, c85 = "a3_p50_s", "a3_p85_s"
    R = R[R[c50] > 0].copy()
print("scored rows", len(R), "p50 col", c50, "p85 col", c85)
dow = pd.to_datetime(dict(year=2026, month=10, day=R.day)).dt.dayofweek.to_numpy()
h = R.depart_s.to_numpy() / 3600.0
peak = ((dow < 5) & ~R.day.isin(HOLI).to_numpy() & (((h >= 8) & (h < 11)) | ((h >= 17) & (h < 21)))).astype(int)
p50 = R[c50].to_numpy(float); p85 = np.maximum(R[c85].to_numpy(float), p50); act = R.real_total_s.to_numpy(float)
bk = np.digitize(p50, [b[1] for b in BUCKETS[:-1]])
s = np.log(act) - np.log(p85)
day = R.day.to_numpy()


def conformal(sc, bk, pk):
    out, ns = {}, {}
    for pf in (0, 1):
        groups = [[k] for k in range(len(BUCKETS))]
        cnt = lambda gr: int(np.sum(np.isin(bk, gr) & (pk == pf)))
        while len(groups) > 1:
            small = [i for i, g in enumerate(groups) if cnt(g) < MIN_CELL]
            if not small: break
            i = small[0]; j = i - 1 if i == len(groups) - 1 else i + 1
            groups[min(i, j)] = sorted(groups[i] + groups[j]); del groups[max(i, j)]
        for gr in groups:
            v = np.sort(sc[np.isin(bk, gr) & (pk == pf)]); n = len(v)
            sh = float(v[min(n, math.ceil((n + 1) * ALPHA)) - 1]) if n else 0.0
            for k in gr:
                key = f"{BUCKETS[k][2]}|{'peak' if pf else 'offpeak'}"; out[key] = round(sh, 5); ns[key] = n
    return out, ns


def apply(sh, bk, pk):
    return np.array([sh[f"{BUCKETS[b][2]}|{'peak' if p else 'offpeak'}"] for b, p in zip(bk, pk)])


def cov_table(p85c, mask=None):
    m = np.ones(len(act), bool) if mask is None else mask
    t = {"all": dict(n=int(m.sum()), coverage85=round(100 * float(np.mean(act[m] <= p85c[m])), 2),
                     under_p50=round(100 * float(np.mean(act[m] <= p50[m])), 2),
                     width_med=round(float(np.median(p85c[m] / p50[m] - 1)), 4))}
    for k, (_, _, nm) in enumerate(BUCKETS):
        for pf in (0, 1):
            mm = m & (bk == k) & (peak == pf)
            if mm.any():
                t[f"{nm}|{'peak' if pf else 'offpeak'}"] = dict(n=int(mm.sum()), coverage85=round(100 * float(np.mean(act[mm] <= p85c[mm])), 2))
    return t


shifts, ns = conformal(s, bk, peak)
lodo = np.empty(len(act))
for d in np.unique(day):
    te = day == d
    sh_d, _ = conformal(s[~te], bk[~te], peak[~te])
    lodo[te] = np.maximum(p50[te], p85[te] * np.exp(apply(sh_d, bk[te], peak[te])))
res = {"uncalibrated": cov_table(p85), "leave_one_day_out_calibrated": cov_table(lodo)}
# lognormal calibration (section 4): share under the predicted 50/85/90th percentiles, LODO band
sig = np.log(lodo / p50) / Z85
for q, z in ((50, 0.0), (85, Z85), (90, 1.2816)):
    res[f"lodo_share_under_p{q}"] = round(100 * float(np.mean(act <= p50 * np.exp(z * sig))), 2)
print(json.dumps(res, indent=1))
meta = dict(version="v2-d2d-1", source=path, p50_col=c50, p85_col=c85, n=ns, shifts=shifts,
            buckets_s=[[lo, hi, nm] for lo, hi, nm in BUCKETS], alpha=ALPHA, min_cell=MIN_CELL,
            peak="weekday (not 2/20 Oct) depart 08-10:59, 17-20:59",
            serve="P85_d2d = max(P50, P85 * exp(shift[bucket(P50)|peak])) (journey_band.journey_band)")
os.makedirs(OUT, exist_ok=True)
SUF = "_a3" if A3 else "_v1p85"
meta["a3_wait_k"] = WAIT_K if A3 else None
json.dump(meta, open(os.path.join(OUT, f"shifts_d2d{SUF}.json"), "w"), indent=1)
json.dump(res, open(os.path.join(OUT, f"d2d_val{SUF}.json"), "w"), indent=1)
for f in (f"shifts_d2d{SUF}.json", f"d2d_val{SUF}.json"):
    if os.path.abspath(OUT) != os.path.abspath(REPO_OUT):
        shutil.copy(os.path.join(OUT, f), os.path.join(REPO_OUT, f))
if A3:   # contract 3 name: shifts for the A3 journey band (journey_band.journey_band applies this file)
    for o in {OUT, REPO_OUT}:
        shutil.copy(os.path.join(OUT, f"shifts_d2d{SUF}.json"), os.path.join(o, "shifts_d2d.json"))
print("written", SUF, shifts)
