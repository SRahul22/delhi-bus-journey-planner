"""Observed (GPS) bus ride times for the test pairs on the held-out days 25-31 Oct.

Runs on the Mac, ships a small job to Windows over the SSH master connection, reads the data agent's
per-day stop-passing files E:\\hackathon-work\\pass\\pass_DD.npz (route, trip, veh, seq, stop, t) and,
for every direct test pair, finds held-out weekday trips on the pair's ticket routes that pass the
boarding stop and later the alighting stop, departing within +-1 h of the pair's departure hour.
Output: eval/actual_rides.csv (pair_id, n_obs, actual_median_s, actual_p25_s, actual_p75_s, days).

  ../planner/.venv/bin/python actual_rides.py [--days 25,26,27,28,29,30,31]
"""
import argparse, io, json, os, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GTFS = os.path.join(HERE, "..", "data", "gtfs")
SSH = ["ssh", "-o", f"ControlPath={os.path.expanduser('~')}/.ssh/cm-%r@%h:%p", "-o", "BatchMode=yes",
       "-o", "LogLevel=ERROR", "<user>@<windows-host>", "python -"]

REMOTE = r'''
import json, sys, datetime, numpy as np
J = json.loads(PAYLOAD)
rows = []
for d in J["days"]:
    p = r"E:\hackathon-work\pass\pass_%s.npz" % d
    try:
        z = np.load(p)
    except Exception as e:
        print("skip", d, e, file=sys.stderr); continue
    wd = datetime.date(2026, 10, int(d)).weekday()
    if wd >= 5:
        continue
    route, trip, stop, t = z["route"], z["trip"], z["stop"], z["t"]
    for q in J["pairs"]:
        m = np.isin(route, q["rids"]) & ((stop == q["a"]) | (stop == q["b"]))
        if not m.any():
            continue
        r_, tr, st, tt = route[m], trip[m], stop[m], t[m]
        key = r_.astype(np.int64) * 10**7 + tr
        for k in np.unique(key):
            s = key == k
            ta = tt[s & (st == q["a"])]; tb = tt[s & (st == q["b"])]
            if len(ta) and len(tb):
                ta0 = ta.min(); later = tb[tb > ta0]
                if len(later):
                    h = (datetime.datetime.utcfromtimestamp(ta0) + datetime.timedelta(hours=5, minutes=30)).hour
                    if abs(h - q["hour"]) <= 1:
                        rows.append((q["pid"], d, h, float(later.min() - ta0)))
print(json.dumps(rows))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="25,26,27,28,29,30,31")
    args = ap.parse_args()
    P = pd.read_csv(os.path.join(HERE, "test_pairs.csv"), dtype={"from_stop_id": str, "to_stop_id": str})
    routes = pd.read_csv(os.path.join(GTFS, "routes.txt"), dtype={"route_id": str})
    s2r = routes.groupby(routes.route_short_name.str.upper()).route_id.apply(list).to_dict()
    pairs = []
    for r in P[P.kind == "direct"].itertuples():
        rids = sorted({int(x) for rt in str(r.routes).split("|") for x in s2r.get(rt.upper(), []) if x.isdigit()})
        if rids and r.from_stop_id.isdigit() and r.to_stop_id.isdigit():
            pairs.append(dict(pid=r.pair_id, rids=rids, a=int(r.from_stop_id), b=int(r.to_stop_id), hour=int(r.depart[11:13])))
    payload = json.dumps(dict(days=args.days.split(","), pairs=pairs))
    code = "PAYLOAD = " + json.dumps(payload) + "\n" + REMOTE
    out = subprocess.run(SSH, input=code, capture_output=True, text=True, timeout=1800)
    if out.returncode:
        raise SystemExit(out.stderr)
    print(out.stderr.strip())
    obs = pd.DataFrame(json.loads(out.stdout.strip().splitlines()[-1]), columns=["pair_id", "day", "hour", "ride_s"])
    obs = obs[obs.ride_s.between(30, 4 * 3600)]
    g = obs.groupby("pair_id").agg(n_obs=("ride_s", "size"), actual_median_s=("ride_s", "median"),
                                   actual_p25_s=("ride_s", lambda x: x.quantile(.25)),
                                   actual_p75_s=("ride_s", lambda x: x.quantile(.75)),
                                   days=("day", lambda x: ",".join(sorted(set(x))))).round(0).reset_index()
    g.to_csv(os.path.join(HERE, "actual_rides.csv"), index=False)
    print(f"{len(g)}/{len(pairs)} direct pairs with observed rides; days {args.days}")
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
