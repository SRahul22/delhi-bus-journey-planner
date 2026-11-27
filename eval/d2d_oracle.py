"""Noise floor for door-to-door error: a 'perfect historical model' on the config-C common subset (n = 1,996).

For every scored journey, keep the itinerary and the walking, and replace
  * each wait (first and transfer) with the oracle mean wait: the mean wait of an arrival uniform in +-30 min around
    the rider's real arrival at that stop, computed from that same day's bus visits with the same acceptance rules
    as realized_journeys.py (bus of the planned route_id that really goes on to the alighting stop);
  * each ride with the oracle typical ride: the median ride of all accepted buses of that (route, board, alight)
    leaving the boarding stop in the same IST clock hour, over days 19-24 (the realized day included).
Decomposition: oracle waits + our planned rides; our planned waits + oracle rides.
Second, stricter-to-beat variant ("typical"): the oracle wait is the mean over days 19-24 of the same expected wait
(same stop, route, alighting stop and clock time), i.e. what a perfect day-agnostic historical model could know. The
same-day oracle knows when the next bus of a sparse route actually came (for gaps over 30 min it equals the real wait).
  python eval/d2d_oracle.py   -> E:\\hackathon-work\\v2\\analysis\\oracle.csv, oracle.json
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

import d2d_common as DC
import realized_journeys as RJ

DAYS = list(range(19, 25))


def work(args):
    d, waitq, cells = args
    V = DC.Valid(RJ.DayVisits(d))
    ep = RJ.day_epoch(d)
    # q = (route, board, alight, time of day s); evaluated on this day
    waits = {q: V.expected_wait(q[0], q[1], q[2], ep + q[3]) for q in waitq}
    rides = {}
    for c in cells:                                           # c = (route, board, alight, hour)
        bt, btl, ta = V.trips(c[0], c[1], c[2], ep + c[3] * 3600, ep + c[3] * 3600 + 3599)
        rides[c] = (ta - btl).tolist()
    return d, waits, rides


def main():
    C = DC.common_C()
    J = []
    for r in C.to_dict("records"):
        legs, walk = DC.journey_legs(r)
        J.append((r, legs, walk))
    waitq = {d: set() for d in DAYS}; cells = set()
    for r, legs, _ in J:
        for l in legs:
            l["tod"] = l["t_arr"] - RJ.day_epoch(int(r["day"]))
            for d in DAYS:
                waitq[d].add((l["route_id"], l["board"], l["alight"], l["tod"]))
            h = int((l["tdep"] + 19800) % 86400 // 3600)
            l["cell"] = (l["route_id"], l["board"], l["alight"], h)
            cells.add(l["cell"])
    RJ.log("journeys", len(J), "cells", len(cells))
    with Pool(6) as pool:
        res = pool.map(work, [(d, sorted(waitq[d]), sorted(cells)) for d in DAYS])
    OW, RIDES = {}, {}
    for d, w, rd in res:
        OW.update({(d,) + k: v for k, v in w.items()})
        for c, v in rd.items():
            RIDES.setdefault(c, []).extend(v)
    RJ.log("oracle computed")
    rows = []
    for r, legs, walk in J:
        d = int(r["day"])
        ow = [OW[(d, l["route_id"], l["board"], l["alight"], l["tod"])] for l in legs]
        tw = [np.nanmean([OW[(dd, l["route_id"], l["board"], l["alight"], l["tod"])] for dd in DAYS]) for l in legs]
        orides = [float(np.median(RIDES[l["cell"]])) if RIDES[l["cell"]] else np.nan for l in legs]
        rw = [l["real_wait"] for l in legs]; rr = [l["real_ride"] for l in legs]
        pw = [l["plan_wait"] for l in legs]; pr = [l["plan_ride"] for l in legs]
        rows.append(dict(key=r["key"], day=d, n_bus=len(legs), real_total=r["real_total_s"], plan_total=r["plan_total_s"],
                         walk=walk, real_check=walk + sum(rw) + sum(rr), plan_check=walk + sum(pw) + sum(pr),
                         typ_wait1=tw[0], typ_twait=sum(tw[1:]),
                         oracle_wait1=ow[0], real_wait1=rw[0], plan_wait1=pw[0],
                         oracle_twait=sum(ow[1:]), real_twait=sum(rw[1:]), plan_twait=sum(pw[1:]),
                         oracle_ride=sum(orides), real_ride=sum(rr), plan_ride=sum(pr),
                         n_ride_cell=min(len(RIDES[l["cell"]]) for l in legs)))
    O = pd.DataFrame(rows)
    O["oracle_total"] = O.walk + O.oracle_wait1 + O.oracle_twait + O.oracle_ride
    O["typ_total"] = O.walk + O.typ_wait1 + O.typ_twait + O.oracle_ride
    O["twait_pride"] = O.walk + O.typ_wait1 + O.typ_twait + O.plan_ride
    O["owait_pride"] = O.walk + O.oracle_wait1 + O.oracle_twait + O.plan_ride
    O["pwait_oride"] = O.walk + O.plan_wait1 + O.plan_twait + O.oracle_ride
    O["owait_rride"] = O.walk + O.oracle_wait1 + O.oracle_twait + O.real_ride   # waits only: rides known exactly
    O["rwait_oride"] = O.walk + O.real_wait1 + O.real_twait + O.oracle_ride     # rides only: waits known exactly
    O.to_csv(DC.OUT / "oracle.csv", index=False)
    ok = O[["oracle_total", "owait_pride", "pwait_oride", "typ_total"]].notna().all(axis=1)
    S = O[ok]
    out = dict(n_all=len(O), n_ok=int(ok.sum()), check_real_max_abs=float((O.real_check - O.real_total).abs().max()),
               check_plan_median_abs=float((O.plan_check - O.plan_total).abs().median()))
    for nm, col in [("C_historical", "plan_total"), ("oracle", "oracle_total"), ("oracle_typical", "typ_total"),
                    ("typical_wait_our_ride", "twait_pride"), ("oracle_wait_our_ride", "owait_pride"),
                    ("our_wait_oracle_ride", "pwait_oride"), ("oracle_wait_exact_ride", "owait_rride"),
                    ("exact_wait_oracle_ride", "rwait_oride")]:
        out[nm] = DC.score(S[col], S.real_total)
    for nm, a, b in [("wait1", "oracle_wait1", "real_wait1"), ("twait", "oracle_twait", "real_twait"),
                     ("ride", "oracle_ride", "real_ride"), ("typ_wait1", "typ_wait1", "real_wait1"), ("plan_wait1", "plan_wait1", "real_wait1"),
                     ("plan_twait", "plan_twait", "real_twait"), ("plan_ride", "plan_ride", "real_ride")]:
        out["bias_" + nm] = float(np.median(S[a] - S[b]))
    out["oracle_ride_ape_legsum"] = DC.score(S.oracle_ride, S.real_ride)
    out["plan_ride_ape_legsum"] = DC.score(S.plan_ride, S.real_ride)
    json.dump(out, open(DC.OUT / "oracle.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    RJ.log("DONE")


if __name__ == "__main__":
    main()

