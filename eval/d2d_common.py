"""Shared helpers for the d2d floor / bias / live analysis (eval/d2d_oracle.py, d2d_calib.py, d2d_live.py)."""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402
import realized_journeys as RJ  # noqa: E402

W = Path(r"E:\hackathon-work\v2")
OUT = W / "analysis"
CFG = dict(A=W / "eval" / "realized_val.csv", B=W / "rerun" / "B" / "realized_val.csv",
           C=W / "rerun" / "C" / "realized_val.csv", C24=W / "rerun" / "C24" / "realized_val.csv")


def load(p):
    R = pd.read_csv(p, dtype={"plan_route_ids": str, "plan_stops": str, "kind": str, "real_legs": str})
    R["key"] = R.od_id.astype(str) + "|" + R.day.astype(str) + "|" + R.depart_s.astype(str)
    R["scored"] = R.scored.astype(bool)
    return R


def common_C():
    """Config C rows on the common subset (scored in A, B, C and C24; n = 1,996 in rerun_val_ABC.md)."""
    D = {k: load(v) for k, v in CFG.items()}
    com = set.intersection(*[set(v.key[v.scored]) for v in D.values()])
    C = D["C"]
    return C[C.key.isin(com)].reset_index(drop=True)


def is_peak(day, depart_s):
    wd = datetime(2026, 10, int(day)).weekday() < 5 and int(day) not in RJ.HOLIDAYS
    h = int(depart_s) // 3600
    return bool(wd and (8 <= h <= 10 or 17 <= h <= 20))


def journey_legs(r):
    """Bus legs with plan values and the realized timeline.
    -> (legs, walk_total) legs: dict(route_id, board, alight, plan_wait, plan_ride, t_arr (real arrival at the stop),
    real_wait, real_ride, tdep); walk_total = all walking (access + transfer + egress) as replayed."""
    pl = json.loads(r["plan_legs_json"])
    rl = str(r["real_legs"]).split(";")
    t = RJ.day_epoch(int(r["day"])) + float(r["depart_s"])
    legs, k, walk = [], 0, 0.0
    for l in pl:
        if l["mode"] == "walk":
            t += l["dur"]; walk += l["dur"]
            continue
        w, ride = map(float, rl[k].split(":")[1].split("+"))
        legs.append(dict(route_id=int(l["route_id"]), board=int(l["board"]), alight=int(l["alight"]),
                         plan_wait=float(l.get("wait", np.nan)), plan_ride=float(l.get("ride", np.nan)),
                         t_arr=t, real_wait=w, real_ride=ride, tdep=t + w))
        t += w + ride; k += 1
    return legs, walk


def score(pred, actual):
    e = M.error(pred, actual); b = M.bias(pred, actual)
    return dict(n=e["n"], median_ape=e["median_ape"], within20=e["within20"], median_bias_s=b["median_bias_s"])


def fmt(name, s, note=""):
    return (f"| {name} | {100 * s['median_ape']:.1f}% | {100 * s['within20']:.1f}% | {s['median_bias_s']:+.0f} | {s['n']} |"
            + (f" {note} |" if note is not None else ""))


class Valid:
    """Buses that the realization (realized_journeys.DayVisits.board) would accept for a leg, on one day."""

    def __init__(self, V):
        self.V = V; self.cache = {}

    def grp(self, stop, route):
        k = (stop, route)
        if k not in self.cache:
            self.cache[k] = self.V.grp(stop, route, False)
        return self.cache[k]

    def trips(self, route, board, alight, t0, t1):
        """Visits at `board` with t in [t0, t1] whose vehicle really goes on to `alight` -> arrays bt, btl, ta."""
        B = self.grp(board, route); A = self.grp(alight, route)
        if B is None or A is None:
            return np.empty(0), np.empty(0), np.empty(0)
        bt, btl, bv = B; at, _, av = A
        out = []
        for k in np.flatnonzero((bt >= t0) & (bt <= t1)):
            v = bv[k]
            m = np.flatnonzero((av == v) & (at > bt[k]) & (at <= btl[k] + RJ.MAX_RIDE))
            if not len(m):
                continue
            ta = at[m[0]]
            if ((bv == v) & (bt > btl[k] + RJ.RETURN_GAP) & (bt < ta)).any():
                continue
            out.append((bt[k], btl[k], ta))
        if not out:
            return np.empty(0), np.empty(0), np.empty(0)
        a = np.array(out, dtype=float)
        return a[:, 0], a[:, 1], a[:, 2]

    def expected_wait(self, route, board, alight, t_mid, half=1800, step=20):
        """Mean realized wait for an arrival uniform in [t_mid-half, t_mid+half] (same rule as board(): the first
        accepted visit in t order with t_last >= arrival and t <= arrival + 2 h; wait = max(t_last, u) - u)."""
        bt, btl, _ = self.trips(route, board, alight, t_mid - half - 6 * 3600, t_mid + half + RJ.MAX_WAIT)
        if not len(bt):
            return np.nan
        ws = []
        for u in np.arange(t_mid - half, t_mid + half + 1, step):
            m = np.flatnonzero((btl >= u) & (bt <= u + RJ.MAX_WAIT))
            if len(m):
                ws.append(max(btl[m[0]], u) - u)
        return float(np.mean(ws)) if ws else np.nan

