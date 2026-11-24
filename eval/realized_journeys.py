"""T0.2 Realized door-to-door journeys (contract 6, with amendment A1).

For each OD and departure, ask the v1 planner (in-process) for its recommended option, then replay that itinerary
against what the buses really did that day:
  * walk legs take the planner's walk time (1.2 m/s, detour 1.3, as planned);
  * at each boarding stop, take the first real bus of the leg's route (stop_visits_DD.npz from the data agent:
    a visit = a bus of that route_id within 60 m of the stop) that is still at the stop when the rider arrives
    (visit t_last >= arrival) and that really goes on to the alighting stop: the same vehicle's next visit at the
    alighting stop comes within 3 h and before that vehicle comes back to the boarding stop (this rejects buses
    passing in the opposite direction, which also show up within 60 m);
  * the rider leaves the boarding stop when that bus leaves (t_last of the visit) and arrives when the same vehicle
    first reaches the alighting stop (t of that visit);
  * repeat for each transfer; the egress walk is added at the end.
A journey is unscored when any leg has no such bus within 2 h (reason given). usable_share (share of the route's
raw trips that map-matched that day, from passing_index_DD.npz) is kept only as a diagnostic column (A1).

Sets: the 60 ticket pairs at their departure clock time on every non-holiday weekday of the set; 5,000 random
OD-times (seed 2026, GTFS stops on planner routes 2-20 km apart, departure 06:00-22:00 uniform, day uniform over
the set's days).  Days: val = 19-24 Oct -> realized_val.csv; test = 25-31 Oct -> realized_test.csv.

  python eval/realized_journeys.py --set val [--random 5000] [--workers 3]
Columns (contract 6): od_id, sample, kind, day, depart_s, plan_*, real_wait_s, real_ride_s, real_transfer_s,
real_total_s, scored (+ diagnostics). wait = first boarding wait; transfer = transfer walks + later waits;
ride = in-vehicle time; total = door to door including access/egress walks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = Path(os.environ.get("V2_DATA_DIR", r"E:\hackathon-work\v2\data"))
OUT = Path(os.environ.get("V2_EVAL_OUT", r"E:\hackathon-work\v2\eval"))
NET = os.environ.get("NET_PATH", r"E:\hackathon-project\planner\cache\network.pkl")
os.environ.setdefault("MODEL_DIR", r"E:\hackathon-project\model")
IST = timezone(timedelta(hours=5, minutes=30))
DAYS = {"val": list(range(19, 25)), "test": list(range(25, 32))}
HOLIDAYS = {2, 20}
MAX_WAIT = 2 * 3600
MAX_RIDE = 3 * 3600
RETURN_GAP = 300  # a later visit of the same vehicle at the boarding stop this long after it left = it came back
t00 = time.time()


def log(*a):
    print(f"[{time.time() - t00:7.1f}s]", *a, flush=True)


def day_epoch(d):
    return int(datetime(2026, 10, d, tzinfo=IST).timestamp())


# ----------------------------------------------------------------------------- OD sets
def ticket_pairs(days):
    P = pd.read_csv(HERE / "test_pairs.csv")
    rows = []
    wdays = [d for d in days if datetime(2026, 10, d).weekday() < 5 and d not in HOLIDAYS]
    for _, r in P.iterrows():
        hh, mm = map(int, r.depart[11:16].split(":"))
        for d in wdays:
            rows.append(dict(od_id=r.pair_id, sample="pairs", kind=r.kind, day=d, depart_s=hh * 3600 + mm * 60,
                             from_lat=r.from_lat, from_lng=r.from_lng, to_lat=r.to_lat, to_lng=r.to_lng, dist_km=r.dist_km))
    return rows


def random_ods(days, n, seed=2026):
    import pickle
    with open(NET, "rb") as f:
        net = pickle.load(f)
    served = np.unique(np.concatenate([r["stops"] for r in net["routes"]]))
    st = net["stops"].iloc[served]
    lat, lon = st.lat.to_numpy(), st.lon.to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        a, b = rng.integers(0, len(st), 2)
        dk = np.hypot((lat[a] - lat[b]) * 111.32, (lon[a] - lon[b]) * 111.32 * np.cos(np.radians(28.6)))
        if not (2 <= dk <= 20):
            continue
        dep = int(rng.integers(6 * 3600, 22 * 3600))
        d = int(rng.choice(days))
        rows.append(dict(od_id=f"R{len(rows):04d}", sample="random", kind="", day=d, depart_s=dep,
                         from_lat=float(lat[a]), from_lng=float(lon[a]), to_lat=float(lat[b]), to_lng=float(lon[b]),
                         dist_km=round(float(dk), 3)))
    return rows


# ----------------------------------------------------------------------------- planning (worker processes)
_P = None


def _init():
    global _P
    sys.path.insert(0, str(REPO / "planner"))
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        from router import Planner
        _P = Planner(net_path=NET)


def _plan(q):
    dep = datetime(2026, 10, q["day"], tzinfo=IST) + timedelta(seconds=q["depart_s"])
    try:
        res = _P.plan(q["from_lat"], q["from_lng"], q["to_lat"], q["to_lng"], depart=dep, max_options=3)
    except Exception as e:  # pragma: no cover
        return dict(q, plan_error=repr(e))
    opts = res["options"]
    if not opts:
        return dict(q, plan_error="no_options")
    o = opts[0]  # recommended (first after the planner's generalized-cost sort)
    legs = []
    for l in o["legs"]:
        if l["mode"] == "walk":
            legs.append(dict(mode="walk", dur=float(l["duration_s"])))
        else:
            legs.append(dict(mode="bus", route_id=int(l["route_id"]), board=int(l["from_stop"]["stop_id"]),
                             alight=int(l["to_stop"]["stop_id"]), wait=float(l["wait_s"]), ride=float(l["ride_s"]),
                             ride85=float(l["ride_p85_s"]), route=l["route"]))
    return dict(q, plan_total_s=o["duration_s"], plan_p85_s=o["duration_p85_s"], plan_labels="|".join(o.get("labels", [])),
                plan_legs=legs, compute_ms=res["compute_ms"])


# ----------------------------------------------------------------------------- realization
def route_families():
    """route_id -> list of route_ids with the same route number (UP/DOWN stripped), from the planner network.
    About 17% of GPS trips run in the opposite direction of their route_id label, so a bus serving the leg can carry
    the paired route_id; the alighting check below verifies its direction."""
    import pickle
    import re
    with open(NET, "rb") as f:
        net = pickle.load(f)
    name = {}
    for r in net["routes"]:
        try:
            rid = int(r["route_id"])
        except ValueError:
            continue
        name[rid] = re.sub(r"[\s-]*(UP|DOWN|Up|Down)$", "", str(r["short_name"]).strip()).strip().upper()
    by = {}
    for rid, nm in name.items():
        by.setdefault(nm, []).append(rid)
    return {rid: sorted(by[nm]) for rid, nm in name.items()}


class DayVisits:
    def __init__(self, d, families=None):
        self.fam = families or {}
        z = np.load(DATA / f"stop_visits_{d:02d}.npz")
        self.t = z["t"]; self.tl = z["t_last"]; self.veh = z["veh"]
        # data agent: files with `params` are the final (densified) visits; without = preliminary raw-ping version
        self.params = "/".join(str(x) for x in z["params"].tolist()) if "params" in z.files else "prelim"
        self.g = {(int(s), int(r)): (int(a), int(n)) for s, r, a, n in zip(z["g_stop"], z["g_route"], z["g_start"], z["g_len"])}
        self.us = {}
        pi = DATA / f"passing_index_{d:02d}.npz"
        if pi.exists():
            zi = np.load(pi)
            self.us = dict(zip(zi["us_route"].tolist(), zi["us_share"].tolist()))

    def grp(self, stop, route, family=False):
        rids = self.fam.get(route, [route]) if family else [route]
        sls = [self.g.get((stop, r)) for r in rids]
        sls = [slice(a[0], a[0] + a[1]) for a in sls if a is not None]
        if not sls:
            return None
        t = np.concatenate([self.t[s] for s in sls]); tl = np.concatenate([self.tl[s] for s in sls])
        v = np.concatenate([self.veh[s] for s in sls])
        o = np.argsort(t, kind="stable")
        return t[o], tl[o], v[o]

    def board(self, route, board, alight, t_arr, family=False):
        """-> (t_depart_board, t_arrive_alight, veh) or (None, reason)."""
        B = self.grp(board, route, family)
        if B is None:
            return None, "no_visits_board"
        A = self.grp(alight, route, family)
        if A is None:
            return None, "no_visits_alight"
        bt, btl, bv = B
        at, _, av = A
        cand = np.flatnonzero((btl >= t_arr) & (bt <= t_arr + MAX_WAIT))
        for k in cand:
            v = bv[k]
            m = np.flatnonzero((av == v) & (at > bt[k]) & (at <= btl[k] + MAX_RIDE))
            if not len(m):
                continue
            ta = at[m[0]]
            back = (bv == v) & (bt > btl[k] + RETURN_GAP) & (bt < ta)
            if back.any():
                continue
            return (max(float(btl[k]), float(t_arr)), float(ta), int(v)), None
        return None, "no_bus_2h" if len(cand) else "no_visit_after_arrival"


def realize(p, V: DayVisits, ep, family=False):
    out = dict(real_wait_s=np.nan, real_ride_s=np.nan, real_transfer_s=np.nan, real_total_s=np.nan,
               scored=False, unscored_reason="", min_usable_share=np.nan, real_legs="")
    if "plan_legs" not in p:
        out["unscored_reason"] = p.get("plan_error", "no_plan")
        return out
    t = ep + p["depart_s"]
    wait0 = 0.0; ride = 0.0; transfer = 0.0; nb = 0; us = []; detail = []
    walk_pending = 0.0
    for l in p["plan_legs"]:
        if l["mode"] == "walk":
            t += l["dur"]; walk_pending += l["dur"]
            continue
        us.append(V.us.get(l["route_id"], np.nan))
        r, why = V.board(l["route_id"], l["board"], l["alight"], t, family)
        if r is None:
            out["unscored_reason"] = f"leg{nb + 1}:{why}"
            out["min_usable_share"] = np.nanmin(us) if np.isfinite(us).any() else np.nan
            return out
        tdep, tarr, veh = r
        w = tdep - t
        if nb == 0:
            wait0 = w
        else:
            transfer += walk_pending + w
        walk_pending = 0.0
        ride += tarr - tdep
        detail.append(f"{l['route_id']}:{int(w)}+{int(tarr - tdep)}")
        t = tarr; nb += 1
    total = t - (ep + p["depart_s"])
    out.update(real_wait_s=wait0, real_ride_s=ride, real_transfer_s=transfer, real_total_s=total, scored=nb > 0,
               unscored_reason="" if nb > 0 else "walk_only", real_legs=";".join(detail),
               min_usable_share=np.nanmin(us) if len(us) and np.isfinite(us).any() else np.nan)
    return out


def plan_columns(p):
    legs = p.get("plan_legs", [])
    bus = [l for l in legs if l["mode"] == "bus"]
    walks = [l["dur"] for l in legs if l["mode"] == "walk"]
    # transfer = walks between bus legs + waits of legs 2..n
    transfer, seen_bus, pend = 0.0, 0, 0.0
    for l in legs:
        if l["mode"] == "walk":
            pend += l["dur"]
        else:
            if seen_bus:
                transfer += pend + l["wait"]
            pend = 0.0; seen_bus += 1
    return dict(plan_wait_s=bus[0]["wait"] if bus else 0.0, plan_ride_s=sum(l["ride"] for l in bus),
                plan_ride_p85_s=sum(l["ride85"] for l in bus), plan_transfer_s=transfer, plan_walk_s=sum(walks),
                plan_n_bus=len(bus), plan_routes="|".join(l["route"] for l in bus),
                plan_route_ids="|".join(str(l["route_id"]) for l in bus),
                plan_stops="|".join(f"{l['board']}>{l['alight']}" for l in bus))


PLAN_COLS = ["plan_total_s", "plan_p85_s", "plan_labels", "compute_ms", "plan_error", "plan_wait_s", "plan_ride_s",
             "plan_ride_p85_s", "plan_transfer_s", "plan_walk_s", "plan_n_bus", "plan_routes", "plan_route_ids", "plan_stops"]
Q_COLS = ["od_id", "sample", "kind", "day", "depart_s", "from_lat", "from_lng", "to_lat", "to_lng", "dist_km"]


def plans_from_csv(path):
    """Rebuild plans from an earlier realized_<set>.csv (first version had no plan cache): bus legs from
    plan_route_ids/plan_stops, walk legs recomputed exactly as router.walk_leg does (straight line x 1.3 / 1.2 m/s,
    rounded, skipped under 1 m) between origin, each leg's board/alight stop and the destination."""
    import pickle
    with open(NET, "rb") as f:
        net = pickle.load(f)
    st = net["stops"]
    ll = dict(zip(st.stop_id.astype(str), zip(st.lat.to_numpy(), st.lon.to_numpy())))
    m_lat, m_lon = 111_320.0, 111_320.0 * np.cos(np.radians(28.6))

    def walk(a, b):
        dm = float(np.hypot((a[0] - b[0]) * m_lat, (a[1] - b[1]) * m_lon)) * 1.3
        return None if dm < 1 else dict(mode="walk", dur=float(round(dm / 1.2)))

    C = pd.read_csv(path, dtype={"plan_route_ids": str, "plan_stops": str, "kind": str})
    plans = []
    for r in C.to_dict("records"):
        p = {k: r.get(k) for k in Q_COLS}
        p["kind"] = "" if pd.isna(p["kind"]) else p["kind"]
        p["_plan_cols"] = {k: r.get(k) for k in PLAN_COLS if k in r}
        if isinstance(r.get("plan_error"), str) and r["plan_error"]:
            p["plan_error"] = r["plan_error"]
            plans.append(p)
            continue
        legs, cur = [], (r["from_lat"], r["from_lng"])
        rids = [] if pd.isna(r.get("plan_route_ids")) else str(r["plan_route_ids"]).split("|")
        sts = [] if pd.isna(r.get("plan_stops")) else str(r["plan_stops"]).split("|")
        for rid, bs in zip(rids, sts):
            b, a_ = bs.split(">")
            w = walk(cur, ll[b])
            if w:
                legs.append(w)
            legs.append(dict(mode="bus", route_id=int(float(rid)), board=int(b), alight=int(a_)))
            cur = ll[a_]
        w = walk(cur, (r["to_lat"], r["to_lng"]))
        if w:
            legs.append(w)
        p["plan_legs"] = legs
        plans.append(p)
    return plans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["val", "test"], required=True)
    ap.add_argument("--random", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pairs-only", action="store_true")
    ap.add_argument("--reuse-plans", action="store_true", help="reuse plans_<set>.pkl from a previous run")
    ap.add_argument("--plans-from-csv", help="rebuild plans from an earlier realized csv instead of planning")
    ap.add_argument("--family", type=int, default=0, help="0 (default, used for realized_val.csv v1): only buses whose "
                    "GPS route_id is the planned route_id; 1: any bus of the route number (UP/DOWN pair) that really "
                    "goes board->alight. The other variant is always written with a strict_/family_ prefix.")
    a = ap.parse_args()
    days = DAYS[a.set]
    missing = [d for d in days if not (DATA / f"stop_visits_{d:02d}.npz").exists()]
    if missing:
        sys.exit(f"stop_visits missing for days {missing}")
    cache = OUT / f"plans_{a.set}.pkl"
    import pickle
    if a.plans_from_csv:
        plans = plans_from_csv(a.plans_from_csv)
        log("plans rebuilt from", a.plans_from_csv, len(plans))
    elif a.reuse_plans and cache.exists():
        plans = pickle.load(open(cache, "rb"))
        log("reused plans", cache, len(plans))
    else:
        Q = ticket_pairs(days) + ([] if a.pairs_only else random_ods(days, a.random))
        log(f"{a.set}: {len(Q)} OD-times ({sum(q['sample'] == 'pairs' for q in Q)} pair-days)")
        with Pool(a.workers, initializer=_init) as pool:
            plans = []
            for i, p in enumerate(pool.imap(_plan, Q, chunksize=8)):
                plans.append(p)
                if (i + 1) % 500 == 0:
                    log("planned", i + 1)
        log("planned all")
        OUT.mkdir(parents=True, exist_ok=True)
        pickle.dump(plans, open(cache, "wb"))
    fams = route_families()
    rows = []
    for d in days:
        idx = [i for i, p in enumerate(plans) if p["day"] == d]
        if not idx:
            continue
        V = DayVisits(d, fams)
        ep = day_epoch(d)
        for i in idx:
            p = plans[i]
            row = {k: v for k, v in p.items() if k not in ("plan_legs", "_plan_cols")}
            row.update(p["_plan_cols"] if "_plan_cols" in p else plan_columns(p))
            row["plan_legs_json"] = json.dumps(p.get("plan_legs", []))
            row["visits_params"] = V.params
            strict = realize(p, V, ep, family=False)
            fam = realize(p, V, ep, family=True)
            # primary columns: strict (only buses whose GPS route_id equals the planned route_id) unless --family 1;
            # family = any bus with this route number that really goes board -> alight
            row.update(fam if a.family else strict)
            alt = strict if a.family else fam
            row.update({("strict_" if a.family else "family_") + k: alt[k]
                        for k in ("real_wait_s", "real_ride_s", "real_transfer_s", "real_total_s", "scored", "unscored_reason")})
            rows.append(row)
        log("realized day", d, len(idx))
    R = pd.DataFrame(rows)
    R["weekday"] = [datetime(2026, 10, d).weekday() < 5 and d not in HOLIDAYS for d in R.day]
    h = R.depart_s // 3600
    R["peak"] = R.weekday & (((h >= 8) & (h <= 10)) | ((h >= 17) & (h <= 20)))
    R["transfers"] = (R.plan_n_bus - 1).clip(lower=0)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"realized_{a.set}.csv"
    tmp = out.with_suffix(".tmp")
    R.to_csv(tmp, index=False)
    os.replace(tmp, out)
    log("wrote", out, len(R))
    # coverage summary only (no error numbers printed for the test set)
    pr = R[R["sample"] == "pairs"]
    per_pair = pr.groupby("od_id").scored.sum()
    s = dict(set=a.set, family=a.family, n=len(R), scored_share=float(R.scored.mean()),
             pairs_scored_on_3plus_days=int((per_pair >= 3).sum()), pairs_total=int(per_pair.size),
             transfer_pairs_scored_on_3plus_days=int((pr[pr.kind == "transfer"].groupby("od_id").scored.sum() >= 3).sum()),
             unscored_reasons=R.loc[~R.scored, "unscored_reason"].str.replace(r"^leg\d:", "", regex=True).value_counts().to_dict())
    json.dump(s, open(OUT / f"realized_{a.set}_coverage.json", "w"), indent=1)
    print(json.dumps(s, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
