"""Live-mode door to door on 22 Oct (the live tuning day; part of validation).

Journeys: config-C plans of random OD-times on 22 Oct, 06:00-21:00 (eval/d2d_plan_extra.py live22), replayed against
22 Oct bus visits exactly as realized_journeys.py does. Predictions on the same scored journeys:
  C_hist      the config-C plan (historical waits and rides)
  live_first  C with the FIRST wait replaced by the live next-bus ETA (V2_LIVE=kf) from the 22 Oct replay with the
              clock at the departure time (only pings up to then): first live bus of the leg's route with ETA at or
              after the rider reaches the stop; capped at 2x the planned wait and falling back to the planned wait
              when no live bus is predicted (same rule as realtime/rt_model.py wait_s).
  live_xfer_dep   + transfer waits from the SAME departure-time snapshot, at the expected arrival at the transfer
                  stop (live first wait + planned ride + transfer walk); fallback planned wait.
  live_xfer_enroute + transfer waits from a snapshot taken at that expected arrival time (an app refreshed on the
                  way); fallback planned wait.
Snapshots on a 60 s grid (clock floored, so never later than the query time).
  python eval/d2d_live.py  -> E:\\hackathon-work\\v2\\analysis\\live22.csv, live22.json
"""
import os
import sys

os.environ.update(V2_LIVE="kf", V3_MODE="xgb", V3_MIN_RIDE_S="3600", V2_WAIT="pooled", REALTIME="0",
                  POOLED_INDEX=r"E:\hackathon-work\v2\planner\pooled_index_visits_1-18.npz",
                  V2_HEADWAYS=r"E:\hackathon-work\v2\data\stop_headways_v2_1_18.csv",
                  MODEL_DIR=r"E:\hackathon-project\model")
import heapq  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "planner"), str(HERE.parent / "realtime"), str(HERE)]
import d2d_common as DC  # noqa: E402
import realized_journeys as RJ  # noqa: E402
from feed import ReplayFeed  # noqa: E402
from live_state import LiveState  # noqa: E402
from ttmodel import TravelTimeModel  # noqa: E402

DAY = 22
REPLAY = r"E:\hackathon-work\v2\realtime\replay_22.npz"
STEP = 60


def main():
    t00 = time.time()
    src = DC.OUT / "realized_live22.csv"
    while not src.exists():
        time.sleep(20)
    time.sleep(15)  # let the writer finish
    R = DC.load(src)
    R = R[R.scored & (R.day == DAY)].reset_index(drop=True)
    net = pickle.load(open(RJ.NET, "rb"))
    base = TravelTimeModel(net)
    ls = LiveState(net, base, mode="kf")
    ls.keep_history = True
    feed = ReplayFeed(REPLAY)
    print("live mode", ls.mode, "journeys", len(R), flush=True)
    ep = RJ.day_epoch(DAY)
    J = []
    for r in R.to_dict("records"):
        legs, walk = DC.journey_legs(r)
        pl = json.loads(r["plan_legs_json"])
        # planned timeline: walk before each bus leg (access / transfer walks)
        pre, acc, bi = [], 0.0, 0
        for l in pl:
            if l["mode"] == "walk":
                acc += l["dur"]
            else:
                pre.append(acc); acc = 0.0
        J.append(dict(r=r, legs=legs, pre=pre, T0=ep + float(r["depart_s"]), live=[None] * len(legs),
                      live_dep=[None] * len(legs), src=[""] * len(legs)))
    ev = []  # (snapshot time, seq, journey idx, kind)
    for i, j in enumerate(J):
        heapq.heappush(ev, (int(j["T0"] // STEP * STEP), i, i, "dep"))
    seq = len(J)

    def eta_after(snap, leg, t_at, cap):
        ri = ls.ri_of.get(int(leg["route_id"]))
        if ri is None:
            return None
        k = ls.stop_pos.get((ri, str(leg["board"])))
        if k is None:
            return None
        for eta, _v in ls.arrivals(ri, k, snap, max_eta=4 * 3600):
            if eta >= t_at:
                return float(min(eta - t_at, cap))
        return None

    def expected_arrival(j, upto, waits):
        t = j["T0"]
        for m in range(upto):
            t += j["pre"][m] + waits[m] + j["legs"][m]["plan_ride"]
        return t + j["pre"][upto]

    t = ep + 6 * 3600
    n_ref = 0
    while ev:
        T = ev[0][0]
        if T > t:
            ls.ingest(feed.window(t, T))
            t = T
        snap = ls.refresh(T); n_ref += 1
        while ev and ev[0][0] == T:
            _, _, i, kind = heapq.heappop(ev)
            j = J[i]; legs = j["legs"]
            if kind == "dep":
                w = eta_after(snap, legs[0], j["T0"] + j["pre"][0], 2 * legs[0]["plan_wait"])
                j["src"][0] = "live" if w is not None else "hist"
                j["live"][0] = j["live_dep"][0] = w if w is not None else legs[0]["plan_wait"]
                for m in range(1, len(legs)):   # transfers from the departure snapshot
                    ta = expected_arrival(j, m, j["live_dep"])
                    w = eta_after(snap, legs[m], ta, 2 * legs[m]["plan_wait"])
                    j["live_dep"][m] = w if w is not None else legs[m]["plan_wait"]
                if len(legs) > 1:
                    ta = expected_arrival(j, 1, j["live"])
                    seq += 1; heapq.heappush(ev, (int(max(ta, T) // STEP * STEP), seq, i, "x1"))
            else:
                m = int(kind[1:])
                ta = expected_arrival(j, m, j["live"])
                w = eta_after(snap, legs[m], ta, 2 * legs[m]["plan_wait"])
                j["src"][m] = "live" if w is not None else "hist"
                j["live"][m] = w if w is not None else legs[m]["plan_wait"]
                if m + 1 < len(legs):
                    ta = expected_arrival(j, m + 1, j["live"])
                    seq += 1; heapq.heappush(ev, (int(max(ta, T) // STEP * STEP), seq, i, f"x{m + 1}"))
        if n_ref % 100 == 0:
            print(f"[{time.time() - t00:6.0f}s] clock {(T - ep) / 3600:5.2f} h, refreshes {n_ref}, pending {len(ev)}", flush=True)
    rows = []
    for j in J:
        r, legs = j["r"], j["legs"]
        pw = [l["plan_wait"] for l in legs]
        rows.append(dict(key=r["key"], depart_s=r["depart_s"], n_bus=len(legs), real_total=r["real_total_s"],
                         C_hist=r["plan_total_s"], real_wait1=legs[0]["real_wait"], plan_wait1=pw[0], live_wait1=j["live"][0],
                         first_src=j["src"][0], xfer_src="|".join(j["src"][1:]),
                         live_first=r["plan_total_s"] - pw[0] + j["live"][0],
                         live_xfer_dep=r["plan_total_s"] - sum(pw) + sum(j["live_dep"]),
                         live_xfer_enroute=r["plan_total_s"] - sum(pw) + sum(j["live"])))
    L = pd.DataFrame(rows)
    L.to_csv(DC.OUT / "live22.csv", index=False)
    out = dict(n=len(L), share_first_live=float((L.first_src == "live").mean()), refreshes=n_ref,
               runtime_s=round(time.time() - t00))
    for sub, m in [("all", np.ones(len(L), bool)), ("first_live_available", (L.first_src == "live").to_numpy()),
                   ("direct", (L.n_bus == 1).to_numpy()), ("transfer", (L.n_bus > 1).to_numpy())]:
        S = L[m]
        out[sub] = {c: DC.score(S[c], S.real_total) for c in ("C_hist", "live_first", "live_xfer_dep", "live_xfer_enroute")}
        out[sub]["wait1_hist_abs_err_median"] = float((S.plan_wait1 - S.real_wait1).abs().median())
        out[sub]["wait1_live_abs_err_median"] = float((S.live_wait1 - S.real_wait1).abs().median())
        out[sub]["wait1_hist_bias"] = float((S.plan_wait1 - S.real_wait1).median())
        out[sub]["wait1_live_bias"] = float((S.live_wait1 - S.real_wait1).median())
    json.dump(out, open(DC.OUT / "live22.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()

