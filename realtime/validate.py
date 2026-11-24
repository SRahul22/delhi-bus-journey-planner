"""Validate the real-time layer on the replay day (held-out 28 Oct 2026, a Wednesday).

Replays the day in 5-minute steps. At each evaluation time T (every 30 min, 07:00-21:00) it freezes the live
snapshot (only GPS up to T is used), then scores rides and waits that happen *after* T against what the GPS
later shows:

  rides:  board in (T, T+30 min] ("lead 0-30") or (T+30, T+60] ("lead 30-60"); actual board->alight time
          from the same projection pipeline; predictions = historical model vs historical x live ratios.
  waits:  for random (route, stop) pairs with a live bus approaching at T: actual wait = next observed
          passing after T - T; predictions = historical headway/2 vs live next-bus ETA.

Usage (from repo root):  planner/.venv/bin/python realtime/validate.py [replay.npz]
Writes realtime/results/validation.json and prints a summary.
"""
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(REPO, "planner"), HERE]

from feed import ReplayFeed, parse_ist, DEFAULT_REPLAY  # noqa: E402
from live_state import LiveState, Snapshot, sod  # noqa: E402
from rt_model import RealtimeTravelTimeModel  # noqa: E402
from ttmodel import TravelTimeModel  # noqa: E402

DAY = "2026-10-28"
rng = np.random.default_rng(0)


def main():
    t00 = time.time()
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPLAY
    net = pickle.load(open(os.path.join(REPO, "planner", "cache", "network.pkl"), "rb"))
    base = TravelTimeModel(net)
    ls = LiveState(net, base)
    ls.keep_history = True
    rt = RealtimeTravelTimeModel(base, ls)
    feed = ReplayFeed(path)
    start, end = parse_ist(f"{DAY}T06:00"), parse_ist(f"{DAY}T22:00")
    evals = [parse_ist(f"{DAY}T07:00") + 1800 * i for i in range(29)]  # 07:00 .. 21:00
    snaps, wait_q = {}, []
    t = start
    while t < end:
        ls.ingest(feed.window(t, t + 300))
        t += 300
        if t in evals:
            snap = ls.refresh(t)
            snaps[t] = Snapshot(t, snap.logr, snap.n_seg, snap.n_area, snap.city, snap.n_obs, {})
            # sample waits: (ri, k) with a live bus approaching
            cand = [(ri, k) for ri, vs in snap.vehicles.items() for k in range(1, len(ls.G[int(net["routes"][ri]["route_id"])][8]))
                    if any(nk <= k for _, _, nk, _, _ in vs)]
            for ri, k in [cand[i] for i in rng.choice(len(cand), min(400, len(cand)), replace=False)]:
                arr = ls.arrivals(ri, k, snap)
                if arr:
                    wait_q.append((t, ri, k, arr[0][0] - t, base.wait_s(ri, k, sod(t), 0)))
    print(f"replayed in {time.time()-t00:.0f}s", ls.stats)

    # ---- ground truth from all passings of the day
    P = ls.passings()
    order = np.lexsort((P["t"], P["trk"]))
    ri_, k_, t_, trk = P["ri"][order], P["k"][order], P["t"][order], P["trk"][order]
    cuts = np.flatnonzero(np.diff(trk)) + 1
    rides = []
    for a, b in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(trk)]])):
        if b - a < 4:
            continue
        for _ in range(3):
            i, j = sorted(rng.choice(b - a, 2, replace=False))
            if k_[a + j] - k_[a + i] < 3:
                continue
            dur = t_[a + j] - t_[a + i]
            if dur < 120 or dur > 3 * 3600:
                continue
            rides.append((int(ri_[a]), int(k_[a + i]), int(k_[a + j]), float(t_[a + i]), float(dur)))
    rides = np.array(rides)
    print("truth rides", len(rides))

    # first passing after T per (ri,k), for waits
    pk = P["ri"].astype(np.int64) * 1000 + P["k"]
    o2 = np.lexsort((P["t"], pk)); pk_s, pt_s = pk[o2], P["t"][o2]

    def next_pass(ri, k, T):
        key = ri * 1000 + k
        lo, hi = np.searchsorted(pk_s, key, "left"), np.searchsorted(pk_s, key, "right")
        j = lo + np.searchsorted(pt_s[lo:hi], T, "right")
        return pt_s[j] if j < hi else None

    res = {"replay_day": DAY, "replay_file_mb": round(os.path.getsize(path) / 1e6, 1), "stats": ls.stats}
    for lead_name, lo, hi in (("lead_0_30min", 0, 1800), ("lead_30_60min", 1800, 3600)):
        rows = []
        for T, snap in snaps.items():
            m = (rides[:, 3] > T + lo) & (rides[:, 3] <= T + hi)
            ls.snapshot = snap
            for ri, i, j, tb, dur in rides[m]:
                ri, i, j = int(ri), int(i), int(j)
                h = base.ride(ri, i, j, sod(tb), 0)[0]
                lv = rt.ride(ri, i, j, sod(tb), 0)[0]
                # area/city-only variant: ignore the segment's own observations
                rows.append((dur, h, lv))
        r = np.array(rows)
        ape_h = np.abs(r[:, 1] - r[:, 0]) / r[:, 0]
        ape_l = np.abs(r[:, 2] - r[:, 0]) / r[:, 0]
        res[lead_name] = dict(
            n=len(r),
            historical=dict(median_ape=round(float(np.median(ape_h)) * 100, 1), within_20pct=round(float((ape_h <= .2).mean()) * 100, 1),
                            median_bias_pct=round(float(np.median((r[:, 1] - r[:, 0]) / r[:, 0])) * 100, 1)),
            live=dict(median_ape=round(float(np.median(ape_l)) * 100, 1), within_20pct=round(float((ape_l <= .2).mean()) * 100, 1),
                      median_bias_pct=round(float(np.median((r[:, 2] - r[:, 0]) / r[:, 0])) * 100, 1)),
            live_better_share=round(float((ape_l < ape_h).mean()) * 100, 1))
    # by hour (lead 0-30)
    byh = {}
    for T, snap in snaps.items():
        m = (rides[:, 3] > T) & (rides[:, 3] <= T + 1800)
        ls.snapshot = snap
        e_h, e_l = [], []
        for ri, i, j, tb, dur in rides[m]:
            ri, i, j = int(ri), int(i), int(j)
            e_h.append(abs(base.ride(ri, i, j, sod(tb), 0)[0] - dur) / dur)
            e_l.append(abs(rt.ride(ri, i, j, sod(tb), 0)[0] - dur) / dur)
        hh = time.strftime("%H:%M", time.gmtime(T + 19800))
        byh[hh] = dict(n=len(e_h), hist=round(float(np.median(e_h)) * 100, 1), live=round(float(np.median(e_l)) * 100, 1),
                       city_ratio=round(snap.city[0], 3))
    res["by_eval_time_lead_0_30"] = byh

    # waits
    W = []
    for T, ri, k, eta, hw in wait_q:
        nxt = next_pass(ri, k, T)
        if nxt is None or nxt - T > 3600:
            continue
        W.append((nxt - T, hw, eta))
    W = np.array(W)
    res["wait_next_bus"] = dict(
        n=len(W),
        actual_median_s=round(float(np.median(W[:, 0]))),
        historical_headway_half=dict(median_abs_err_s=round(float(np.median(np.abs(W[:, 1] - W[:, 0])))),
                                     within_2min=round(float((np.abs(W[:, 1] - W[:, 0]) <= 120).mean()) * 100, 1)),
        live_eta=dict(median_abs_err_s=round(float(np.median(np.abs(W[:, 2] - W[:, 0])))),
                      within_2min=round(float((np.abs(W[:, 2] - W[:, 0]) <= 120).mean()) * 100, 1)))
    res["model_source"] = base.source
    res["runtime_s"] = round(time.time() - t00)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "validation.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "by_eval_time_lead_0_30"}, indent=1))
    print("by eval time:", {k: (v["hist"], v["live"]) for k, v in byh.items()})


if __name__ == "__main__":
    main()
