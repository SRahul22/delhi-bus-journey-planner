"""Add the planner's blend prediction (SEG_MODE=blend, median for the first BLEND_S s then mean) to the validation
rides (fold4.parquet, days 19-24), using lookup tables rebuilt from days 1-18 exactly as
model/v3/scripts/build_rides.py does. Checks itself by reproducing fold4's base_med.

  python eval/val_blend.py [--blend-s 600] [--lookup-days 1-18]
Output: E:\\hackathon-work\\v2\\eval\\val_blend.parquet (row-aligned with fold4.parquet: base_med_check, base_blend<S>)
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(r"E:\hackathon-work")
OUT = Path(os.environ.get("V2_EVAL_OUT", r"E:\hackathon-work\v2\eval"))
IST = 19800
MIN_N = 3
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend-s", type=int, nargs="+", default=[600])
    ap.add_argument("--lookup-days", default="1-18")
    a = ap.parse_args()
    lo, hi = map(int, a.lookup_days.split("-"))
    days = [f"{d:02d}" for d in range(lo, hi + 1)]
    pat = pd.read_csv(W / "route_patterns.csv").sort_values(["route_id", "stop_sequence"]).reset_index(drop=True)
    routes = pat.route_id.unique()
    rsize = pat.groupby("route_id").size().reindex(routes).to_numpy()
    nseg_r = rsize - 1
    soff = np.concatenate([[0], np.cumsum(nseg_r)])[:-1]
    poff = np.concatenate([[0], np.cumsum(rsize)])[:-1]
    NSEG = int(nseg_r.sum())
    ridx = pd.Series(np.arange(len(routes)), index=routes)
    pdist = pat.dist_m.to_numpy()
    seg_route = np.repeat(np.arange(len(routes)), nseg_r)
    seg_k = np.concatenate([np.arange(n) for n in nseg_r])
    seg_len = np.maximum(pdist[poff[seg_route] + seg_k + 1] - pdist[poff[seg_route] + seg_k], 0)
    dfs = []
    for i, d in enumerate(days):
        z = np.load(W / "pass" / f"pass_{d}.npz")
        df = pd.DataFrame({k: z[k] for k in ("trip", "route", "seq", "t")})
        df["trip"] = df["trip"].astype(np.int64) + i * 10_000_000
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df.route.isin(routes)]
    df["ri"] = ridx.reindex(df.route).to_numpy()
    df["dist"] = pdist[poff[df.ri.to_numpy()] + df.seq.to_numpy()]
    lt = pd.to_datetime(df.t + IST, unit="s")
    df["hour"] = lt.dt.hour.astype(np.int8)
    df["dt"] = (lt.dt.dayofweek >= 5).astype(np.int8)
    df = df.sort_values(["trip", "seq"], kind="stable").reset_index(drop=True)
    log("lookup passings", len(df))
    nx = df.shift(-1)
    m = ((nx.trip == df.trip) & (nx.seq == df.seq + 1)).to_numpy()
    s = pd.DataFrame({"g": (soff[df.ri.to_numpy()[m]] + df.seq.to_numpy()[m]).astype(np.int64),
                      "dt": df.dt.to_numpy()[m], "hour": df.hour.to_numpy()[m],
                      "dur": (nx.t.to_numpy()[m] - df.t.to_numpy()[m]), "dd": (nx.dist.to_numpy()[m] - df.dist.to_numpy()[m])})
    good = (s.dur >= 0) & ((s.dur > 0) | (s.dd < 30)) & (s.dd <= 19.4 * s.dur + 20) & (s.dur <= np.maximum(1800, s.dd / 1.0))
    s = s[good]
    del df, nx
    gb = s.groupby(["g", "dt", "hour"]).dur
    agg = pd.concat([gb.median().rename("med"), gb.mean().rename("mean"), gb.size().rename("n")], axis=1).reset_index()
    shape = (NSEG, 2, 24)
    MED = np.full(shape, np.nan, np.float32); MEAN = MED.copy()
    ix = (agg.g.to_numpy(), agg.dt.to_numpy(), agg.hour.to_numpy())
    ok = agg.n.to_numpy() >= MIN_N
    oix = tuple(x[ok] for x in ix)
    MED[oix] = agg.med.to_numpy()[ok]; MEAN[oix] = agg["mean"].to_numpy()[ok]
    sp = s.groupby(["dt", "hour"])[["dd", "dur"]].sum()
    SPEED = np.full((2, 24), 17 / 3.6, np.float32)
    for (dt_, h_), r_ in sp.iterrows():
        if r_.dur > 0:
            SPEED[int(dt_), int(h_)] = np.clip(r_.dd / r_.dur, 2, 15)
    for A, col in ((MED, "med"), (MEAN, "mean")):
        w = agg.assign(w=agg[col] * agg.n)
        gd = w.groupby(["g", "dt"])[["w", "n"]].sum(); vd = np.full((NSEG, 2), np.nan, np.float32)
        vd[gd.index.get_level_values(0), gd.index.get_level_values(1)] = (gd.w / gd.n).to_numpy()
        ga = w.groupby("g")[["w", "n"]].sum(); va = np.full(NSEG, np.nan, np.float32)
        va[ga.index.to_numpy()] = (ga.w / ga.n).to_numpy()
        miss = np.isnan(A)
        fill = np.where(np.isfinite(vd)[:, :, None], vd[:, :, None], va[:, None, None])
        fill = np.where(np.isfinite(fill), fill, seg_len[:, None, None] / SPEED[None, :, :])
        A[miss] = np.broadcast_to(fill, shape)[miss]
    log("lookup built")
    R = pd.read_parquet(W / "v3" / "fold4.parquet", columns=["route_id", "board_seq", "n_stops", "t0", "dt", "base_med"])
    ri = ridx.reindex(R.route_id).to_numpy()
    g = soff[ri] + R.board_seq.to_numpy()
    steps = R.n_stops.to_numpy(); dtv = R.dt.to_numpy()
    lt = pd.to_datetime(R.t0 + IST, unit="s")
    tsec = (lt.dt.hour * 3600 + lt.dt.minute * 60 + lt.dt.second).to_numpy().astype(np.float64)
    smed = np.zeros(len(R)); sbl = {b: np.zeros(len(R)) for b in a.blend_s}
    for k in range(int(steps.max())):
        i = np.flatnonzero(steps > k)
        if not len(i):
            break
        gg = g[i] + k; d_ = dtv[i]
        h = ((tsec[i] + smed[i]) // 3600 % 24).astype(np.int64)
        smed[i] += MED[gg, d_, h]
        for b, acc in sbl.items():
            hb = ((tsec[i] + acc[i]) // 3600 % 24).astype(np.int64)
            acc[i] += np.where(acc[i] >= b, MEAN[gg, d_, hb], MED[gg, d_, hb])
    rel = np.abs(smed - R.base_med.to_numpy()) / R.base_med.to_numpy()
    log("check vs fold4 base_med: median rel diff", float(np.median(rel)), "share > 1%", float((rel > 0.01).mean()))
    out = pd.DataFrame({"base_med_check": smed, **{f"base_blend{b}": v for b, v in sbl.items()}})
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "val_blend.parquet", index=False)
    log("wrote", OUT / "val_blend.parquet")


if __name__ == "__main__":
    main()
