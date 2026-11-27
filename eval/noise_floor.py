"""T0.3 noise floor from all bus traversals of the evaluation rides' stop pairs.

The 600k test rides are random (trip, board, alight) draws, so almost no (route, board, alight, hour) cell has
>= 10 of them (only 14 cells on the validation set). The noise floor is therefore measured on every GPS trip of
the same week that passed both stops: for a sample of ride keys (route, board_seq, alight_seq), all map-matched
traversals on the set's days are collected (pass_DD.npz), grouped by (route, board, alight, board hour), and in
cells with >= 10 traversals each traversal's error against its cell median is taken.
This is the error of an oracle that knows the exact same-week median for that stop pair and hour.

  python eval/noise_floor.py --set test [--keys 50000]   # days 25-31 (measurement of the data, no model)
  python eval/noise_floor.py --set val                    # days 19-24
Output: E:\\hackathon-work\\v2\\eval\\noise_floor_<set>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402

W = Path(os.environ.get("WORK_DIR", r"E:\hackathon-work"))
OUT = Path(os.environ.get("V2_EVAL_OUT") or W / "v2" / "eval")
IST = 19800
DAYS = {"test": range(25, 32), "val": range(19, 25)}
RIDES = {"test": W / "v3" / "test.parquet", "val": W / "v3" / "fold4.parquet"}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["val", "test"], required=True)
    ap.add_argument("--keys", type=int, default=50000, help="number of distinct ride keys sampled (seed 2026)")
    ap.add_argument("--min-n", type=int, default=10)
    a = ap.parse_args()
    R = pd.read_parquet(RIDES[a.set], columns=["route_id", "board_seq", "alight_seq", "actual_s"])
    K = R[["route_id", "board_seq", "alight_seq"]].drop_duplicates()
    if len(K) > a.keys:
        K = K.sample(a.keys, random_state=2026)
    K = K.reset_index(drop=True)
    K["key"] = np.arange(len(K))
    log("ride keys", len(K))
    dfs = []
    for i, d in enumerate(DAYS[a.set]):
        z = np.load(W / "pass" / f"pass_{d:02d}.npz")
        df = pd.DataFrame({c: z[c] for c in ("trip", "route", "seq", "t")})
        df = df[df.route.isin(K.route_id.unique())]
        df["trip"] = df.trip.astype(np.int64) + i * 10_000_000
        dfs.append(df)
    P = pd.concat(dfs, ignore_index=True)
    log("passings on key routes", len(P))
    b = K.merge(P, left_on=["route_id", "board_seq"], right_on=["route", "seq"])[["key", "route_id", "alight_seq", "trip", "t"]]
    b = b.rename(columns={"t": "tb"})
    log("board matches", len(b))
    x = b.merge(P[["trip", "seq", "t"]], left_on=["trip", "alight_seq"], right_on=["trip", "seq"])
    x["dur"] = x.t - x.tb
    x = x[(x.dur > 0) & (x.dur <= 3 * 3600)]
    x["hour"] = ((x.tb + IST) // 3600 % 24).astype(np.int16)
    log("traversals", len(x))
    g = x.groupby(["key", "hour"]).dur
    x["n_cell"] = g.transform("size")
    x = x[x.n_cell >= a.min_n].copy()
    x["med"] = x.groupby(["key", "hour"]).dur.transform("median")
    x["ape"] = (x.med - x.dur).abs() / x.dur
    x["bucket"] = M.bucket(x.dur.to_numpy())
    res = dict(set=a.set, days=list(DAYS[a.set]), definition="per (route, board, alight, board hour) cell with >= "
               f"{a.min_n} same-period GPS traversals: median |traversal - cell median| / traversal",
               n_keys_sampled=int(len(K)), n_cells=int(x.groupby(["key", "hour"]).ngroups), n_traversals=int(len(x)),
               keys_with_cells=int(x.key.nunique()), median_ape=float(x.ape.median()), within20=float((x.ape <= 0.2).mean()),
               by_bucket={bk: dict(n=int((x.bucket == bk).sum()), median_ape=float(x.ape[x.bucket == bk].median()),
                                   within20=float((x.ape[x.bucket == bk] <= 0.2).mean()))
                          for bk in M.BUCKET_NAMES if (x.bucket == bk).any()})
    # ride-weighted variant: weight each cell like the ride sample (one value per key-hour cell = its median ape)
    cell = x.groupby(["key", "hour"]).ape.median()
    res["median_of_cell_medians"] = float(cell.median())
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"noise_floor_{a.set}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
