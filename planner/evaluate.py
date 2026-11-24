"""Evaluate predicted travel times.

1) Holdout rides (from the model agent):  holdout_trips.csv
     route_id,board_stop,alight_stop,depart_ts_ist,actual_s
   Compares predicted in-vehicle ride time for (a) the GPS model and (b) the schedule baseline
   (GTFS x prior hour factor) against the observed ride time.

2) Google Maps comparison:  pairs CSV
     from_lat,from_lng,to_lat,to_lng,depart,google_s
   Runs the full planner and compares the fastest option's door-to-door duration with google_s.

Usage:
  .venv/bin/python evaluate.py                       # holdout at $MODEL_DIR/holdout_trips.csv if it exists
  .venv/bin/python evaluate.py --holdout path.csv
  .venv/bin/python evaluate.py --pairs google_pairs.csv
Writes per-row predictions to eval_out/*.csv.
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from router import Planner
from ttmodel import MODEL_DIR, TravelTimeModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_out")


def metrics(pred, actual):
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    m = np.isfinite(pred) & np.isfinite(actual) & (actual > 0)
    if not m.any():
        return dict(n=0)
    ape = np.abs(pred[m] - actual[m]) / actual[m]
    err = pred[m] - actual[m]
    return dict(n=int(m.sum()), median_ape_pct=round(float(np.median(ape) * 100), 1),
                mean_ape_pct=round(float(np.mean(ape) * 100), 1),
                within_20pct=round(float(np.mean(ape <= 0.2) * 100), 1),
                median_bias_s=round(float(np.median(err)), 1),
                mae_s=round(float(np.mean(np.abs(err))), 1))


def by_group(df, key, cols):
    rows = []
    for k, g in df.groupby(key):
        r = {key: k}
        for c in cols:
            mm = metrics(g[c], g.actual_s)
            r[f"{c}_n"] = mm.get("n", 0)
            r[f"{c}_mdape"] = mm.get("median_ape_pct")
            r[f"{c}_w20"] = mm.get("within_20pct")
        rows.append(r)
    return pd.DataFrame(rows)


def eval_holdout(planner, path):
    df = pd.read_csv(path, dtype={"route_id": str, "board_stop": str, "alight_stop": str})
    print(f"holdout rows: {len(df)}  ({path})")
    gps = planner.model
    base = TravelTimeModel(planner.net, model_dir="/nonexistent")
    sidx = {s: i for i, s in enumerate(planner.stop_id)}
    pos_maps = {}
    pg, pb, hours, n_stops = [], [], [], []
    for r in df.itertuples(index=False):
        ri = planner.route_by_id.get(str(r.route_id))
        a, b = sidx.get(str(r.board_stop)), sidx.get(str(r.alight_stop))
        ts = pd.Timestamp(r.depart_ts_ist)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
        hours.append(ts.hour)
        if ri is None or a is None or b is None:
            pg.append(np.nan); pb.append(np.nan); n_stops.append(np.nan); continue
        pm = pos_maps.get(ri)
        if pm is None:
            pm = {}
            for p, s in enumerate(planner.route_stops[ri]):
                pm.setdefault(s, []).append(p)
            pos_maps[ri] = pm
        i = j = None
        for pi in pm.get(a, []):
            later = [q for q in pm.get(b, []) if q > pi]
            if later:
                i, j = pi, later[0]
                break
        if i is None:
            pg.append(np.nan); pb.append(np.nan); n_stops.append(np.nan); continue
        t = ts.hour * 3600 + ts.minute * 60 + ts.second
        dt = 1 if ts.weekday() >= 5 else 0
        pg.append(gps.ride(ri, i, j, t, dt)[0])
        pb.append(base.ride(ri, i, j, t, dt)[0])
        n_stops.append(j - i)
    df["pred_model_s"] = pg
    df["pred_schedule_s"] = pb
    df["hour"] = hours
    df["n_stops"] = n_stops
    cov = np.isfinite(df.pred_model_s).mean() * 100
    print(f"coverage (route+stops found in canonical pattern): {cov:.1f}%")
    print("model    :", metrics(df.pred_model_s, df.actual_s))
    print("schedule :", metrics(df.pred_schedule_s, df.actual_s))
    long = df.actual_s > 3600
    print("model, rides > 1 h :", metrics(df.pred_model_s[long], df.actual_s[long]))
    band = pd.cut(df.actual_s, [0, 600, 1800, 3600, 1e9], labels=["<10m", "10-30m", "30-60m", ">60m"])
    for b, g in df.groupby(band, observed=True):
        mm = metrics(g.pred_model_s, g.actual_s)
        bias = float(np.median((g.pred_model_s - g.actual_s) / g.actual_s) * 100)
        print(f"  {b:>6}: n={mm['n']} MdAPE={mm['median_ape_pct']}% w20={mm['within_20pct']}% bias={bias:+.1f}% ({mm['median_bias_s']} s)")
    print("model mode:", gps.stats.get("seg_mode"))
    bh = by_group(df, "hour", ["pred_model_s", "pred_schedule_s"])
    with pd.option_context("display.width", 200, "display.max_rows", 50):
        print(bh.to_string(index=False))
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "holdout_predictions.csv"), index=False)
    bh.to_csv(os.path.join(OUT, "holdout_by_hour.csv"), index=False)
    return df


def eval_pairs(planner, path):
    df = pd.read_csv(path)
    print(f"pairs: {len(df)}  ({path})")
    pred, pred85, routes, ms = [], [], [], []
    for r in df.itertuples(index=False):
        dep = datetime.fromisoformat(str(r.depart)) if isinstance(r.depart, str) and r.depart else None
        res = planner.plan(r.from_lat, r.from_lng, r.to_lat, r.to_lng, dep)
        opts = res["options"]
        best = min(opts, key=lambda o: o["duration_s"]) if opts else None
        pred.append(best["duration_s"] if best else np.nan)
        pred85.append(best["duration_p85_s"] if best else np.nan)
        routes.append("+".join(best["routes"]) if best else "")
        ms.append(res["compute_ms"])
    df["pred_s"] = pred
    df["pred_p85_s"] = pred85
    df["routes"] = routes
    df["compute_ms"] = ms
    df["actual_s"] = df.google_s
    print("planner vs google:", metrics(df.pred_s, df.google_s))
    if "depart" in df:
        df["hour"] = pd.to_datetime(df.depart).dt.hour
        print(by_group(df, "hour", ["pred_s"]).to_string(index=False))
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "pairs_predictions.csv"), index=False)
    print(df[["from_lat", "from_lng", "to_lat", "to_lng", "depart", "google_s", "pred_s", "routes"]].to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=os.path.join(MODEL_DIR, "holdout_trips.csv"))
    ap.add_argument("--pairs", default=None)
    args = ap.parse_args()
    planner = Planner()
    did = False
    if args.holdout and os.path.exists(args.holdout):
        eval_holdout(planner, args.holdout)
        did = True
    if args.pairs:
        eval_pairs(planner, args.pairs)
        did = True
    if not did:
        print(f"nothing to evaluate: {args.holdout} not found and no --pairs given")


if __name__ == "__main__":
    main()
