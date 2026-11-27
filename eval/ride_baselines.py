"""T0.3 + T0.4 (eval part): ride-level error tables, GTFS schedule baseline, blend column, noise floor.

  python eval/ride_baselines.py --set test   # the existing 600k held-out rides, days 25-31 (v1 measurement only)
  python eval/ride_baselines.py --set val    # fold-4 rides, days 19-24, features from the 1-18 lookup

Models compared per ride (all v1, nothing fitted here):
  gtfs       : GTFS scheduled in-vehicle time on the same route pattern, board -> alight
               (planner network sched_seg, median per trip-start hour; hour = ride start hour)
  sum_median : sum of segment medians (lookup table)
  sum_mean   : sum of segment means
  blend      : planner SEG_MODE=blend, median for the first 600 s then mean (test: base_blend600; val: from
               val_blend.parquet written by eval/val_blend.py)
  xgb        : final XGBoost (model/v3/xgb_final.json) on sum_median, clipped [0.5, 2.5]
               (test set only: the final model was refit on 1-24 so it has seen the validation rides)
  v1 band    : p50 = blend (test) or sum_median (val), p85 = sum of segment p85s (what v1 shows as the range)

Outputs in E:\\hackathon-work\\v2\\eval\\:  rides_<set>_metrics.csv (model x bucket x peak),
  rides_<set>_noise_floor.json, rides_<set>_summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402

V3 = Path(os.environ.get("WORK_DIR", r"E:\hackathon-work")) / "v3"
OUT = Path(os.environ.get("V2_EVAL_OUT") or Path(os.environ.get("WORK_DIR", r"E:\hackathon-work")) / "v2" / "eval")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", r"E:\hackathon-project\model"))
NET = Path(os.environ.get("NET_PATH", r"E:\hackathon-project\planner\cache\network.pkl"))
HOLIDAYS = {2, 20}  # October 2026 day-of-month; Gandhi Jayanti, Dussehra (AGENTS.md section 4)
IST = 19800
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def load_rides(which: str) -> pd.DataFrame:
    if which == "test":
        return pd.read_parquet(V3 / "test.parquet")
    if which == "val":
        R = pd.read_parquet(V3 / "fold4.parquet")
        vb = OUT / "val_blend.parquet"  # eval/val_blend.py: blend from the 1-18 tables (base_med reproduced exactly)
        if vb.exists():
            B = pd.read_parquet(vb)
            if len(B) == len(R) and np.allclose(B.base_med_check, R.base_med, rtol=1e-3):
                R["base_blend600"] = B.base_blend600.to_numpy()
        return R
    raise ValueError(which)


def add_calendar(R: pd.DataFrame) -> pd.DataFrame:
    R["hour"] = np.floor(R.hour_f).astype(np.int16)
    R["weekday"] = (R.dow < 5) & ~R.day.isin(list(HOLIDAYS))
    R["peak"] = M.is_peak(R.hour_f, R.weekday)
    return R


def gtfs_schedule(R: pd.DataFrame, net_path: Path = NET) -> np.ndarray:
    """GTFS scheduled seconds board->alight on the ride's route pattern; NaN when the planner network has a
    different stop pattern for the route (ride's stops not at those positions)."""
    with open(net_path, "rb") as f:
        net = pickle.load(f)
    stop_id = net["stops"].stop_id.astype(str).to_numpy()
    out = np.full(len(R), np.nan)
    by_route = R.groupby("route_id").indices
    for r in net["routes"]:
        try:
            rid = int(r["route_id"])
        except ValueError:
            continue
        idx = by_route.get(rid)
        if idx is None:
            continue
        pat = stop_id[r["stops"]]
        cum = np.vstack([np.zeros((1, 24), np.float64), np.cumsum(r["sched_seg"].astype(np.float64), axis=0)])
        bs = R.board_seq.to_numpy()[idx]; as_ = R.alight_seq.to_numpy()[idx]
        ok = (as_ < len(pat))
        ok[ok] &= (pat[bs[ok]] == R.board_stop.to_numpy()[idx][ok].astype(str)) & \
                  (pat[as_[ok]] == R.alight_stop.to_numpy()[idx][ok].astype(str))
        h = R.hour.to_numpy()[idx] % 24
        v = cum[np.minimum(as_, len(pat) - 1), h] - cum[np.minimum(bs, len(pat) - 1), h]
        out[idx[ok]] = v[ok]
    return out


def xgb_pred(R: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb
    meta = json.load(open(MODEL_DIR / "v3" / "xgb_final_meta.json"))
    b = xgb.Booster()
    b.load_model(str(MODEL_DIR / "v3" / "xgb_final.json"))
    d = R
    X = pd.DataFrame(index=d.index)
    X["log_base_med"] = np.log(d.base_med); X["skew"] = d.base_mean / d.base_med; X["n_stops"] = d.n_stops
    X["log_dist"] = np.log1p(d.dist_m); X["base_speed"] = d.dist_m / d.base_med; X["hour_f"] = d.hour_f
    X["weekend"] = d.dt; X["route_len_m"] = d.route_len_m; X["route_nstops"] = d.route_nstops
    X["route_speed"] = d.route_speed; X["log_route_obs"] = np.log1p(d.route_obs); X["route_cov"] = d.route_cov
    X["route_skew"] = d.route_skew; X["board_pos"] = d.board_pos
    X["b_lat"] = d.b_lat; X["b_lon"] = d.b_lon; X["a_lat"] = d.a_lat; X["a_lon"] = d.a_lon
    X["fb_frac"] = d.fb_frac; X["mean_logn"] = d.mean_logn
    f = np.exp(b.predict(xgb.DMatrix(X[meta["features"]])))
    lo, hi = meta.get("clip", [0.5, 2.5])
    return d.base_med.to_numpy() * np.clip(f, lo, hi)


def compute(which: str, with_xgb: bool = True):
    R = add_calendar(load_rides(which))
    log("rides", which, len(R), "days", sorted(R.day.unique().tolist()))
    preds = {}
    R["gtfs"] = gtfs_schedule(R)
    log("gtfs matched", round(float(np.isfinite(R.gtfs).mean()), 4))
    preds["gtfs"] = R.gtfs.to_numpy()
    preds["sum_median"] = R.base_med.to_numpy()
    preds["sum_mean"] = R.base_mean.to_numpy()
    if "base_blend600" in R:
        preds["blend"] = R.base_blend600.to_numpy()
    if with_xgb and which == "test":
        preds["xgb"] = xgb_pred(R)
        log("xgb done")
    p50_band = preds.get("blend", preds["sum_median"])
    band_name = "v1_band(" + ("blend" if "blend" in preds else "sum_median") + ",sum_p85)"
    preds[band_name] = p50_band
    p85 = {band_name: R.base_p85.to_numpy()}
    bands = [band_name]
    # ml agent's validation predictions (1-18 models), row-aligned with fold4.parquet; used only if they line up
    mlf = Path(os.environ.get("V2_ML_DIR") or Path(os.environ.get("WORK_DIR", r"E:\hackathon-work")) / "v2" / "ml") / "band_val_preds.parquet"
    if which == "val" and mlf.exists():
        B = pd.read_parquet(mlf)
        if len(B) == len(R) and np.allclose(B.actual_s, R.actual_s) and np.allclose(B.base_med, R.base_med):
            if "p50_l1_18" in B:
                preds["xgb_l1_1-18 (ml)"] = B.p50_l1_18.to_numpy()
            if {"q50_18", "p85_c18"} <= set(B.columns):
                nm = "v2_band_ml(q50,p85_c18)"
                preds[nm] = R.base_med.to_numpy() * np.exp(B.q50_18.to_numpy())
                p85[nm] = np.maximum(B.p85_c18.to_numpy(), preds[nm])
                bands.append(nm)
            log("ml val predictions added from", mlf)
        else:
            log("ml val predictions not row-aligned with fold4; skipped")
    a = R.actual_s.to_numpy()
    # restrict every model to rides where all predictions exist (gtfs may be missing) -> same n everywhere
    common = np.all([np.isfinite(v) & (v > 0) for v in preds.values()], axis=0)
    log("rides with all predictions", int(common.sum()), "of", len(R))
    tab = M.by_bucket_peak(a[common], R.peak.to_numpy()[common], {k: v[common] for k, v in preds.items()},
                           {k: v[common] for k, v in p85.items()})
    tabp = M.by_bucket_peak(a[common], R.peak.to_numpy()[common], {k: preds[k][common] for k in p85},
                            {k: v[common] for k, v in p85.items()}, bucket_by="pred")
    tabp["bucket_by"] = "pred"
    tab["bucket_by"] = "actual"
    tab = pd.concat([tab, tabp], ignore_index=True)
    nf, nfd = M.noise_floor(R)
    nfb = {b: dict(n=int((nfd.bucket == b).sum()), median_ape=float(nfd.ape[nfd.bucket == b].median()))
           for b in M.BUCKET_NAMES if (nfd.bucket == b).any()}
    nf["by_bucket"] = nfb
    nf["cell"] = "route x board stop x alight stop x hour, >= 10 rides"
    nf["n_all_rides"] = int(len(R))
    return R, tab, nf, dict(n_rides=int(len(R)), n_common=int(common.sum()), gtfs_match=float(np.isfinite(R.gtfs).mean()),
                            band=band_name, bands=bands)


def summary_md(which, tab, nf, info) -> str:
    L = [f"### Ride error, {which} set ({info['n_common']:,} rides with all predictions, of {info['n_rides']:,})", "",
         "Median absolute % error (share within ±20%) by actual ride length; n per bucket in the header.", ""]
    tab_act = tab[tab.bucket_by == "actual"]
    allp = tab_act[tab_act.peak == "all"]
    models = list(dict.fromkeys(tab_act.model))
    bks = M.BUCKET_NAMES + ["all"]
    ns = {b: int(allp[(allp.bucket == b) & (allp.model == models[0])].n.iloc[0]) for b in bks}
    L.append("| Model | " + " | ".join(f"{b} min (n={ns[b]:,})" if b != "all" else f"All (n={ns[b]:,})" for b in bks) + " |")
    L.append("|---" * (len(bks) + 1) + "|")
    for m in models:
        cells = []
        for b in bks:
            r = allp[(allp.model == m) & (allp.bucket == b)].iloc[0]
            cells.append(f"{100 * r.median_ape:.2f}% ({100 * r.within20:.0f}%)")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "Peak vs off-peak (all lengths): " + "; ".join(
        f"{m}: peak {100 * tab_act[(tab_act.model == m) & (tab_act.bucket == 'all') & (tab_act.peak == 'peak')].median_ape.iloc[0]:.2f}% / "
        f"off-peak {100 * tab_act[(tab_act.model == m) & (tab_act.bucket == 'all') & (tab_act.peak == 'offpeak')].median_ape.iloc[0]:.2f}%"
        for m in models), ""]
    L.append("Bias (median, all rides): " + "; ".join(
        f"{m} {tab_act[(tab_act.model == m) & (tab_act.bucket == 'all') & (tab_act.peak == 'all')].median_bias_s.iloc[0]:+.0f} s"
        for m in models))
    L += ["", "Ride bands: coverage (actual <= p85) / width / pinball@0.85 / share under p50, p85, p90. "
              "Buckets here are by the band's own predicted p50 (what a user sees; bucketing by the actual length "
              "conditions on the outcome and pushes coverage down in the >60 bucket; that version is in "
              "rides_<set>_metrics.csv with bucket_by=actual). G1 asks 82-88% coverage in every bucket:", "",
          "| Band | Bucket | Peak | n | Coverage | Width | Pinball (s) | <p50 | <p85 | <p90 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for bn in info.get("bands", [info["band"]]):
        band = tab[(tab.model == bn) & (tab.bucket_by == "pred")]
        for _, r in band.iterrows():
            if r.n == 0:
                continue
            L.append(f"| {bn} | {r.bucket} | {r.peak} | {r.n:,} | {100 * r.coverage:.1f}% | {100 * r.width:.1f}% | {r.pinball85:.1f} | "
                     f"{100 * r.under_p50:.1f}% | {100 * r.under_p85:.1f}% | {100 * r.under_p90:.1f}% |")
    L += ["", f"In-sample noise floor on the sampled rides only (underpowered; see noise_floor_<set>.json for the real one) ({nf['cell']}): {nf['n_rides']:,} rides in {nf['n_cells']:,} cells "
              f"({100 * nf['n_rides'] / nf['n_all_rides']:.1f}% of rides): median error vs own cell median "
              f"{100 * nf['median_ape']:.2f}%, within ±20% {100 * nf['within20']:.1f}%. By bucket: "
          + ", ".join(f"{b} {100 * v['median_ape']:.2f}% (n={v['n']:,})" for b, v in nf["by_bucket"].items()), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["val", "test"], required=True)
    ap.add_argument("--no-xgb", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    R, tab, nf, info = compute(a.set, with_xgb=not a.no_xgb)
    tab.to_csv(OUT / f"rides_{a.set}_metrics.csv", index=False)
    json.dump({**nf, **info}, open(OUT / f"rides_{a.set}_noise_floor.json", "w"), indent=1)
    md = summary_md(a.set, tab, nf, info)
    (OUT / f"rides_{a.set}_summary.md").write_text(md, encoding="utf-8")
    print(md)
    log("DONE")


if __name__ == "__main__":
    main()
