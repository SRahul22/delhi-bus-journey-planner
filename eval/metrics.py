"""v2 evaluation metrics (AGENTS.md section 4). Pure numpy/pandas, no I/O.

All functions take arrays of predictions and actuals in seconds.

  error        : median |pred - actual| / actual, share within +-20%          -> error()
  bias         : mean and median of pred - actual, in s and in %              -> bias()
  coverage     : share of actual <= p85                                       -> coverage()
  width        : median (p85 - p50) / p50                                     -> width()
  pinball 0.85 : mean pinball loss of the p85 prediction                      -> pinball()
  calibration  : share of actuals under the predicted 50th/85th/90th pct      -> calibration()
                 (50th = p50, 85th = p85, 90th from the log-normal fitted to the band)
  noise floor  : per (route, board, alight, hour) cell with >= 10 rides,
                 median of |ride - cell median| / ride                        -> noise_floor()

Grouping: bucket (<10, 10-30, 30-60, >60 min, by ACTUAL duration) x peak (weekday 08:00-10:59, 17:00-20:59)
                                                                             -> by_bucket_peak()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = [(0, 600, "<10"), (600, 1800, "10-30"), (1800, 3600, "30-60"), (3600, np.inf, ">60")]
BUCKET_NAMES = [b[2] for b in BUCKETS]
Z85 = 1.0364333894937898  # Phi^-1(0.85)
Z90 = 1.2815515655446004  # Phi^-1(0.90)


def _a(x):
    return np.asarray(x, dtype=np.float64)


def bucket(actual_s):
    """Ride-length bucket label per element (by actual duration)."""
    a = _a(actual_s)
    out = np.empty(a.shape, dtype=object)
    for lo, hi, nm in BUCKETS:
        out[(a >= lo) & (a < hi)] = nm
    return out


def is_peak(hour, weekday):
    """Peak = weekday and hour in [8, 11) or [17, 21). hour may be fractional (hour_f). weekday: bool array
    (True for Mon-Fri and not a holiday)."""
    h = np.floor(_a(hour)).astype(int)
    wd = np.asarray(weekday, dtype=bool)
    return wd & (((h >= 8) & (h <= 10)) | ((h >= 17) & (h <= 20)))


def error(pred, actual):
    p, a = _a(pred), _a(actual)
    ape = np.abs(p - a) / a
    return dict(n=int(len(a)), median_ape=float(np.median(ape)) if len(a) else np.nan,
                within20=float(np.mean(ape <= 0.2)) if len(a) else np.nan)


def bias(pred, actual):
    p, a = _a(pred), _a(actual)
    e = p - a
    if not len(a):
        return dict(mean_bias_s=np.nan, median_bias_s=np.nan, mean_bias_pct=np.nan, median_bias_pct=np.nan)
    r = e / a
    return dict(mean_bias_s=float(e.mean()), median_bias_s=float(np.median(e)),
                mean_bias_pct=float(r.mean()), median_bias_pct=float(np.median(r)))


def coverage(p85, actual):
    a = _a(actual)
    return float(np.mean(a <= _a(p85))) if len(a) else np.nan


def width(p50, p85):
    p50 = _a(p50)
    return float(np.median((_a(p85) - p50) / p50)) if len(p50) else np.nan


def pinball(q_pred, actual, tau=0.85):
    """Mean pinball (quantile) loss in seconds: tau*(a-q) if a>=q else (1-tau)*(q-a)."""
    d = _a(actual) - _a(q_pred)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d))) if len(d) else np.nan


def lognormal_sigma(p50, p85):
    """sigma = ln(p85/p50) / 1.0364 (section 4); clipped at 0."""
    return np.maximum(np.log(_a(p85) / _a(p50)), 0) / Z85


def lognormal_quantile(p50, p85, z):
    return _a(p50) * np.exp(lognormal_sigma(p50, p85) * z)


def calibration(p50, p85, actual):
    """Share of actuals at or under the predicted 50th, 85th and 90th percentiles (90th from the log-normal)."""
    a = _a(actual)
    if not len(a):
        return dict(under_p50=np.nan, under_p85=np.nan, under_p90=np.nan)
    p90 = lognormal_quantile(p50, p85, Z90)
    return dict(under_p50=float(np.mean(a <= _a(p50))), under_p85=float(np.mean(a <= _a(p85))),
                under_p90=float(np.mean(a <= p90)))


def all_metrics(pred, actual, p85=None):
    """Point metrics for pred; band metrics too when p85 is given (pred is then the p50)."""
    out = {**error(pred, actual), **bias(pred, actual)}
    if p85 is not None:
        out.update(coverage=coverage(p85, actual), width=width(pred, p85), pinball85=pinball(p85, actual),
                   **calibration(pred, p85, actual))
    return out


def by_bucket_peak(actual, peak, preds: dict, p85: dict | None = None, margins=True, bucket_by="actual"):
    """Table of all_metrics per (bucket, peak) for each named prediction.
    preds: {name: array}; p85: {name: p85 array} (optional, same names). Returns a long DataFrame with columns
    model, bucket, peak, n, metrics... Rows with bucket='all' / peak='all' are the margins.
    bucket_by: "actual" (ride length actually taken; right for point error) or "pred" (each model's own predicted
    p50; right for band coverage, since conditioning on the actual length biases coverage in the end buckets)."""
    a = _a(actual)
    b_act = bucket(a)
    pk = np.asarray(peak, dtype=bool)
    groups = [(bn, pv) for bn in BUCKET_NAMES for pv in ("peak", "offpeak")]
    if margins:
        groups += [(bn, "all") for bn in BUCKET_NAMES] + [("all", "peak"), ("all", "offpeak"), ("all", "all")]
    rows = []
    for name, p in preds.items():
        p = _a(p)
        q = None if not p85 or name not in p85 else _a(p85[name])
        b = bucket(p) if bucket_by == "pred" else b_act
        for bn, pv in groups:
            m = np.ones(len(a), bool)
            if bn != "all":
                m &= b == bn
            if pv == "peak":
                m &= pk
            elif pv == "offpeak":
                m &= ~pk
            r = all_metrics(p[m], a[m], None if q is None else q[m])
            rows.append(dict(model=name, bucket=bn, peak=pv, **r))
    return pd.DataFrame(rows)


def noise_floor(df, keys=("route_id", "board_stop", "alight_stop", "hour"), actual="actual_s", min_n=10):
    """Noise floor: in each cell with >= min_n rides, the error of each ride against its own cell median.
    Returns (summary dict, per-ride frame with columns ape and bucket). This is the error an oracle that knows
    the true cell median would get, i.e. irreducible ride-to-ride variation at this resolution."""
    d = df[list(keys) + [actual]].copy()
    g = d.groupby(list(keys), sort=False)[actual]
    d["n_cell"] = g.transform("size")
    d = d[d.n_cell >= min_n].copy()
    d["cell_med"] = d.groupby(list(keys), sort=False)[actual].transform("median")
    d["ape"] = np.abs(d.cell_med - d[actual]) / d[actual]
    d["bucket"] = bucket(d[actual].to_numpy())
    summ = dict(n_rides=int(len(d)), n_cells=int(d.groupby(list(keys), sort=False).ngroups),
                median_ape=float(d.ape.median()) if len(d) else np.nan,
                within20=float((d.ape <= 0.2).mean()) if len(d) else np.nan)
    return summ, d


def fmt_pct(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}%"
