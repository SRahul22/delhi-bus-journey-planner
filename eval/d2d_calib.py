"""Wait / d2d bias calibration fitted on days 13-18 ONLY (config-C plans replayed on those days,
eval/d2d_plan_extra.py calib), applied to config C on the validation common subset (days 19-24, n = 1,996).

Calibrations (all medians, so they target the median-based metrics):
  wait_add   per-leg additive offset (real - planned wait) by planned-wait bucket x peak, first wait and transfer
             waits fitted separately (cells with < 30 legs fall back to the peak/off-peak offset)
  wait_ratio per-leg multiplicative factor median(real / planned wait), same cells
  total_add  one additive offset on the door-to-door time by (direct / transfer) x peak
  total_ratio one factor median(real / planned d2d) by (direct / transfer) x peak
  python eval/d2d_calib.py  -> E:\\hackathon-work\\v2\\analysis\\calib.json
"""
import json

import numpy as np
import pandas as pd

import d2d_common as DC

EDGES = [0, 300, 600, 900, 1200, np.inf]
MIN_N = 30


def leg_frame(R):
    rows = []
    for r in R.to_dict("records"):
        legs, _ = DC.journey_legs(r)
        pk = DC.is_peak(r["day"], r["depart_s"])
        for k, l in enumerate(legs):
            rows.append(dict(key=r["key"], first=k == 0, peak=pk, plan=l["plan_wait"], real=l["real_wait"]))
    L = pd.DataFrame(rows)
    L["b"] = np.digitize(L.plan, EDGES) - 1
    return L


def fit(L, fn):
    tab = {}
    for first in (True, False):
        for pk in (True, False):
            S = L[(L["first"] == first) & (L.peak == pk)]
            fb = fn(S) if len(S) else (0.0 if fn is add else 1.0)
            for b in range(len(EDGES) - 1):
                c = S[S.b == b]
                tab[(first, pk, b)] = (fn(c) if len(c) >= MIN_N else fb, len(c))
    return tab


def add(S):
    return float(np.median(S.real - S.plan))


def ratio(S):
    return float(np.median(S.real / S.plan.clip(lower=30)))


def main():
    F = DC.load(DC.OUT / "realized_calib.csv")
    assert F.day.max() <= 18
    F = F[F.scored].reset_index(drop=True)
    LF = leg_frame(F)
    t_add, t_ratio = fit(LF, add), fit(LF, ratio)
    F["pk"] = [DC.is_peak(d, s) for d, s in zip(F.day, F.depart_s)]
    tot = {}
    for xf in (False, True):
        for pk in (True, False):
            S = F[((F.plan_n_bus > 1) == xf) & (F.pk == pk)]
            tot[(xf, pk)] = (float(np.median(S.real_total_s - S.plan_total_s)),
                             float(np.median(S.real_total_s / S.plan_total_s)), len(S))
    C = DC.common_C()
    C["pk"] = [DC.is_peak(d, s) for d, s in zip(C.day, C.depart_s)]
    LC = leg_frame(C)
    LC["add"] = [t_add[(f, p, b)][0] for f, p, b in zip(LC["first"], LC.peak, LC.b)]
    LC["fac"] = [t_ratio[(f, p, b)][0] for f, p, b in zip(LC["first"], LC.peak, LC.b)]
    d_add = LC.groupby("key")["add"].sum()
    d_rat = (LC.plan * (LC.fac - 1)).groupby(LC.key).sum()
    C["wait_add"] = C.plan_total_s + C.key.map(d_add)
    C["wait_ratio"] = C.plan_total_s + C.key.map(d_rat)
    C["total_add"] = [p + tot[(n > 1, k)][0] for p, n, k in zip(C.plan_total_s, C.plan_n_bus, C.pk)]
    C["total_ratio"] = [p * tot[(n > 1, k)][1] for p, n, k in zip(C.plan_total_s, C.plan_n_bus, C.pk)]
    out = dict(fit_days="13-18 (config C plans, %d scored journeys, %d legs)" % (len(F), len(LF)),
               fit_bias_first_wait_s=add(LF[LF["first"]]), fit_bias_transfer_wait_s=add(LF[~LF["first"]]) if (~LF["first"]).any() else None,
               fit_bias_total_s=float(np.median(F.plan_total_s - F.real_total_s)),
               fit_C_score=DC.score(F.plan_total_s, F.real_total_s),
               wait_add_table={f"{'first' if f else 'xfer'}|{'peak' if p else 'off'}|{EDGES[b]}-{EDGES[b + 1]}": v for (f, p, b), v in t_add.items()},
               total_table={f"{'xfer' if x else 'direct'}|{'peak' if p else 'off'}": v for (x, p), v in tot.items()})
    for c in ("plan_total_s", "wait_add", "wait_ratio", "total_add", "total_ratio"):
        out[c] = DC.score(C[c], C.real_total_s)
        out[c + "_direct"] = DC.score(C[c][C.plan_n_bus == 1], C.real_total_s[C.plan_n_bus == 1])
        out[c + "_transfer"] = DC.score(C[c][C.plan_n_bus > 1], C.real_total_s[C.plan_n_bus > 1])
    # how much of the error is the bias: remove the per-journey-type median bias in-sample on val (upper bound)
    oracle_shift = C.groupby(C.plan_n_bus > 1).apply(lambda S: float(np.median(S.real_total_s - S.plan_total_s)))
    C["val_debiased"] = C.plan_total_s + (C.plan_n_bus > 1).map(oracle_shift)
    out["val_debiased_in_sample"] = DC.score(C.val_debiased, C.real_total_s)
    C[["key", "plan_total_s", "real_total_s", "wait_add", "wait_ratio", "total_add", "total_ratio"]].to_csv(DC.OUT / "calib.csv", index=False)
    json.dump(out, open(DC.OUT / "calib.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

