"""Model v3: learned correction of bus ride times (one bus leg = board stop -> alight stop on one route).

    from predict import correct            # (sys.path.insert(0, ".../model/v3") first)
    secs = correct(route_id, board_stop, alight_stop, depart_dt, base_median_s, base_mean_s, n_stops, dist_m)

How it works
  1. Base: sum of segment medians and means along the route pattern, time-dependent by hour, from the same
     1-24 Oct lookup the planner uses (shipped in v3_lookup.npz, with the same n>=3 / all-hour / distance fallbacks
     the model was trained with). Computing the base here keeps training and serving consistent.
  2. Correction: LightGBM residual model (pure-python tree evaluation of lgbm_model.json; LightGBM itself is NOT
     needed) predicts log(actual / sum_of_medians) from ~21 features (base time, mean/median skew, distance,
     stops, hour, weekday, route speed/length/coverage, board/alight coordinates, data coverage).
     The multiplicative correction is clipped to [0.5, 2.5].
  3. If the route or stops are not in v3's patterns (or the stop order is reversed), it falls back to the
     caller's base_median_s (else base_mean_s) times a bucket calibration factor (calib.json).

Environment: V3_MODE=xgb (final XGBoost, needs xgboost) | lgbm (default) | calib (bucket calibration only) | off (return base_median_s unchanged).
Arguments base_median_s/base_mean_s/n_stops/dist_m are only used for that fallback; pass None if unknown.
depart_dt: naive datetime in IST (or tz-aware; converted to IST), or seconds since midnight IST (weekday assumed).
"""
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get("V3_MODE", "lgbm")
IST = timezone(timedelta(hours=5, minutes=30))
CLIP = (0.5, 2.5)
_M = None


class _Model:
    def __init__(self, d=HERE):
        z = np.load(os.path.join(d, "v3_lookup.npz"), allow_pickle=False)
        self.N = z["N"]; self.MED = z["MED"].astype(np.float32); self.MEAN = z["MEAN"].astype(np.float32)
        self.soff = z["soff"]; self.poff = z["poff"]; self.rsize = z["rsize"]
        self.pat_stop = z["pat_stop"]; self.pat_dist = z["pat_dist"].astype(np.float64)
        self.ridx = {int(r): i for i, r in enumerate(z["routes"])}
        self.ll = {int(s): (float(a), float(b)) for s, a, b in zip(z["stop_id"], z["stop_lat"], z["stop_lon"])}
        # column order of rf as written by build_rides.py (rf_cols in the npz is an object array; not loaded)
        self.rf_cols = ["route_len_m", "route_nstops", "route_speed", "route_obs", "route_cov", "route_skew"]; self.rf = z["rf"]
        meta = json.load(open(os.path.join(d, "calib.json")))
        self.features = meta["features"]
        c = meta["calib"]; self.cx = np.array(c["x_log_base_med"]); self.cy = np.array(c["log_factor"])
        self.trees = []
        self.xgb = None
        if MODE == "xgb":  # final submission model (scripts/xgb_final.py); needs xgboost installed
            import xgboost
            xm = json.load(open(os.path.join(d, "xgb_final_meta.json")))
            self.features = xm["features"]
            self.xgb = xgboost.Booster(); self.xgb.load_model(os.path.join(d, "xgb_final.json"))
        if MODE == "lgbm":
            dump = json.load(open(os.path.join(d, "lgbm_model.json")))
            assert dump["feature_names"] == self.features, "feature order mismatch"
            for t in dump["tree_info"]:
                self.trees.append(self._flat(t["tree_structure"]))

    @staticmethod
    def _flat(node):
        # flat arrays: feat, thr, default_left, missing(0 none,1 zero,2 nan), left, right, value ; leaves have feat=-1
        F, T, DL, MS, L, R, V = [], [], [], [], [], [], []
        def add(n):
            i = len(F)
            F.append(-1); T.append(0.0); DL.append(True); MS.append(0); L.append(-1); R.append(-1); V.append(0.0)
            if "leaf_value" in n:
                V[i] = n["leaf_value"]; return i
            F[i] = n["split_feature"]; T[i] = n["threshold"]; DL[i] = n["default_left"]
            MS[i] = {"None": 0, "Zero": 1, "NaN": 2}[n["missing_type"]]
            L[i] = add(n["left_child"]); R[i] = add(n["right_child"])
            return i
        add(node)
        return F, T, DL, MS, L, R, V

    def tree_sum(self, x):
        s = 0.0
        for F, T, DL, MS, L, R, V in self.trees:
            i = 0
            while F[i] >= 0:
                v = x[F[i]]
                ms = MS[i]
                if v != v:  # NaN
                    if ms == 2: i = L[i] if DL[i] else R[i]; continue
                    v = 0.0
                if ms == 1 and v == 0.0:
                    i = L[i] if DL[i] else R[i]; continue
                i = L[i] if v <= T[i] else R[i]
            s += V[i]
        return s

    def calib(self, base):
        return base * math.exp(float(np.interp(math.log(max(base, 1.0)), self.cx, self.cy)))

    def locate(self, route_id, board_stop, alight_stop):
        try:
            r = self.ridx[int(route_id)]
        except (KeyError, ValueError, TypeError):
            return None
        p0, n = int(self.poff[r]), int(self.rsize[r])
        st = self.pat_stop[p0:p0 + n]
        b = np.flatnonzero(st == int(board_stop))
        if not len(b):
            return None
        b = int(b[0])
        a = np.flatnonzero(st[b + 1:] == int(alight_stop))
        if not len(a):
            return None
        return r, b, b + 1 + int(a[0])

    def features_for(self, r, b, a, tsec, minute_h, dow):
        dt = 1 if dow >= 5 else 0
        g0 = int(self.soff[r]) + b
        smed = smean = nfb = slogn = 0.0
        for k in range(a - b):
            h = int((tsec + smed) // 3600) % 24
            smed += float(self.MED[g0 + k, dt, h]); smean += float(self.MEAN[g0 + k, dt, h])
            n = int(self.N[g0 + k, dt, h]); nfb += n < 3; slogn += math.log1p(n)
        steps = a - b
        p0 = int(self.poff[r])
        dist = self.pat_dist[p0 + a] - self.pat_dist[p0 + b]
        rf = dict(zip(self.rf_cols, self.rf[r]))
        bs, as_ = int(self.pat_stop[p0 + b]), int(self.pat_stop[p0 + a])
        blat, blon = self.ll.get(bs, (float("nan"), float("nan"))); alat, alon = self.ll.get(as_, (float("nan"), float("nan")))
        smed = max(smed, 1.0)
        f = dict(log_base_med=math.log(smed), skew=smean / smed, n_stops=steps, log_dist=math.log1p(dist),
                 base_speed=dist / smed, hour_f=minute_h, dow=dow, weekend=dt,
                 route_len_m=rf["route_len_m"], route_nstops=rf["route_nstops"], route_speed=rf["route_speed"],
                 log_route_obs=math.log1p(rf["route_obs"]), route_cov=rf["route_cov"], route_skew=rf["route_skew"],
                 board_pos=b / (rf["route_nstops"] - 1), b_lat=blat, b_lon=blon, a_lat=alat, a_lon=alon,
                 fb_frac=nfb / steps, mean_logn=slogn / steps)
        return [float(f[c]) for c in self.features], smed, smean


def _model():
    global _M
    if _M is None:
        _M = _Model()
    return _M


def _time_parts(depart_dt):
    if isinstance(depart_dt, (int, float)):
        s = float(depart_dt) % 86400
        return s, s / 3600 - (s % 60) / 3600, 2  # weekday (Wednesday) assumed
    if depart_dt.tzinfo is not None:
        depart_dt = depart_dt.astimezone(IST)
    s = depart_dt.hour * 3600 + depart_dt.minute * 60 + depart_dt.second
    return float(s), depart_dt.hour + depart_dt.minute / 60, depart_dt.weekday()


def correct(route_id, board_stop, alight_stop, depart_dt, base_median_s=None, base_mean_s=None, n_stops=None, dist_m=None):
    """Corrected ride seconds for one bus leg. See module docstring."""
    if MODE == "off":
        return base_median_s
    m = _model()
    loc = m.locate(route_id, board_stop, alight_stop)
    if loc is None:
        base = base_median_s if base_median_s else base_mean_s
        return None if base is None else m.calib(float(base))
    tsec, hour_f, dow = _time_parts(depart_dt)
    x, smed, _ = m.features_for(*loc, tsec, hour_f, dow)
    if MODE == "calib":
        return m.calib(smed)
    if m.xgb is not None:
        f = math.exp(float(m.xgb.inplace_predict(np.asarray([x], dtype=np.float32))[0]))
    else:
        f = math.exp(m.tree_sum(x))
    return smed * min(max(f, CLIP[0]), CLIP[1])


if __name__ == "__main__":
    # self-test against predictions made by LightGBM itself on the training machine
    import time
    chk = json.load(open(os.path.join(HERE, "check_rides.json")))
    t0 = time.time()
    errs, errs_c = [], []
    for c in chk:
        dtm = datetime.strptime(c["depart_ist"], "%Y-%m-%d %H:%M:%S")
        p = correct(c["route_id"], c["board_stop"], c["alight_stop"], dtm)
        errs.append(abs(p / c["pred_lgbm"] - 1)); errs_c.append(abs(p / c["actual_s"] - 1))
    n = len(chk)
    print(f"{n} rides, {1000*(time.time()-t0)/n:.2f} ms/ride; max rel diff vs LightGBM {max(errs):.4f} "
          f"(median {sorted(errs)[n//2]:.5f}); median APE vs actual {100*sorted(errs_c)[n//2]:.1f}%")
