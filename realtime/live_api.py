"""Live interface for the planner (contract 4, AGENTS.md section 5; T3.4).

    api = LiveInterface(live_state)          # realtime.service.RealtimeService builds one as service.api
    api.segment_state(seg_id, now, horizon_s) -> (log_ratio, var)
    api.next_buses(stop_id, route_id, now)    -> [(eta_s, sd_s), ...]   (soonest first)

seg_id: flat segment index (int) or a (route_id, from_stop_id, to_stop_id) tuple.
now: unix seconds (the live clock; the snapshot time is used if None). horizon_s: seconds after `now`.

log_ratio = log(expected travel time / historical p50 time) for that segment at now + horizon_s.
  V2_LIVE=kf:    Kalman forecast x(h) and its state variance P(h) (realtime/kf.py).
  V2_LIVE=ratio: v1 shrunk ratio faded with the v1 horizon weight (1 to 30 min, 0 at 75 min); var is the
                 22-Oct predictive variance of segment log ratios by horizon minus the median measurement
                 noise (floored at 0.005).
  Unknown segment or no snapshot: (0.0, 0.09).
eta_s: seconds from `now` until a live bus of that route reaches the stop (v1 ETA logic; with kf the segment
  ratios are horizon forecasts). sd_s: spread of (actual - predicted) first-bus wait on 22 Oct, by ETA bucket
  (next_bus_sd.json), for the active mode: the Normal sd matching the 15-85% residual interval,
  (p85 - p15) / 2.0728. The residuals are heavy-tailed and right-skewed (core 1.4826*MAD is only 60-440 s,
  std 450-780 s), so this matches the central 70% that a P>=0.7 transfer decision looks at. Later buses use
  the same table.
"""
import json
import os

import numpy as np

import kf as kfmod
from rt_model import RealtimeTravelTimeModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VAR = 0.09
VAR_FLOOR = 0.005


def _load_sd(path=None):
    for cand in (path, os.environ.get("NEXT_BUS_SD"), os.path.join(HERE, "next_bus_sd.json"),
                 r"E:\hackathon-work\v2\realtime\next_bus_sd.json"):
        if cand and os.path.exists(cand):
            with open(cand) as f:
                return json.load(f)
    return None


class LiveInterface:
    def __init__(self, live, sd_path=None):
        self.live = live
        self.sd = _load_sd(sd_path)

    @property
    def mode(self):
        return getattr(self.live, "mode", "ratio")

    # ------------------------------------------------------------------ helpers
    def _flat(self, seg_id):
        if isinstance(seg_id, (int, np.integer)):
            return int(seg_id) if 0 <= int(seg_id) < self.live.n_flat else None
        r, a, b = seg_id
        return self.live.flat(r, a, b)

    def _ratio_var(self, h):
        if self.sd:
            for row in self.sd.get("ratio_segment_var", []):
                if row["lo_s"] <= h < row["hi_s"]:
                    return max(row["resid_var"] - row["median_R"], VAR_FLOOR)
            rows = self.sd.get("ratio_segment_var", [])
            if rows:
                return max(rows[-1]["resid_var"] - rows[-1]["median_R"], VAR_FLOOR)
        return DEFAULT_VAR

    @staticmethod
    def _sd(row):
        if "p15_s" in row and "p85_s" in row:
            return float((row["p85_s"] - row["p15_s"]) / 2.0728)
        return float(row["sd_s"])

    def sd_for(self, eta_s):
        rows = (self.sd or {}).get("next_bus", {}).get(self.mode) or []
        for row in rows:
            if row["lo_s"] <= eta_s < row["hi_s"]:
                return self._sd(row)
        if rows:
            return self._sd(rows[-1] if eta_s >= rows[-1]["hi_s"] else rows[0])
        return float(60 + 0.25 * max(eta_s, 0))

    # ------------------------------------------------------------------ contract 4
    def segment_state(self, seg_id, now=None, horizon_s=0.0):
        snap = self.live.snapshot
        f = self._flat(seg_id)
        if snap is None or f is None:
            return 0.0, DEFAULT_VAR
        h = max(float(horizon_s) + ((float(now) - snap.now) if now is not None else 0.0), 0.0)
        if snap.kf is not None:
            x, v = kfmod.forecast(snap.kf, h, f)
            return float(x), float(v)
        return float(snap.logr[f]) * RealtimeTravelTimeModel.weight(h), float(self._ratio_var(h))

    def next_buses(self, stop_id, route_id, now=None):
        snap = self.live.snapshot
        if snap is None:
            return []
        lag = (float(now) - snap.now) if now is not None else 0.0
        out = []
        for eta, _veh in self.live.next_bus_eta(stop_id, route_id, snap):
            e = max(float(eta) - lag, 0.0)
            out.append((e, self.sd_for(e)))
        return out
