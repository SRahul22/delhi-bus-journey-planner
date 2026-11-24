"""RealtimeTravelTimeModel: wraps the planner's historical TravelTimeModel and applies live ratios.

Same interface as planner/ttmodel.TravelTimeModel (cum, waits, ride, wait_s, seg, stats, source ...), so a
Planner can use it unchanged:   rt_planner = copy.copy(planner); rt_planner.model = RealtimeTravelTimeModel(...)

* ride(): each segment's historical time x live_ratio ** w(t), where t is when the bus reaches that segment.
  w = 1 up to FULL_S after "now", fading linearly to 0 at ZERO_S (live conditions decay; beyond ~1 h it is
  the historical model again).
* cum(): the RAPTOR scan's cumulative ride table, with the same weighting for the hour, recomputed whenever
  the live snapshot changes.
* wait_s(): if a live bus of the route is heading to the stop, the wait is (its predicted arrival - t),
  capped at the historical headway (another bus may start from the terminal meanwhile); otherwise the
  historical headway/2.
All times are seconds since IST midnight on the live clock's day; the app only uses this model for
queries on that day.
"""
import numpy as np

from live_state import sod

FULL_S = 1800
ZERO_S = 4500


class RealtimeTravelTimeModel:
    def __init__(self, base, live):
        self.base = base
        self.live = live
        self._cache = {}
        self._snap_id = None

    def __getattr__(self, name):  # seg, p85, wait, stats, factor, ...
        return getattr(self.base, name)

    @property
    def source(self):
        return self.base.source + " + live GPS ratios"

    # ------------------------------------------------------------------ helpers
    def _snap(self):
        s = self.live.snapshot
        if s is not None and id(s) != self._snap_id:
            self._cache = {}
            self._snap_id = id(s)
        return s

    @staticmethod
    def weight(delta_s):
        if delta_s <= FULL_S:
            return 1.0
        if delta_s >= ZERO_S:
            return 0.0
        return (ZERO_S - delta_s) / (ZERO_S - FULL_S)

    # ------------------------------------------------------------------ TravelTimeModel interface
    def cum(self, dt, h):
        s = self._snap()
        if s is None:
            return self.base.cum(dt, h)
        now = sod(s.now)
        w = self.weight(max(h * 3600, now) - now) if h * 3600 + 3600 > now - 3600 else 0.0
        if w <= 0:
            return self.base.cum(dt, h)
        k = (dt, h)
        c = self._cache.get(k)
        if c is None:
            off = self.live.off
            fac = np.exp(s.logr * w)
            c = [np.concatenate([[0.0], np.cumsum(seg[:, dt, h] * fac[off[ri]:off[ri + 1]])]).tolist()
                 for ri, seg in enumerate(self.base.seg)]
            self._cache[k] = c
        return c

    def waits(self, dt, h):
        return self.base.waits(dt, h)

    def ride(self, ri, i, j, t_sec, dt):
        s = self._snap()
        if s is None:
            return self.base.ride(ri, i, j, t_sec, dt)
        now = sod(s.now)
        seg, p85 = self.base.seg[ri], self.base.p85[ri]
        # planner blend mode: segments after the first blend_s seconds use the mean (see planner/ttmodel.py)
        segm = self.base.seg_mean[ri] if getattr(self.base, "mode", "") == "blend" and getattr(self.base, "seg_mean", None) is not None else None
        blend_s = getattr(self.base, "blend_s", 0)
        lr = s.logr[self.live.off[ri]:self.live.off[ri + 1]]
        t = t_sec
        tot = tot85 = 0.0
        for k in range(i, j):
            h = int(t // 3600) % 24
            f = float(np.exp(lr[k] * self.weight(t - now)))
            v = float((segm if segm is not None and tot >= blend_s else seg)[k, dt, h]) * f
            tot += v
            tot85 += float(p85[k, dt, h]) * f
            t += v
        return tot, tot85

    def wait_s(self, ri, i, t_sec, dt):
        hist = self.base.wait_s(ri, i, t_sec, dt)
        s = self._snap()
        if s is None:
            return hist
        now = sod(s.now)
        if not (-120 <= t_sec - now <= 3600):
            return hist
        day0 = s.now - now
        for eta, _veh in self.live.arrivals(ri, i, s):
            if eta - day0 >= t_sec:
                return float(min(eta - day0 - t_sec, 2 * hist))
        return hist

    def live_wait_detail(self, ri, i, t_sec):
        """For the API response: (wait_s, 'live'|'historical')."""
        s = self._snap()
        if s is None:
            return None
        day0 = s.now - sod(s.now)
        for eta, veh in self.live.arrivals(ri, i, s):
            if eta - day0 >= t_sec:
                return float(eta - day0 - t_sec), int(veh)
        return None
