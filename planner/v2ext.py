"""v2 planner extensions (AGENTS.md section 8): band maths and on-time probability (T2.4),
pooled waits over parallel routes (T2.2). Since 2026-11-24 (config C) V2_WAIT defaults to pooled; V2_BAND stays opt-in.

Flags
  V2_WAIT=pooled   (default) pooled waits over the attractive route set (T2.2). Index file: POOLED_INDEX
                   (default E:\\hackathon-work\\v2\\planner\\pooled_index_visits_1-24.npz, stop-visits index,
                   days 1-24). If the index is missing, the pooled overlay is skipped. V2_WAIT=v1 rolls back.
  V2_BAND=1        journey band combined in log-normal space (A3) + door-to-door shift if
                   MODEL_DIR\\v2\\shifts_d2d.json exists (T2.6). Otherwise band_version = v1_p85.
"""
import json
import math
import os
import threading

import numpy as np

Z85 = 1.0364  # standard normal 85th percentile
WAIT_CAP = 1200.0
ATTRACT = 1.1  # attractive set: ride time <= 1.1 x best
MIN_GAPS = 20  # minimum observed gaps (over all matching days) before the pooled wait replaces v1
HOLIDAYS = {2, 20}  # Oct 2026: Gandhi Jayanti, Dussehra (pooled with weekends, section 4)
DEFAULT_WAIT = "pooled"  # config C
DEFAULT_INDEX = r"E:\hackathon-work\v2\planner\pooled_index_visits_1-24.npz"  # stop-visits index, days 1-24


def wait_mode():
    return os.environ.get("V2_WAIT", DEFAULT_WAIT).lower()


def band_on():
    return os.environ.get("V2_BAND", "0") == "1"


# ------------------------------------------------------------------ band maths (section 4)
def lognormal_from_band(p50, p85):
    p50 = max(float(p50), 1.0)
    p85 = max(float(p85), p50)
    return math.log(p50), max(math.log(p85 / p50) / Z85, 1e-6)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def on_time_probability(p50, p85, deadline_s):
    """P(T <= deadline) for T log-normal with median p50 and 85th percentile p85."""
    if deadline_s is None:
        return None
    if deadline_s <= 0:
        return 0.0
    mu, sig = lognormal_from_band(p50, p85)
    return norm_cdf((math.log(deadline_s) - mu) / sig)


def combine_lognormal(parts, fixed_s=0.0):
    """A3: sum of independent log-normal components (each given as (p50, p85)) plus a fixed part (walking).
    Fenton-Wilkinson moment match. Returns the (p50, p85) of the total."""
    m = v = 0.0
    for p50, p85 in parts:
        if p50 <= 0:
            continue
        mu, s = lognormal_from_band(p50, p85)
        mi = math.exp(mu + s * s / 2)
        m += mi
        v += (math.exp(s * s) - 1) * mi * mi
    if m <= 0:
        return fixed_s, fixed_s
    s2 = math.log(1 + v / (m * m))
    mu = math.log(m) - s2 / 2
    s = math.sqrt(s2)
    return fixed_s + math.exp(mu), fixed_s + math.exp(mu + Z85 * s)


_D2D = None


def d2d_shifts(model_dir):
    """shifts_d2d.json from ml (T2.6), or {} if not published. Expected: {"<bucket>|<peak>": log-shift, ...}
    or {"global": log-shift}; values are added to log(p85)."""
    global _D2D
    if _D2D is None:
        p = os.path.join(model_dir, "v2", "shifts_d2d.json")
        try:
            with open(p) as f:
                raw = json.load(f)
            _D2D = {str(k): float(v) for k, v in (raw.get("shifts", raw)).items() if isinstance(v, (int, float))}
        except Exception:  # noqa: BLE001
            _D2D = {}
    return _D2D


def bucket(sec):
    m = sec / 60
    return "<10" if m < 10 else "10-30" if m < 30 else "30-60" if m < 60 else ">60"


def is_peak(depart_dt):
    """Section 4: weekdays 08:00-10:59 and 17:00-20:59; Oct 2026 holidays are off-peak (as in ml's predict.py)."""
    if depart_dt.weekday() >= 5 or (depart_dt.year == 2026 and depart_dt.month == 10 and depart_dt.day in HOLIDAYS):
        return False
    h = depart_dt.hour
    return 8 <= h <= 10 or 17 <= h <= 20


# ------------------------------------------------------------------ pooled waits (T2.2)
def oct_daytype(day):
    """0 = weekday, 1 = weekend or holiday (Oct 2026; 1 Oct is a Thursday)."""
    if day in HOLIDAYS:
        return 1
    return 1 if (day + 2) % 7 in (5, 6) else 0  # weekday(): 1 Oct -> 3 (Thu)


def expected_wait_over_window(times, h0, h1):
    """Exact expected wait for an arrival uniform on [h0, h1), given sorted bus times (one service day).
    Returns (integral_of_wait, covered_seconds, n_gaps). Arrivals after the last bus are not covered
    (censored, no next bus that day)."""
    if len(times) == 0:
        return 0.0, 0.0, 0
    i = int(np.searchsorted(times, h0, side="left"))
    prev = float(h0)
    integ = cov = 0.0
    n = 0
    while i < len(times):
        b = float(times[i])
        lo = prev
        hi = min(b, float(h1))
        if hi > lo:
            # wait at tau = min(b - tau, cap)
            integ += _capped_integral(b, lo, hi)
            cov += hi - lo
            n += 1
        if b >= h1:
            break
        prev = max(prev, b)
        i += 1
    return integ, cov, n


def _capped_integral(b, lo, hi):
    """Integral over tau in [lo, hi] of min(b - tau, WAIT_CAP)."""
    c = b - WAIT_CAP  # for tau < c, wait = cap
    tot = 0.0
    if lo < c:
        x = min(hi, c)
        tot += WAIT_CAP * (x - lo)
        lo = x
    if hi > lo:
        tot += ((b - lo) ** 2 - (b - hi) ** 2) / 2
    return tot


class PooledWaits:
    """Stop-visit index (built by build_pooled_index.py) + cached pooled-wait lookups."""

    def __init__(self, path=None):
        self.path = path or os.environ.get("POOLED_INDEX", DEFAULT_INDEX)
        z = np.load(self.path)
        self.stops = z["stops"]
        self.ptr = z["ptr"]
        self.day = z["day"]
        self.t = z["t"]
        self.route = z["route"]
        self.days = sorted(set(np.unique(self.day).tolist()))
        self._cache = {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def _rows(self, stop_id):
        k = int(np.searchsorted(self.stops, stop_id))
        if k >= len(self.stops) or self.stops[k] != stop_id:
            return None
        return slice(int(self.ptr[k]), int(self.ptr[k + 1]))

    def wait(self, stop_id, route_ids, hour, dt):
        """Expected wait (s) at stop for the first bus of any route in route_ids, random arrival in `hour`,
        days of daytype dt. Returns (wait_s, n_gaps, n_days) or None when there are not enough data."""
        key = (int(stop_id), tuple(sorted(int(r) for r in route_ids)), int(hour), int(dt))
        with self._lock:
            if key in self._cache:
                self.hits += 1
                return self._cache[key]
        self.misses += 1
        res = self._compute(*key)
        with self._lock:
            self._cache[key] = res
        return res

    def _compute(self, stop_id, routes, hour, dt):
        sl = self._rows(stop_id)
        if sl is None:
            return None
        day = self.day[sl]; t = self.t[sl]; rt = self.route[sl]
        m = np.isin(rt, np.asarray(routes, dtype=np.int32))
        if not m.any():
            return None
        day = day[m]; t = t[m]
        h0, h1 = hour * 3600, (hour + 1) * 3600
        integ = cov = 0.0
        n = nd = 0
        for d in np.unique(day):
            if oct_daytype(int(d)) != dt:
                continue
            tt = t[day == d]  # already sorted by (day, t)
            a, c, k = expected_wait_over_window(tt, h0, h1)
            if k:
                integ += a; cov += c; n += k; nd += 1
        if n < MIN_GAPS or cov <= 0:
            return None
        return (min(integ / cov, WAIT_CAP), n, nd)


class _NoIndex:
    def wait(self, *a, **k):
        return None


_POOLED = None


def pooled():
    """Lazily loaded singleton (the app preloads it at startup when V2_WAIT=pooled)."""
    global _POOLED
    if _POOLED is None:
        try:
            _POOLED = PooledWaits()
            print(f"pooled waits: {_POOLED.path} ({len(_POOLED.t)} visits, days {_POOLED.days[0]}-{_POOLED.days[-1]})")
        except Exception as e:  # noqa: BLE001
            print("pooled waits unavailable (falling back to v1 waits):", repr(e))
            _POOLED = _NoIndex()
    return _POOLED
