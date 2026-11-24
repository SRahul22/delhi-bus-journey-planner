"""v2 journey band (AGENTS.md amendment A3 + T2.6): combine legs in log-normal space, then calibrate door to door.

    from journey_band import journey_band
    p50, p85, sigma = journey_band([(walk_s, walk_s), (wait_p50, wait_p85), (ride_p50, ride_p85), ...], peak=is_peak)

Each component (walk, wait, ride, transfer) is a (p50_s, p85_s) pair, taken as log-normal with
mu = ln p50, sigma = ln(p85/p50) / 1.0364 (section 4). Components with p85 <= p50 are deterministic.
The total is approximated by one log-normal (Fenton-Wilkinson moment matching for the spread). Per A3 the centre is
the sum of medians, so: P50 = sum(p50_i), sigma_T from the matched variance, P85 = P50 * exp(1.0364 * sigma_T).
If shifts_d2d.json is present (T2.6), P85 is multiplied by exp(shift[bucket(P50)|peak]) and sigma_T is recomputed
from the calibrated P85, so on_time_probability (T2.4) uses the calibrated band.
Pure python (math only); about 10 us per journey.
"""
import json
import math
import os

Z85 = 1.0364
HERE = os.path.dirname(os.path.abspath(__file__))
_S = None


def _shifts():
    global _S
    if _S is None:
        p = os.environ.get("V2_D2D_SHIFTS") or os.path.join(HERE, "shifts_d2d.json")
        _S = json.load(open(p)) if os.path.exists(p) else {}
    return _S


def combine(components):
    """components: iterable of (p50_s, p85_s or None). Returns (P50, P85, sigma_T) before door-to-door calibration."""
    m1 = var = med = 0.0
    for p50, p85 in components:
        if p50 is None or p50 <= 0:
            continue
        med += p50
        s = math.log(p85 / p50) / Z85 if (p85 is not None and p85 > p50) else 0.0
        mean = p50 * math.exp(s * s / 2)
        m1 += mean
        var += (math.exp(s * s) - 1) * mean * mean
    if med <= 0:
        return 0.0, 0.0, 0.0
    # Fenton-Wilkinson: log-variance of the matched log-normal
    sig = math.sqrt(math.log1p(var / (m1 * m1))) if m1 > 0 else 0.0
    return med, med * math.exp(Z85 * sig), sig


def _key(p50, peak, buckets):
    for lo, hi, nm in buckets:
        if lo <= p50 < hi:
            return f"{nm}|{'peak' if peak else 'offpeak'}"
    return f"{buckets[-1][2]}|{'peak' if peak else 'offpeak'}"


def journey_band(components, peak=False):
    """Door-to-door (P50, P85, sigma) with the T2.6 shift applied when shifts_d2d.json exists. P85 >= P50."""
    med, p85, sig = combine(components)
    S = _shifts()
    if S and med > 0:
        sh = S["shifts"].get(_key(med, peak, S["buckets_s"]), 0.0)
        p85 = max(med, p85 * math.exp(sh))
        sig = math.log(p85 / med) / Z85
    return med, p85, sig


def on_time_probability(deadline_s, p50, sigma):
    """P(total <= deadline) under the journey log-normal (section 4 / T2.4)."""
    if deadline_s <= 0:
        return 0.0
    if sigma <= 0:
        return 1.0 if deadline_s >= p50 else 0.0
    z = (math.log(deadline_s) - math.log(p50)) / sigma
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))
