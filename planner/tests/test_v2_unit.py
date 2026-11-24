"""Unit tests for v2 planner maths (T2.2, T2.4). Fast; no model loading."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v2ext  # noqa: E402


def test_lognormal_band_roundtrip():
    mu, s = v2ext.lognormal_from_band(1000, 1300)
    assert abs(math.exp(mu) - 1000) < 1e-6
    assert abs(math.exp(mu + 1.0364 * s) - 1300) < 0.5


def test_on_time_probability_at_band_points():
    assert abs(v2ext.on_time_probability(1000, 1300, 1000) - 0.5) < 1e-9
    assert abs(v2ext.on_time_probability(1000, 1300, 1300) - 0.85) < 2e-3
    assert v2ext.on_time_probability(1000, 1300, 0) == 0.0
    assert v2ext.on_time_probability(1000, 1300, None) is None
    # monotone in the deadline
    ps = [v2ext.on_time_probability(1000, 1300, d) for d in (600, 900, 1200, 2000)]
    assert ps == sorted(ps)


def test_degenerate_band():
    # p85 == p50 -> (almost) a step at p50
    assert v2ext.on_time_probability(1000, 1000, 999) < 0.01
    assert v2ext.on_time_probability(1000, 1000, 1001) > 0.99


def test_combine_lognormal_single_part_identity():
    p50, p85 = v2ext.combine_lognormal([(1000, 1300)])
    assert abs(p50 - 1000) < 1 and abs(p85 - 1300) < 1


def test_combine_lognormal_independent_parts_narrower_than_sum():
    parts = [(1000, 1300), (600, 900), (300, 510)]
    p50, p85 = v2ext.combine_lognormal(parts, fixed_s=200)
    assert p85 < 200 + sum(b for _, b in parts)  # independence: less than the summed p85s
    assert p85 > p50 > 200
    # Monte Carlo check of the 85th percentile
    rng = np.random.default_rng(0)
    tot = np.full(200_000, 200.0)
    for a, b in parts:
        mu, s = v2ext.lognormal_from_band(a, b)
        tot += np.exp(rng.normal(mu, s, len(tot)))
    assert abs(np.quantile(tot, 0.85) - p85) / p85 < 0.03


def test_expected_wait_regular_headway():
    # buses every 600 s all day: random arrival waits 300 s on average
    times = np.arange(0, 86400 + 600, 600, dtype=np.int32)
    integ, cov, n = v2ext.expected_wait_over_window(times, 8 * 3600, 9 * 3600)
    assert cov == 3600 and n == 6
    assert abs(integ / cov - 300) < 1e-6


def test_expected_wait_irregular_matches_formula():
    # alternating gaps 120/1080 s: E[W] = mean(H^2) / (2 mean(H)) = (120^2+1080^2)/2 / (2*600) = 492
    g = np.tile([120, 1080], 100)
    times = np.concatenate([[0], np.cumsum(g)]).astype(np.int32)
    integ, cov, _ = v2ext.expected_wait_over_window(times, 0, 24000)
    assert abs(integ / cov - 492) < 1e-6


def test_expected_wait_cap():
    times = np.array([0, 4000], dtype=np.int32)
    integ, cov, _ = v2ext.expected_wait_over_window(times, 0, 4000)
    # min(4000 - tau, 1200) averaged over [0, 4000): (2800*1200 + 1200^2/2)/4000
    assert abs(integ / cov - (2800 * 1200 + 1200 ** 2 / 2) / 4000) < 1e-6


def test_oct_daytypes():
    assert v2ext.oct_daytype(1) == 0  # Thu
    assert v2ext.oct_daytype(2) == 1  # Gandhi Jayanti
    assert v2ext.oct_daytype(3) == 1 and v2ext.oct_daytype(4) == 1  # Sat, Sun
    assert v2ext.oct_daytype(5) == 0
    assert v2ext.oct_daytype(20) == 1  # Dussehra (Tue)


def test_pooled_waits_on_toy_index(tmp_path):
    # stop 7: route 1 every 1200 s, route 2 every 1200 s offset 600 -> pooled 300 s; route 1 alone 600 s
    t1 = np.arange(0, 86400, 1200); t2 = t1 + 600
    rows = []
    for d in (5, 6, 7, 8, 9, 12, 13, 14):  # eight weekdays (MIN_GAPS = 20)
        for t, r in [(x, 1) for x in t1] + [(x, 2) for x in t2]:
            rows.append((7, d, t, r))
    rows.sort()
    a = np.array(rows)
    np.savez(tmp_path / "idx.npz", stops=np.array([7]), ptr=np.array([0, len(a)]), day=a[:, 1].astype(np.int8),
             t=a[:, 2].astype(np.int32), route=a[:, 3].astype(np.int32))
    pw = v2ext.PooledWaits(str(tmp_path / "idx.npz"))
    w, n, nd = pw.wait(7, [1, 2], 8, 0)
    assert abs(w - 300) < 1e-6 and nd == 8
    w1, _, _ = pw.wait(7, [1], 8, 0)
    assert abs(w1 - 600) < 1e-6
    assert pw.wait(7, [1, 2], 8, 1) is None  # no weekend days in the toy index
    assert pw.wait(99, [1], 8, 0) is None
    pw.wait(7, [2, 1], 8, 0)
    assert pw.hits >= 1  # cache key ignores route order
