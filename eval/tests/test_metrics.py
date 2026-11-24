"""Unit tests for eval/metrics.py on toy arrays. Run: python -m pytest eval/tests  or  python eval/tests/test_metrics.py"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import metrics as M  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_error(self):
        r = M.error([110, 90, 150, 100], [100, 100, 100, 100])
        self.assertAlmostEqual(r["median_ape"], 0.10)  # apes 0.1, 0.1, 0.5, 0
        self.assertAlmostEqual(r["within20"], 0.75)
        self.assertEqual(r["n"], 4)

    def test_within20_boundary_inclusive(self):
        self.assertAlmostEqual(M.error([120], [100])["within20"], 1.0)

    def test_bias(self):
        r = M.bias([110, 90, 130], [100, 100, 100])
        self.assertAlmostEqual(r["mean_bias_s"], 10.0)
        self.assertAlmostEqual(r["median_bias_s"], 10.0)
        self.assertAlmostEqual(r["mean_bias_pct"], 0.10)
        r = M.bias([200], [100])
        self.assertAlmostEqual(r["median_bias_pct"], 1.0)

    def test_coverage_width(self):
        self.assertAlmostEqual(M.coverage([100, 100, 100, 100], [50, 100, 101, 200]), 0.5)
        self.assertAlmostEqual(M.width([100, 200], [150, 220]), (0.5 + 0.1) / 2)

    def test_pinball(self):
        # actual above q: 0.85 * 10; below: 0.15 * 10
        self.assertAlmostEqual(M.pinball([100], [110]), 8.5)
        self.assertAlmostEqual(M.pinball([100], [90]), 1.5)
        self.assertAlmostEqual(M.pinball([100, 100], [110, 90]), 5.0)

    def test_pinball_minimised_at_quantile(self):
        rng = np.random.default_rng(0)
        a = rng.lognormal(6, 0.3, 20000)
        qs = np.quantile(a, [0.7, 0.85, 0.95])
        losses = [M.pinball(np.full_like(a, q), a) for q in qs]
        self.assertLess(losses[1], losses[0])
        self.assertLess(losses[1], losses[2])

    def test_lognormal_calibration(self):
        # actuals drawn from the log-normal implied by the band -> shares ~0.5/0.85/0.9
        rng = np.random.default_rng(1)
        n = 200000
        p50 = np.full(n, 600.0)
        sigma = 0.25
        p85 = p50 * np.exp(sigma * M.Z85)
        np.testing.assert_allclose(M.lognormal_sigma(p50[:1], p85[:1]), [sigma])
        a = p50 * np.exp(sigma * rng.standard_normal(n))
        c = M.calibration(p50, p85, a)
        self.assertAlmostEqual(c["under_p50"], 0.50, delta=0.01)
        self.assertAlmostEqual(c["under_p85"], 0.85, delta=0.01)
        self.assertAlmostEqual(c["under_p90"], 0.90, delta=0.01)

    def test_bucket_and_peak(self):
        b = M.bucket([0, 599, 600, 1799, 1800, 3599, 3600, 99999])
        self.assertEqual(list(b), ["<10", "<10", "10-30", "10-30", "30-60", "30-60", ">60", ">60"])
        pk = M.is_peak([7.99, 8.0, 10.99, 11.0, 16.5, 17.0, 20.99, 21.0, 9.0], [1, 1, 1, 1, 1, 1, 1, 1, 0])
        self.assertEqual(list(pk), [False, True, True, False, False, True, True, False, False])

    def test_by_bucket_peak(self):
        a = np.array([300, 300, 1200, 1200, 4000])
        pk = np.array([1, 0, 1, 0, 0], bool)
        t = M.by_bucket_peak(a, pk, {"m": a * 1.1}, {"m": a * 1.2})
        row = t[(t.bucket == "all") & (t.peak == "all")].iloc[0]
        self.assertEqual(row.n, 5)
        self.assertAlmostEqual(row.median_ape, 0.1)
        self.assertAlmostEqual(row.coverage, 1.0)
        self.assertEqual(int(t[(t.bucket == ">60") & (t.peak == "peak")].n.iloc[0]), 0)
        self.assertEqual(int(t[(t.bucket == "<10") & (t.peak == "all")].n.iloc[0]), 2)

    def test_bucket_by_pred(self):
        a = np.array([500, 700])
        t = M.by_bucket_peak(a, np.array([0, 0], bool), {"m": np.array([700, 500])}, bucket_by="pred")
        self.assertEqual(int(t[(t.bucket == "<10") & (t.peak == "all")].n.iloc[0]), 1)
        r = t[(t.bucket == "10-30") & (t.peak == "all")].iloc[0]
        self.assertAlmostEqual(r.median_ape, 0.4)  # pred 700 vs actual 500

    def test_noise_floor(self):
        rows = []
        for i in range(10):  # cell A: 10 rides, median 100 (values 95..104.5)
            rows.append(dict(route_id=1, board_stop=1, alight_stop=2, hour=8, actual_s=95 + i))
        for i in range(5):  # cell B: too few rides -> excluded
            rows.append(dict(route_id=2, board_stop=1, alight_stop=2, hour=8, actual_s=100))
        s, d = M.noise_floor(pd.DataFrame(rows))
        self.assertEqual(s["n_rides"], 10)
        self.assertEqual(s["n_cells"], 1)
        med = np.median(np.arange(95, 105))
        exp = np.median(np.abs(med - np.arange(95, 105)) / np.arange(95, 105))
        self.assertAlmostEqual(s["median_ape"], exp)


if __name__ == "__main__":
    unittest.main()
