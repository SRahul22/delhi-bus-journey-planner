"""pytest: v2 T2.1 wait formulas.  Run: python -m pytest model/scripts/test_headways_v2.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_headways_v2 import wait_quantile_sorted


def _mc(H, q, n=400_000, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.choice(len(H), size=n, p=H / H.sum())
    return np.quantile(rng.uniform(0, 1, n) * H[g], q)


def test_constant_headway():
    s, w = wait_quantile_sorted(np.array([600.0, 600.0, 600.0]), np.zeros(3, int), 0.85)
    assert abs(w[0] - 510.0) < 1e-6  # uniform wait on [0, 600]


def test_matches_simulation_two_groups():
    A = np.array([1500.0, 900, 300, 200, 60]); B = np.array([1200.0, 100, 100, 100])
    H = np.r_[A, B]; gid = np.r_[np.zeros(5, int), np.ones(4, int)]
    s, w = wait_quantile_sorted(H, gid, 0.85)
    assert list(s) == [0, 5]
    assert abs(w[0] - _mc(A, 0.85)) < 10 and abs(w[1] - _mc(B, 0.85)) < 10


def test_expected_wait_formula():
    H = np.array([1500.0, 900, 300, 200, 60])
    ew = (H ** 2).mean() / (2 * H.mean())
    rng = np.random.default_rng(1); g = rng.choice(len(H), size=400_000, p=H / H.sum())
    assert abs(ew - (rng.uniform(0, 1, len(g)) * H[g]).mean()) < 5
