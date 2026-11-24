"""Unit tests for the v2 live layer (realtime/kf.py, realtime/live_api.py). Run: python -m pytest realtime/tests -q"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kf as kfmod  # noqa: E402
from live_api import LiveInterface  # noqa: E402

P = dict(tau_s=600.0, tau_c=3600.0, q=1e-5, P0=0.09, Pc0=0.04, rc_add=0.04, step=60.0)


def make(n=4):
    gx = np.array([0, 0, 10, 10])[:n]; gy = np.zeros(n, int)
    sigma = np.full((n, 2, 24), 0.2, np.float32)
    return kfmod.SegmentKF(n, gx, gy, sigma, **P)


def obs(f, t1, z):
    f = np.asarray(f, np.int64); t1 = np.asarray(t1, float)
    return dict(f=f, t0=t1 - 60, t1=t1, logr=np.asarray(z, float))


def test_no_obs_stays_zero():
    k = make(); k.advance(obs([], [], []), 1000.0)
    x, v = kfmod.forecast(k.state(), 0.0)
    assert np.allclose(x, 0) and np.allclose(v, P["P0"])


def test_update_moves_toward_measurement_and_cell():
    k = make(); o = obs([0], [100.0], [0.5])
    k.advance(o, 100.0)
    x, v = kfmod.forecast(k.state(), 0.0)
    assert 0 < x[0] < 0.5                 # observed segment pulled toward z
    assert 0 < x[1] < x[0]                # same 3 km cell, unobserved: follows the cell
    assert x[2] == 0 and x[3] == 0        # other cell untouched
    assert v[0] < P["P0"]


def test_relaxation_and_horizon_forecast():
    k = make(); k.advance(obs([0], [100.0], [0.5]), 100.0)
    s = k.state()
    x0 = kfmod.forecast(s, 0.0, 0)[0]
    xh = kfmod.forecast(s, 1800.0, 0)[0]
    exp = s["xc"][0] * math.exp(-1800 / P["tau_c"]) + s["d"][0] * math.exp(-1800 / P["tau_s"])
    assert abs(xh - exp) < 1e-6 and abs(xh) < abs(x0)
    k.advance(obs([0], [100.0], [0.5]), 100.0 + 1800)   # no new obs: state relaxes like the forecast
    assert abs(kfmod.forecast(k.state(), 0.0, 0)[0] - xh) < 1e-6


def test_incremental_equals_bulk():
    rng = np.random.default_rng(1)
    t1 = np.sort(rng.uniform(0, 3600, 200)); f = rng.integers(0, 4, 200); z = rng.normal(0.2, 0.3, 200)
    o = obs(f, t1, z)
    a = make(); a.advance(o, 3600.0)
    b = make()
    for T in range(300, 3601, 300):
        b.advance(o, float(T), n_max=int(np.searchsorted(t1, T, "right")))
    assert np.allclose(a.state()["d"], b.state()["d"], atol=1e-6) and np.allclose(a.xc, b.xc, atol=1e-6)


def test_clip():
    s = dict(xc=np.array([5.0], np.float32), d=np.array([0.0], np.float32), P=np.array([0.01], np.float32),
             tau_s=600.0, tau_c=3600.0, q=1e-5, P0=0.09)
    assert abs(kfmod.forecast(s, 0.0)[0][0] - math.log(2.5)) < 1e-6


class _Snap:
    def __init__(self, kfs):
        self.now = 1000.0; self.kf = kfs; self.logr = np.array([0.2, 0.0], np.float32)


class _Live:
    mode = "kf"; n_flat = 2

    def __init__(self, snap):
        self.snapshot = snap

    def flat(self, r, a, b):
        return 0 if (str(r), str(a), str(b)) == ("1", "10", "11") else None

    def next_bus_eta(self, stop_id, route_id, snap):
        return [(400, 7), (1500, 8)]


def test_live_interface_contract():
    k = make(2); k.advance(obs([0], [900.0], [0.4]), 1000.0)
    api = LiveInterface(_Live(_Snap(k.state())), sd_path="__missing__")
    api.sd = dict(next_bus=dict(kf=[dict(lo_s=0, hi_s=600, sd_s=90.0), dict(lo_s=600, hi_s=3600, sd_s=200.0)]))
    lr, var = api.segment_state(("1", "10", "11"), 1000.0, 0)
    lr2, var2 = api.segment_state(0, 1000.0, 1800)
    assert lr > 0 and var > 0 and abs(lr2) < abs(lr) and var2 >= var
    assert api.segment_state(("9", "9", "9"), 1000.0, 0) == (0.0, 0.09)
    nb = api.next_buses("10", "1", 1100.0)   # 100 s after the snapshot
    assert nb == [(300.0, 90.0), (1400.0, 200.0)]
