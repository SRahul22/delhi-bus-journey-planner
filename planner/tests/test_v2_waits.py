"""v2/waits: V2_WAIT=gaps knobs V2_WAIT_MIN_N, V2_WAIT_CAP, V2_WAIT_FILL on a toy one-route network. Fast."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ttmodel  # noqa: E402

FLAGS = ("V2_WAIT", "V2_HEADWAYS", "V2_WAIT_MIN_N", "V2_WAIT_CAP", "V2_WAIT_FILL", "V2_WAIT_UNOBS")


def toy_net():
    stops = pd.DataFrame(dict(stop_id=["100", "101", "102"]))
    route = dict(route_id="7", stops=np.array([0, 1, 2]), sched_seg=np.full((2, 24), 120.0), dist=np.array([500.0, 500.0]),
                 trips_per_hour=np.full(24, 6.0), offset=np.zeros(3))  # GTFS: 6 trips/h -> 300 s wait
    return dict(routes=[route], stops=stops)


def write_gaps(tmp_path):
    # stop 100, weekday: 08h dense cell (n=40, E[W]=700), 09h sparse cell (n=4, E[W]=2500); no other hours.
    rows = [dict(stop_id=100, route_id=7, hour=8, daytype="weekday", n=40, median_headway_s=400, mean_headway_s=900,
                 mean_sq_headway_s=1.26e6, expected_wait_s=700.0, wait_p85_s=1500.0, usable_share=0.7),
            dict(stop_id=100, route_id=7, hour=9, daytype="weekday", n=4, median_headway_s=4000, mean_headway_s=4000,
                 mean_sq_headway_s=2e7, expected_wait_s=2500.0, wait_p85_s=4000.0, usable_share=0.7)]
    p = tmp_path / "hw.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k in FLAGS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SEG_MODE", "median")
    hw = write_gaps(tmp_path)

    def make(**flags):
        for k in FLAGS:
            monkeypatch.delenv(k, raising=False)
        for k, v in flags.items():
            monkeypatch.setenv(k, str(v))
        return ttmodel.TravelTimeModel(toy_net(), model_dir=str(tmp_path))  # no v1 tables: GTFS waits

    return make, hw


def w(m, h, stop=0, dt=0):
    return m.wait_s(0, stop, h * 3600 + 60, dt)


def test_v1_waits(env):
    make, hw = env
    m = make(V2_WAIT="v1")  # explicit: main defaults to pooled (config C)
    assert w(m, 8) == 300 and w(m, 9) == 300 and w(m, 12) == 300


def test_gaps_old_knobs(env):
    # V2_WAIT=gaps with the pre-v2/waits knobs: only n >= 20 cells, cap 1200, no fill (was the gaps default before F)
    make, hw = env
    m = make(V2_WAIT="gaps", V2_HEADWAYS=hw, V2_WAIT_MIN_N=20, V2_WAIT_CAP=1200, V2_WAIT_FILL=0, V2_WAIT_UNOBS=0)
    assert w(m, 8) == 700 and w(m, 9) == 300 and w(m, 12) == 300


def test_default_is_config_f(env):
    # no flags: V2_WAIT=gaps with MIN_N 3, CAP 3600, FILL 1, UNOBS 1 (config F, default since 2026-11-27)
    make, hw = env
    m = make(V2_HEADWAYS=hw)
    assert m.wait_mode == "gaps"
    assert m.wait_knobs == dict(V2_WAIT="gaps", V2_WAIT_MIN_N=3, V2_WAIT_CAP=3600.0, V2_WAIT_FILL=1, V2_WAIT_UNOBS=1)
    assert w(m, 8) == 700 and w(m, 9) == 2500 and w(m, 12) == 3600
    assert w(m, 12, stop=1) == pytest.approx(700, abs=1)
    # V2_WAIT=pooled alone restores the old knobs (config C base waits)
    m = make(V2_WAIT="pooled", V2_HEADWAYS=hw)
    assert m.wait_knobs["V2_WAIT_MIN_N"] == 20 and m.wait_knobs["V2_WAIT_FILL"] == 0
    assert w(m, 8) == 700 and w(m, 9) == 300 and w(m, 12) == 300


def test_min_n_and_cap(env):
    make, hw = env
    m = make(V2_WAIT="gaps", V2_HEADWAYS=hw, V2_WAIT_MIN_N=3, V2_WAIT_CAP=1200)
    assert w(m, 9) == 1200  # sparse cell now used, but v1 cap
    m = make(V2_WAIT="gaps", V2_HEADWAYS=hw, V2_WAIT_MIN_N=3, V2_WAIT_CAP=3600, V2_WAIT_FILL=0)
    assert w(m, 9) == 2500 and w(m, 8) == 700
    assert m.wait_p85_s(0, 0, 9 * 3600, 0) == 4000


def test_fill_only_touches_stop_routes_in_gap_table(env):
    make, hw = env
    m = make(V2_WAIT="gaps", V2_HEADWAYS=hw, V2_WAIT_MIN_N=3, V2_WAIT_CAP=3600, V2_WAIT_FILL=1, V2_WAIT_UNOBS=0)
    assert w(m, 12) == 3600 and w(m, 8) == 700 and w(m, 9) == 2500
    assert w(m, 12, dt=1) == 300  # weekend has no gap data for this stop-route: timetable wait kept
    assert w(m, 12, stop=1) == 300  # other stop: no gap data, kept
    assert m.stats["wait_cells_v2_filled"] == 22


def test_unobs_scales_stop_routes_without_gap_data(env):
    # only well-observed cell: 08h at stop 100, gap wait 700 vs timetable 300 -> factor 7/3
    make, hw = env
    m = make(V2_WAIT="gaps", V2_HEADWAYS=hw, V2_WAIT_MIN_N=3, V2_WAIT_CAP=3600, V2_WAIT_UNOBS=1, V2_WAIT_FILL=0)
    assert m.stats["wait_unobs_factor"] == pytest.approx(7 / 3, abs=1e-3)
    assert w(m, 12, stop=1) == pytest.approx(700, abs=1)  # unobserved stop: 300 x 7/3
    assert w(m, 12, dt=1) == pytest.approx(700, abs=1)  # unobserved day type at stop 100
    assert w(m, 8) == 700 and w(m, 9) == 2500 and w(m, 12) == 300  # observed stop-route, no fill: unchanged
    m = make(V2_WAIT="gaps", V2_HEADWAYS=hw, V2_WAIT_MIN_N=3, V2_WAIT_CAP=500, V2_WAIT_UNOBS=1, V2_WAIT_FILL=0)
    assert w(m, 12, stop=1) == 500  # capped
