"""V3_MIN_RIDE_S: the v3 (XGBoost) ride correction applies only to bus legs whose planner ride estimate is >= it."""
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import router  # noqa: E402


def test_default_is_config_c(monkeypatch):
    monkeypatch.delenv("V3_MIN_RIDE_S", raising=False)
    assert router.v3_min_ride_s() == 3600.0
    assert not router.v3_applies(30) and router.v3_applies(7200)


def test_zero_applies_everywhere(monkeypatch):
    monkeypatch.setenv("V3_MIN_RIDE_S", "0")
    assert router.v3_min_ride_s() == 0.0
    assert router.v3_applies(0) and router.v3_applies(30) and router.v3_applies(7200)


def test_threshold(monkeypatch):
    monkeypatch.setenv("V3_MIN_RIDE_S", "3600")
    assert router.v3_min_ride_s() == 3600.0
    assert not router.v3_applies(3599.9)
    assert router.v3_applies(3600) and router.v3_applies(5000)


def test_bad_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("V3_MIN_RIDE_S", "abc")
    assert router.v3_min_ride_s() == float(router.DEFAULT_V3_MIN_RIDE_S)


@pytest.mark.skipif(not os.path.exists(router.NET_PATH) and not os.environ.get("NET_PATH"), reason="no network.pkl")
def test_time_journey_gates_correction(monkeypatch):
    """With a fake correction (x2), a leg below the threshold keeps the planner's own ride; above it is doubled."""
    P = router.Planner(net_path=os.environ.get("NET_PATH", router.NET_PATH))
    monkeypatch.setattr(router, "v3_correct", lambda *a, **k: 2.0 * a[4])
    monkeypatch.setattr(router, "v3_band", None)
    monkeypatch.setenv("V3_MODE", "xgb")
    ri = max(range(len(P.routes)), key=lambda r: len(P.route_stops[r]))
    j = len(P.route_stops[ri]) - 1
    base = datetime(2026, 10, 21, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    t0, dt = 9 * 3600, 1
    s0, s1 = P.route_stops[ri][0], P.route_stops[ri][j]
    o, d = (P.stop_lat[s0], P.stop_lon[s0]), (P.stop_lat[s1], P.stop_lon[s1])
    skel = [("bus", ri, 0, j)]

    def ride_with(thr):
        monkeypatch.setenv("V3_MIN_RIDE_S", str(thr))
        return [l for l in P.time_journey(skel, o, d, t0, dt, base)["legs"] if l["mode"] == "bus"][0]["ride_s"]

    corrected, plain = ride_with(0), ride_with(10 ** 9)
    assert abs(corrected - 2 * plain) <= 1
    assert ride_with(plain - 5) == corrected  # estimate above threshold -> corrected
    assert ride_with(plain + 5) == plain      # estimate below threshold -> planner's own ride
