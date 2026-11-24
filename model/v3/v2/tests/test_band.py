"""pytest for the V2_BAND path of model/v3/predict.py (T1.1/T1.2). Run from the repo root:
    E:\\hackathon-project\\planner\\.venv\\Scripts\\python.exe -m pytest model\\v3\\v2\\tests -q
Uses check_rides.json only for (route, stops, departure) inputs; actual times are not read."""
import importlib, json, os, sys
from datetime import datetime
from pathlib import Path
import pytest

V3 = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def P():
    os.environ["V3_MODE"] = "xgb"; os.environ["V2_BAND"] = "1"
    sys.path.insert(0, str(V3))
    import predict
    importlib.reload(predict)
    return predict


def legs():
    chk = json.load(open(V3 / "check_rides.json"))[:200]
    return [(c["route_id"], c["board_stop"], c["alight_stop"], datetime.strptime(c["depart_ist"], "%Y-%m-%d %H:%M:%S")) for c in chk]


def test_p85_at_least_p50(P):
    for r, b, a, d in legs():
        p50, p85 = P.band(r, b, a, d)
        assert p50 is not None and p85 is not None and p85 >= p50 > 0


def test_fallback_unknown_route(P):
    p50, p85 = P.band(-1, 1, 2, datetime(2026, 10, 21, 9, 0), 1200.0, 1300.0)
    assert p85 >= p50 > 0


def test_peak_definition(P):
    assert P._is_peak(datetime(2026, 10, 21, 9, 0))          # Wednesday
    assert not P._is_peak(datetime(2026, 10, 21, 12, 0))
    assert P._is_peak(datetime(2026, 10, 21, 20, 59))
    assert not P._is_peak(datetime(2026, 10, 20, 9, 0))      # Dussehra
    assert not P._is_peak(datetime(2026, 10, 24, 9, 0))      # Saturday


def test_shift_keys_complete(P):
    m = json.load(open(V3 / "v2" / "conformal_shifts.json"))
    keys = {f"{nm}|{p}" for _, _, nm in m["buckets_s"] for p in ("peak", "offpeak")}
    assert keys <= set(m["shifts"]) and keys <= set(m["fallback_log_ratio"])


def test_flag_off_returns_none(P):
    P.V2_BAND = False
    try:
        r, b, a, d = legs()[0]
        assert P.band(r, b, a, d)[1] is None
    finally:
        P.V2_BAND = True
