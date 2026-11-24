"""API tests for T4.5 (admin token, preload) and T2.4 (arrive_by). Loads the planner once (~20-40 s)."""
import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("REALTIME", "0")

OD = dict(from_lat=28.6328, from_lng=77.2197, to_lat=28.5672, to_lng=77.2100)  # Connaught Place -> AIIMS


@pytest.fixture(scope="module")
def app_mod():
    os.environ["ADMIN_TOKEN"] = "test-token"
    import app
    importlib.reload  # noqa: B018  (module loaded once per session)
    return app


@pytest.fixture(scope="module")
def client(app_mod):
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app)


def test_health_reports_models_loaded(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    ml = h["models_loaded"]
    assert ml["network"] is True and ml["ttmodel"]
    assert h["admin_enabled"] is True


def test_admin_requires_token(client):
    assert client.post("/admin/reload-model").status_code == 401
    assert client.post("/admin/reload-model", headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.post("/realtime/clock?start=2026-10-28T08:30").status_code == 401


def test_admin_disabled_without_token(app_mod, client):
    old = app_mod.ADMIN_TOKEN
    app_mod.ADMIN_TOKEN = None
    try:
        assert client.post("/admin/reload-model", headers={"X-Admin-Token": "test-token"}).status_code == 403
    finally:
        app_mod.ADMIN_TOKEN = old


def test_journey_arrive_by(client):
    r = client.get("/journey", params=dict(OD, depart="2026-10-21T09:00", arrive_by="2026-10-21T09:50")).json()
    opts = r["options"]
    assert opts and r["query"]["arrive_by"].startswith("2026-10-21T09:50")
    ps = [o["on_time_probability"] for o in opts]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert ps == sorted(ps, reverse=True)
    assert any("most_likely_on_time" in o["labels"] for o in opts)
    for o in opts:
        assert o["duration_p85_s"] >= o["duration_s"]
        assert o["band_version"]


def test_journey_without_arrive_by_unchanged_shape(client):
    r = client.get("/journey", params=dict(OD, depart="2026-10-21T09:00")).json()
    assert r["options"] and all("on_time_probability" not in o for o in r["options"])


def test_arrive_by_bad_value(client):
    assert client.get("/journey", params=dict(OD, arrive_by="tomorrow")).status_code == 400
