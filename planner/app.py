"""FastAPI app: Delhi bus journey planner.  Run: ./run.sh  (uvicorn app:app --port 8000)"""
import hmac
import os
import time
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import router
import router as _router  # noqa: F811  (v2/planner name; same module)
import v2ext
from router import Planner

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Delhi Bus Journey Planner", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
planner = Planner()
STARTED = time.time()

# T4.5: admin endpoints (POST /admin/reload-model, POST /realtime/clock) need header X-Admin-Token equal to
# ADMIN_TOKEN. With ADMIN_TOKEN unset they are disabled (403), per AGENTS.md section 5.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN") or None


def require_admin(x_admin_token: Optional[str] = Header(None)):
    if ADMIN_TOKEN is None:
        raise HTTPException(403, "admin endpoints are disabled (ADMIN_TOKEN not set on the server)")
    if not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), ADMIN_TOKEN.encode()):
        raise HTTPException(401, "missing or invalid X-Admin-Token")

# Optional real-time layer (../realtime): live GPS ratios + next-bus ETAs, fed by a replay of 28 Oct 2026.
# Used per request with ?realtime=1. Disable loading with REALTIME=0.
RT = None
if os.environ.get("REALTIME", "1") != "0":
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "realtime"))
        from service import RealtimeService
        RT = RealtimeService(planner)
    except Exception as e:  # noqa: BLE001  (missing replay file etc. -> historical only)
        print("realtime layer disabled:", repr(e))


def preload():
    """T4.5: load every model at startup (no lazy load on the first request). Returns what is loaded."""
    out = dict(network=True, ttmodel=planner.model.source, v3=None, realtime=RT is not None,
               pooled_waits=None, band=os.environ.get("V2_BAND", "0"), d2d_shifts=None)
    v3 = os.environ.get("V3_MODE", "xgb")
    if _router.v3_correct is not None and v3 != "off":
        try:
            import predict
            m = predict._model()
            # single-row predictions: one thread is ~30x faster than xgboost's default thread pool (9 ms -> 0.3 ms)
            if getattr(m, "xgb", None) is not None:
                m.xgb.set_param({"nthread": 1})
            out["v3"] = dict(mode=v3, booster="xgboost" if getattr(m, "xgb", None) is not None else "trees")
            if v2ext.band_on() and hasattr(predict, "_bandm"):
                bm = predict._bandm()
                bm.q.set_param({"nthread": 1})
                out["v3"]["band"] = "quantile+conformal (ml)"
        except Exception as e:  # noqa: BLE001
            out["v3"] = "failed: " + repr(e)
    if v2ext.wait_mode() == "pooled":
        pw = v2ext.pooled()
        out["pooled_waits"] = getattr(pw, "path", "unavailable (v1 waits)")
    if v2ext.band_on():
        out["d2d_shifts"] = len(v2ext.d2d_shifts(planner.model.model_dir))
    return out


MODELS_LOADED = preload()
print("models loaded:", MODELS_LOADED)

# Delhi NCR bounding box (generous)
BBOX = (28.2, 29.0, 76.7, 77.6)


def _check(lat, lng, name):
    if not (BBOX[0] <= lat <= BBOX[1] and BBOX[2] <= lng <= BBOX[3]):
        raise HTTPException(400, f"{name} ({lat},{lng}) is outside the Delhi service area")


@app.get("/health")
def health():
    mode = os.environ.get("V3_MODE", "xgb")
    return dict(status="ok", routes=len(planner.routes), stops=planner.N, model=planner.model.source,
                ride_correction=f"model v3 ({mode})" if router.v3_correct is not None and mode != "off" else "off",
                model_stats=planner.model.stats, uptime_s=round(time.time() - STARTED),
                models_loaded=MODELS_LOADED, admin_enabled=ADMIN_TOKEN is not None,
                flags=dict(V2_WAIT=v2ext.wait_mode(), V2_BAND=os.environ.get("V2_BAND", "0"),
                           V3_MIN_RIDE_S=router.v3_min_ride_s(),
                           POOLED_INDEX=os.environ.get("POOLED_INDEX", v2ext.DEFAULT_INDEX)))


def _resolve(text, lat, lng, name):
    """An endpoint from either lat/lng params or free text: "28.66,77.22" or a stop name (best name match)."""
    if lat is not None and lng is not None:
        _check(lat, lng, name)
        return lat, lng, None
    if not text or not text.strip():
        key = "from" if name == "origin" else "to"
        raise HTTPException(400, f"give the {name} as {key}_lat and {key}_lng, or as '{key}' (stop name or 'lat,lng')")
    parts = text.split(",")
    if len(parts) == 2:
        try:
            la, ln = float(parts[0]), float(parts[1])
        except ValueError:
            pass
        else:
            _check(la, ln, name)
            return la, ln, None
    hits = planner.search_stops(text, 1)
    if not hits:
        raise HTTPException(404, f"no stop matches {name} '{text}'; try /stops/search?q=...")
    h = hits[0]
    return h["lat"], h["lng"], dict(query=text, name=h["name"], stop_id=h["stop_id"], lat=h["lat"], lng=h["lng"])


@app.get("/journey")
def journey(from_: Optional[str] = Query(None, alias="from", description="origin: stop name or 'lat,lng'"),
            to: Optional[str] = Query(None, description="destination: stop name or 'lat,lng'"),
            from_lat: Optional[float] = None, from_lng: Optional[float] = None,
            to_lat: Optional[float] = None, to_lng: Optional[float] = None,
            depart: Optional[str] = Query(None, description="ISO datetime, IST if no offset; default now"),
            max_options: int = Query(3, ge=1, le=6),
            realtime: bool = Query(False, description="apply live GPS conditions (replay clock if depart omitted)"),
            arrive_by: Optional[str] = Query(None, description="ISO datetime deadline (IST if no offset); adds "
                                             "on_time_probability per option and ranks by it")):
    from_lat, from_lng, from_stop = _resolve(from_, from_lat, from_lng, "origin")
    to_lat, to_lng, to_stop = _resolve(to, to_lat, to_lng, "destination")
    dep = None
    if depart:
        try:
            dep = datetime.fromisoformat(depart.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(400, "depart must be an ISO datetime, e.g. 2025-11-25T09:00")
    arr = None
    if arrive_by:
        try:
            arr = datetime.fromisoformat(arrive_by.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(400, "arrive_by must be an ISO datetime, e.g. 2026-10-21T10:00")
    p = planner
    if realtime and RT is not None:
        dep = dep or RT.now_dt()
        p = RT.planner_for(dep) or planner
    # the quiet-hour reference always uses the historical model (live ratios describe "now", not other hours)
    res = p.plan(from_lat, from_lng, to_lat, to_lng, dep, max_options=max_options, quiet_planner=planner,
                 arrive_by=arr)
    if from_stop or to_stop:
        res["query"]["resolved"] = dict(origin=from_stop, destination=to_stop)
    if realtime and RT is not None and p is not planner:
        _attach_live_etas(res)
    if realtime:
        res["realtime"] = dict(applied=p is not planner, clock=RT.feed.now_iso() if RT else None,
                               note=None if p is not planner else "historical only: realtime unavailable or depart not near the live clock")
    return res


def _attach_live_etas(res):
    """Add live next-bus ETAs (from the replay clock) to the first bus leg of each option."""
    try:
        clock = RT.feed.now()
        for o in res["options"]:
            leg = next((l for l in o["legs"] if l["mode"] == "bus"), None)
            if leg is None:
                continue
            etas = [e for e, _ in RT.live.next_bus_eta(leg["from_stop"]["stop_id"], leg["route_id"])]
            if not etas:
                continue
            # seconds from the live clock until the rider reaches the boarding stop
            reach = (datetime.fromisoformat(leg["arrive_at_stop"]).timestamp() - clock)
            catch = [e for e in etas if e >= reach]
            leg["live_next_buses_s"] = etas[:3]
            leg["live_catchable_bus_s"] = catch[0] if catch else None
            leg["live_wait_s"] = round(catch[0] - reach) if catch else None
    except Exception as e:  # noqa: BLE001
        res["realtime_eta_error"] = repr(e)


@app.get("/realtime/status")
def realtime_status():
    return RT.status() if RT else dict(enabled=False)


@app.post("/realtime/clock", dependencies=[Depends(require_admin)])
def realtime_clock(start: str, speed: float = 1.0):
    """Jump the replay clock, e.g. start=2026-10-28T18:00&speed=10."""
    if RT is None:
        raise HTTPException(404, "realtime layer not loaded")
    RT.set_clock(start, speed)
    return RT.status()


@app.get("/realtime/eta")
def realtime_eta(stop_id: str, route_id: str):
    """Live next-bus ETAs (seconds from the live clock) for a route at a stop."""
    if RT is None:
        raise HTTPException(404, "realtime layer not loaded")
    return dict(clock=RT.feed.now_iso(), etas_s=[e for e, _ in RT.live.next_bus_eta(stop_id, route_id)])


@app.get("/stops/search")
def stops_search(q: str = Query(..., min_length=2), limit: int = Query(15, ge=1, le=50)):
    return dict(results=planner.search_stops(q, limit))


@app.post("/admin/reload-model", dependencies=[Depends(require_admin)])
def reload_model():
    """Re-read model CSVs from the model dir (after the model agent writes new tables)."""
    global MODELS_LOADED
    planner.reload_model()
    if RT is not None:
        RT.reload()
    MODELS_LOADED = preload()
    return dict(model=planner.model.source, stats=planner.model.stats)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))
