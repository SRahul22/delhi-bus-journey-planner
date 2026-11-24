"""Glue for the planner app: replay feed + live state + background updater + realtime planner.

    from realtime.service import RealtimeService      # (repo root on sys.path)
    rt = RealtimeService(planner)                      # loads replay, warms up 45 min, starts a thread
    rt.planner_for(depart_dt)  -> Planner using live ratios if depart is on the live day within 90 min of
                                  the live clock, else None
    rt.status()                -> dict for GET /realtime/status

Environment variables:
    RT_REPLAY  replay file      (default realtime/data/replay_28.npz)
    RT_START   simulated start  (default 2026-10-28T08:30, IST)
    RT_SPEED   clock multiplier (default 1 = real time; 10 = ten simulated minutes per minute)
    RT_TICK    seconds between updates (default 15)
"""
import copy
import os
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from feed import IST, DEFAULT_REPLAY, ReplayFeed  # noqa: E402
from live_state import LiveState, sod  # noqa: E402
from rt_model import RealtimeTravelTimeModel  # noqa: E402

WARMUP_S = 2700
HORIZON_S = 5400  # use live ratios for departures up to 90 min after the live clock


class RealtimeService:
    def __init__(self, planner, replay=None, start=None, speed=None, tick=None):
        t0 = time.time()
        self.planner = planner
        self.feed = ReplayFeed(replay or os.environ.get("RT_REPLAY", DEFAULT_REPLAY),
                               start=start or os.environ.get("RT_START", "2026-10-28T08:30"),
                               speed=float(speed or os.environ.get("RT_SPEED", 1)))
        self.tick = float(tick or os.environ.get("RT_TICK", 15))
        self.live = LiveState(planner.net, planner.model)
        now = self.feed.now()
        self.live.ingest(self.feed.window(now - WARMUP_S, now))
        self.feed.poll()
        self.live.refresh(self.feed.now())
        self._rebuild()
        self.error = None
        self.updates = 0
        self.startup_s = round(time.time() - t0, 1)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="realtime-feed")
        self.thread.start()

    def _rebuild(self):
        """(Re)wrap the planner's current historical model, e.g. after /admin/reload-model."""
        self.live.model = self.planner.model
        self.rt_model = RealtimeTravelTimeModel(self.planner.model, self.live)
        p = copy.copy(self.planner)
        p.model = self.rt_model
        self.rt_planner = p

    def reload(self):
        import numpy as np
        self.live.hist = np.concatenate(self.planner.model.seg).astype(np.float32)
        self.live._obs_np = None
        self._rebuild()

    def _run(self):
        while not self._stop.wait(self.tick):
            try:
                self.live.ingest(self.feed.poll())
                self.live.refresh(self.feed.now())
                self.updates += 1
            except Exception as e:  # keep serving historical answers if the feed breaks
                self.error = repr(e)

    def stop(self):
        self._stop.set()

    def set_clock(self, start, speed=None):
        """Jump the replay clock (re-warms the live state from the replay)."""
        self.feed.set_clock(start, speed)
        self.live = LiveState(self.planner.net, self.planner.model)
        now = self.feed.now()
        self.live.ingest(self.feed.window(now - WARMUP_S, now))
        self.live.refresh(now)
        self._rebuild()

    def now_dt(self):
        return datetime.fromtimestamp(self.feed.now(), IST)

    def planner_for(self, depart):
        """Realtime planner if `depart` (datetime or None) is on the live day and near the live clock."""
        snap = self.live.snapshot
        if snap is None:
            return None
        if depart is None:
            return self.rt_planner
        if depart.tzinfo is None:
            depart = depart.replace(tzinfo=IST)
        delta = depart.timestamp() - snap.now
        if -900 <= delta <= HORIZON_S and sod(depart.timestamp()) >= sod(snap.now) - 900:
            return self.rt_planner
        return None

    def status(self):
        d = dict(enabled=True, source="replay " + os.path.basename(self.feed.path), clock=self.feed.now_iso(),
                 speed=self.feed.speed, tick_s=self.tick, updates=self.updates, startup_s=self.startup_s,
                 error=self.error)
        d.update(self.live.status())
        return d
