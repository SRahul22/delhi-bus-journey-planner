"""Replay of one day's bus GPS feed with a simulated clock.

The replay file (realtime/data/replay_DD.npz, made on Windows by scripts/extract_replay.py) holds one ping
per vehicle every >= 30 s, 06:00-22:00 IST, sorted by timestamp:
  ts (int64 unix UTC), veh, route (GTFS route_id), trip (int32 indices), lat5/lng5 (int32, degrees*1e5)

    feed = ReplayFeed("realtime/data/replay_28.npz", start="2026-10-28T08:00", speed=10)
    feed.now()                    # simulated unix time (advances `speed` x wall clock)
    batch = feed.poll()           # all pings since the previous poll, up to now()
    feed.latest()                 # latest position per vehicle seen in the last 5 min

A real deployment would replace ReplayFeed with a client of the live DTC/DIMTS feed that returns the same
batch dict; nothing downstream depends on the replay.
"""
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPLAY = next((p for p in (os.path.join(HERE, "data", "replay_28.npz"),
                       "E:/hackathon-work/realtime/replay_28.npz") if os.path.exists(p)),
                      os.path.join(HERE, "data", "replay_28.npz"))


def parse_ist(s):
    """ISO string (IST if no offset) or unix seconds -> unix seconds."""
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=IST)
    return d.timestamp()


class ReplayFeed:
    def __init__(self, path=DEFAULT_REPLAY, start=None, speed=1.0):
        z = np.load(path)
        self.ts = z["ts"].astype(np.int64)
        self.veh = z["veh"]
        self.route = z["route"]
        self.trip = z["trip"]
        self.lat = z["lat5"].astype(np.float64) / 1e5
        self.lng = z["lng5"].astype(np.float64) / 1e5
        self.veh_names = z["veh_names"] if "veh_names" in z else None
        self.t_first, self.t_last = int(self.ts[0]), int(self.ts[-1])
        self.path = path
        self._lock = threading.Lock()
        self.speed = float(speed)
        self.set_clock(start if start is not None else self.t_first + 2 * 3600)

    # ---------------------------------------------------------------- clock
    def set_clock(self, start, speed=None):
        """Jump the simulated clock to `start` (ISO IST string or unix s). The next poll() returns nothing
        older than `start`; use poll(since=...) to warm up a state from earlier pings."""
        with self._lock:
            if speed is not None:
                self.speed = float(speed)
            self._sim0 = parse_ist(start)
            self._wall0 = time.time()
            self._cursor = int(np.searchsorted(self.ts, self._sim0, side="right"))

    def now(self):
        t = self._sim0 + (time.time() - self._wall0) * self.speed
        return min(t, float(self.t_last))

    def now_iso(self):
        return datetime.fromtimestamp(self.now(), IST).isoformat(timespec="seconds")

    # ---------------------------------------------------------------- data
    def _slice(self, a, b):
        return dict(ts=self.ts[a:b], veh=self.veh[a:b], route=self.route[a:b], trip=self.trip[a:b],
                    lat=self.lat[a:b], lng=self.lng[a:b])

    def window(self, t0, t1):
        """All pings with t0 < ts <= t1 (does not move the cursor)."""
        a = int(np.searchsorted(self.ts, t0, side="right"))
        b = int(np.searchsorted(self.ts, t1, side="right"))
        return self._slice(a, b)

    def poll(self):
        """Pings that arrived since the last poll, up to the simulated now()."""
        with self._lock:
            b = int(np.searchsorted(self.ts, self.now(), side="right"))
            a, self._cursor = self._cursor, max(self._cursor, b)
        return self._slice(a, b)

    def latest(self, max_age=300):
        """Latest ping per vehicle within max_age seconds of now(): dict of arrays."""
        t = self.now()
        w = self.window(t - max_age, t)
        if len(w["ts"]) == 0:
            return w
        # last occurrence of each vehicle (arrays are time sorted)
        _, idx = np.unique(w["veh"][::-1], return_index=True)
        idx = len(w["veh"]) - 1 - idx
        return {k: v[idx] for k, v in w.items()}
