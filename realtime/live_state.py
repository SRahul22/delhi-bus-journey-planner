"""Live traffic state from a stream of bus GPS pings.

For every (vehicle, trip) the pings are projected onto the route shape exactly as the historical pipeline
does it (model/scripts/process_day.py: nearest shape segment, <=150 m off-shape, forward-moving chain,
<=25 m/s progress, interpolation across gaps <=300 s, terminal departure = last ping within 50 m of stop 0).
The GPS route_id is wrong-direction for ~17% of trips, so each trip is also tracked on its paired UP/DOWN
route; the wrong hypothesis never makes forward progress and so never emits anything.

Stop passings -> segment observations (consecutive pattern stops of one trip) -> log(observed / historical
median), where the historical median is whatever the planner's TravelTimeModel uses for that segment, hour
and day type. A snapshot at time `now` uses observations finished in the last `window` seconds (default
30 min) and pools them hierarchically, each level shrunk toward the next (weights = observed minutes):

    segment  --(n0=SEG_N0 min)-->  3x3 km area around the segment  --(AREA_N0)-->  city  --(CITY_N0)-->  1.0

and clips the ratio to [0.6, 2.5]. Every segment of every route gets a ratio, so segments with no recent
bus still use their area's recent conditions.

Main API
    ls = LiveState(net, model)              # planner network + historical TravelTimeModel
    ls.ingest(batch)                        # batch dict from realtime.feed (ts, veh, route, trip, lat, lng)
    snap = ls.refresh(now)                  # immutable snapshot used by readers (thread safe swap)
    ls.live_ratio(route_id, from_stop, to_stop)
    ls.next_bus_eta(stop_id, route_id)      # [(eta_s, vehicle)] from live vehicle positions
"""
import csv
import math
import os
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

import kf as kfmod

# V2_LIVE=ratio (default, v1 behaviour: hierarchical shrunk ratios over the last 30 min) or kf (per-segment
# Kalman state with horizon forecasts, realtime/kf.py). Read at LiveState construction.
V2_LIVE_DEFAULT = "ratio"

IST = timezone(timedelta(hours=5, minutes=30))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# projection (same constants as model/scripts/process_day.py)
LAT0 = 28.6
KX = 111320.0 * np.cos(np.radians(LAT0))
KY = 110540.0
MAXD = 150.0
VMAX = 25.0
MAXGAP = 300.0
RESET_AFTER = 6          # consecutive rejected pings -> restart the chain at the current ping
STATE_TTL = 1200         # forget vehicles silent for 20 min
# ratio model
WINDOW = 1800
SEG_N0, AREA_N0, CITY_N0 = 3.0, 8.0, 30.0   # prior strength in observed minutes
CLIP = (0.6, 2.5)
CELL_M = 1000.0
ETA_MAX_AGE = 180        # vehicle position must be this fresh to count for next_bus_eta


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[-1]


def ist_hour_dt(t):
    d = datetime.fromtimestamp(float(t), IST)
    return d.hour, int(d.weekday() >= 5)


def sod(t):
    """unix -> seconds since IST midnight."""
    return (float(t) + 19800) % 86400


class _Track:
    __slots__ = ("uid", "ri", "lastA", "lastT", "moveT", "rej", "next_k", "prev_k", "prev_t", "at_term", "moved", "veh")

    _n = 0

    def __init__(self, ri, veh):
        _Track._n += 1
        self.uid = _Track._n  # unique id (python id() is reused after garbage collection)
        self.ri = ri
        self.veh = veh
        self.lastA = None
        self.rej = 0
        self.moved = 0.0


class Snapshot:
    """Immutable view used by readers: per flat segment log-ratio and counts, plus live vehicles."""

    def __init__(self, now, logr, n_seg, n_area, city, n_obs, vehicles, kf=None):
        self.now = now
        self.logr = logr            # float32 [n_flat]
        self.ratio = np.exp(logr)
        self.n_seg = n_seg          # observed minutes on the segment itself
        self.n_area = n_area        # observed minutes in its 3x3 km area
        self.city = city            # (ratio, minutes)
        self.n_obs = n_obs
        self.vehicles = vehicles    # ri -> list of (along_m, t_unix, next_k, veh)
        self.kf = kf                # V2_LIVE=kf: kf.SegmentKF.state() dict (horizon forecasts), else None


class LiveState:
    def __init__(self, net, model, shapes_csv=None, patterns_csv=None, routes_txt=None, window=WINDOW, mode=None):
        self.mode = (mode or os.environ.get("V2_LIVE", V2_LIVE_DEFAULT)).strip().lower()
        if self.mode not in ("ratio", "kf"):
            self.mode = V2_LIVE_DEFAULT
        self.net = net
        self.model = model
        self.window = window
        shapes_csv = _first_existing(shapes_csv, os.environ.get("ROUTE_SHAPES"),
                                     os.path.join(REPO, "model", "work", "route_shapes.csv"),
                                     "E:/hackathon-work/route_shapes.csv")
        patterns_csv = _first_existing(patterns_csv, os.path.join(os.environ.get("MODEL_DIR", ""), "route_patterns.csv"),
                                       os.path.join(REPO, "model", "route_patterns.csv"))
        routes_txt = _first_existing(routes_txt, os.path.join(REPO, "data", "gtfs", "routes.txt"),
                                     "E:/hackathon-data/gtfs/routes.txt")
        routes = net["routes"]
        stop_ids = net["stops"].stop_id.astype(str).to_numpy()
        self.ri_of = {}
        for ri, r in enumerate(routes):
            try:
                self.ri_of[int(r["route_id"])] = ri
            except ValueError:
                pass
        # flat segment index
        nseg = [len(r["stops"]) - 1 for r in routes]
        self.off = np.concatenate([[0], np.cumsum(nseg)]).astype(np.int64)
        self.n_flat = int(self.off[-1])
        self.seg_key = {}
        for ri, r in enumerate(routes):
            s = stop_ids[r["stops"]]
            for k in range(len(s) - 1):
                self.seg_key.setdefault((r["route_id"], s[k], s[k + 1]), int(self.off[ri] + k))
        self.stop_pos = {}  # (ri, stop_id) -> first pattern position
        for ri, r in enumerate(routes):
            for k, s in enumerate(stop_ids[r["stops"]]):
                self.stop_pos.setdefault((ri, s), k)
        # area cell of each flat segment (midpoint of its two stops)
        lat = net["stops"].lat.to_numpy(); lon = net["stops"].lon.to_numpy()
        A = np.concatenate([r["stops"][:-1] for r in routes]); B = np.concatenate([r["stops"][1:] for r in routes])
        mx = ((lon[A] + lon[B]) / 2 - 76.6) * KX; my = ((lat[A] + lat[B]) / 2 - 28.1) * KY
        self.gx = np.clip((mx // CELL_M).astype(np.int64), 0, 149)
        self.gy = np.clip((my // CELL_M).astype(np.int64), 0, 149)
        self.cell = self.gx * 150 + self.gy
        # historical seconds per flat segment [n_flat, 2, 24]
        self.hist = np.concatenate(model.seg).astype(np.float32)
        self.seg_dist = np.concatenate([r["dist"] for r in routes]).astype(np.float32)
        self._load_geometry(shapes_csv, patterns_csv)
        self._load_pairs(routes_txt)
        self.tracks = {}
        self.vstate = {}  # (veh, trip) -> list of _Track
        self._obs = defaultdict(list)    # columns of segment observations
        self._pass = defaultdict(list)   # columns of stop passings
        self._obs_np = None
        self._lock = threading.Lock()
        self.snapshot = None
        self.keep_history = False  # True: never trim old observations/passings (offline validation)
        self.stats = dict(pings=0, obs=0, passings=0)
        self.kf = None
        if self.mode == "kf":
            self.kf = kfmod.SegmentKF(self.n_flat, self.gx, self.gy, kfmod.seg_sigma(model), **kfmod.load_params())

    # ------------------------------------------------------------------ setup
    def _load_geometry(self, shapes_csv, patterns_csv):
        shapes = defaultdict(list)
        with open(shapes_csv) as f:
            next(f)
            for line in f:
                r, k, la, lo, d = line.split(",")
                shapes[int(r)].append((float(la), float(lo), float(d)))
        pd_ = defaultdict(list)
        with open(patterns_csv) as f:
            next(f)
            for line in f:
                r, k, s, d = line.split(",")
                pd_[int(r)].append((int(k), float(d)))
        self.G = {}
        for rid, pts in shapes.items():
            ri = self.ri_of.get(rid)
            if ri is None or len(pts) < 2:
                continue
            a = np.array(pts)
            x = (a[:, 1] - 77.0) * KX; y = (a[:, 0] - LAT0) * KY
            dx, dy = np.diff(x), np.diff(y)
            L2 = np.maximum(dx * dx + dy * dy, 1e-9)
            pdist = np.array([d for _, d in sorted(pd_[rid])])
            if len(pdist) != len(self.net["routes"][ri]["stops"]):
                continue
            self.G[rid] = (ri, x[:-1], y[:-1], dx, dy, L2, np.sqrt(L2), a[:-1, 2], pdist)

    def _load_pairs(self, routes_txt):
        self.PAIR = {}
        by = defaultdict(dict)
        if not os.path.exists(routes_txt):
            return
        with open(routes_txt, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sn = row["route_short_name"].strip().upper()
                for suf in ("UP", "DOWN"):
                    if sn.endswith(suf):
                        by[sn[: -len(suf)].rstrip("_- ")][suf] = int(row["route_id"])
        for d in by.values():
            if "UP" in d and "DOWN" in d:
                self.PAIR[d["UP"]] = d["DOWN"]; self.PAIR[d["DOWN"]] = d["UP"]

    # ------------------------------------------------------------------ ingest
    def _project(self, g, lat, lng):
        ri, ax, ay, dx, dy, L2, L, cum, pdist = g
        px = (lng - 77.0) * KX; py = (lat - LAT0) * KY
        X = px[:, None]; Y = py[:, None]
        t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1)
        d = np.hypot(ax + t * dx - X, ay + t * dy - Y)
        i = np.argmin(d, axis=1)
        r_ = np.arange(len(i))
        return cum[i] + t[r_, i] * L[i], d[r_, i]

    def ingest(self, batch):
        ts = np.asarray(batch["ts"])
        if len(ts) == 0:
            return
        veh, trip, route = batch["veh"], batch["trip"], batch["route"]
        lat, lng = batch["lat"], batch["lng"]
        key = veh.astype(np.int64) * 10_000_000 + trip
        order = np.lexsort((ts, key))
        ks = key[order]
        cuts = np.flatnonzero(np.diff(ks)) + 1
        starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [len(ks)]])
        with self._lock:
            for a, b in zip(starts, ends):
                idx = order[a:b]
                vk = (int(veh[idx[0]]), int(trip[idx[0]]))
                rid = int(route[idx[0]])
                tl = self.vstate.get(vk)
                if tl is None:
                    tl = []
                    for r in (rid, self.PAIR.get(rid)):
                        g = self.G.get(r) if r is not None else None
                        if g is not None:
                            tl.append((g, _Track(g[0], vk[0])))
                    self.vstate[vk] = tl
                T = ts[idx].astype(np.float64)
                for g, tr in tl:
                    A, D = self._project(g, lat[idx], lng[idx])
                    self._advance(g, tr, T, A, D)
            self.stats["pings"] += len(ts)

    def _advance(self, g, tr, T, A, D):
        pdist = g[8]
        ri = tr.ri
        nst = len(pdist)
        for t, a, d in zip(T, A, D):
            if d > MAXD:
                continue
            if tr.lastA is None or tr.rej >= RESET_AFTER or t - tr.lastT > MAXGAP * 2:
                tr.lastA, tr.lastT, tr.moveT, tr.rej = a, t, t, 0
                tr.at_term = a <= pdist[0] + 50
                tr.next_k = 0 if tr.at_term else int(np.searchsorted(pdist, a, side="right"))
                tr.prev_k, tr.prev_t = -1, float("nan")
                continue
            if t <= tr.lastT:
                continue
            if a < tr.lastA - 100 or a - tr.lastA > VMAX * (t - tr.moveT) + 100:
                tr.rej += 1
                continue
            tr.rej = 0
            if a <= tr.lastA:
                tr.lastT = t
                continue
            gap = t - tr.lastT
            k = tr.next_k
            while k < nst:
                if k == 0:  # terminal: departure = last ping within 50 m of stop 0 (excludes layover)
                    if a <= pdist[0] + 50:
                        break
                    tp = tr.lastT if gap <= MAXGAP else float("nan")
                elif pdist[k] <= a:
                    tp = tr.lastT + (pdist[k] - tr.lastA) / max(a - tr.lastA, 1e-6) * gap if gap <= MAXGAP else float("nan")
                else:
                    break
                self._emit(ri, k, tp, tr, pdist)
                k += 1
            tr.next_k = k
            tr.moved += a - tr.lastA
            tr.lastA, tr.lastT, tr.moveT = a, t, t

    def _emit(self, ri, k, tp, tr, pdist):
        if tp == tp:
            P = self._pass
            P["ri"].append(ri); P["k"].append(k); P["t"].append(tp); P["veh"].append(tr.veh); P["trk"].append(tr.uid)
            self.stats["passings"] += 1
            if tr.prev_k == k - 1 and tr.prev_t == tr.prev_t:
                dur = tp - tr.prev_t
                dist = pdist[k] - pdist[k - 1]
                if dur >= 0 and dist <= 70 / 3.6 * max(dur, 1e-6) and dur <= max(1800.0, dist / 1.0):
                    O = self._obs
                    O["f"].append(int(self.off[ri] + k - 1)); O["t0"].append(tr.prev_t); O["t1"].append(tp); O["dur"].append(dur)
                    self.stats["obs"] += 1
        tr.prev_k, tr.prev_t = k, tp

    # ------------------------------------------------------------------ observations
    def observations(self):
        """Segment observations as numpy arrays: f (flat seg), t0, t1, dur, hist, logr, w."""
        with self._lock:
            n = len(self._obs["f"])
            if self._obs_np is not None and self._obs_np["n"] == n:
                return self._obs_np
            f = np.array(self._obs["f"], np.int64); t0 = np.array(self._obs["t0"]); t1 = np.array(self._obs["t1"])
            dur = np.array(self._obs["dur"])
        loc = t0 + 19800
        h = ((loc % 86400) // 3600).astype(np.int64)
        dt = (((loc // 86400) + 3) % 7 >= 5).astype(np.int64)  # 1970-01-01 was a Thursday (weekday 3)
        hist = self.hist[f, dt, h] if n else np.zeros(0, np.float32)
        logr = np.clip(np.log(np.maximum(dur, 5) / np.maximum(hist, 5)), math.log(1 / 3), math.log(3))
        w = np.minimum(hist, 300) / 60.0
        self._obs_np = dict(n=n, f=f, t0=t0, t1=t1, dur=dur, hist=hist, logr=logr, w=w)
        return self._obs_np

    def passings(self):
        with self._lock:
            return {k: np.array(v) for k, v in self._pass.items()}

    # ------------------------------------------------------------------ snapshot
    def compute(self, now, window=None, n_max=None):
        """Ratios as of `now` (unix) from observations that finished in (now-window, now].
        n_max: only the first n_max observations (those emitted by `now`; used by offline evaluation)."""
        window = window or self.window
        o = self.observations()
        m = (o["t1"] <= now) & (o["t1"] > now - window)
        if n_max is not None:
            m[n_max:] = False
        f, lr, w = o["f"][m], o["logr"][m], o["w"][m]
        nf = self.n_flat
        S_seg = np.bincount(f, weights=w * lr, minlength=nf)
        N_seg = np.bincount(f, weights=w, minlength=nf)
        Sc = np.zeros((150, 150)); Nc = np.zeros((150, 150))
        np.add.at(Sc, (self.gx[f], self.gy[f]), w * lr)
        np.add.at(Nc, (self.gx[f], self.gy[f]), w)
        # pool 3x3 neighbourhood
        Sp = np.zeros_like(Sc); Np_ = np.zeros_like(Nc)
        P = np.pad(Sc, 1); Q = np.pad(Nc, 1)
        for ddx in range(3):
            for ddy in range(3):
                Sp += P[ddx:ddx + 150, ddy:ddy + 150]; Np_ += Q[ddx:ddx + 150, ddy:ddy + 150]
        Ncity = w.sum()
        city = (w * lr).sum() / (Ncity + CITY_N0)
        area = (Sp + AREA_N0 * city) / (Np_ + AREA_N0)
        a_seg = area[self.gx, self.gy]
        logr = (S_seg + SEG_N0 * a_seg) / (N_seg + SEG_N0)
        logr = np.clip(logr, math.log(CLIP[0]), math.log(CLIP[1])).astype(np.float32)
        return logr, N_seg.astype(np.float32), Np_[self.gx, self.gy].astype(np.float32), (float(np.exp(city)), float(Ncity)), int(m.sum())

    def refresh(self, now):
        logr, n_seg, n_area, city, n_obs = self.compute(now)
        kfs = None
        if self.kf is not None:
            if self.kf.t is not None and now < self.kf.t:   # clock moved back (POST /realtime/clock): restart
                self.kf.reset()
            self.kf.advance(self.observations(), now)
            kfs = self.kf.state()
            logr = kfmod.forecast(kfs, 0.0)[0].astype(np.float32)
        vehicles = defaultdict(list)
        with self._lock:
            dead = []
            for vk, tl in self.vstate.items():
                alive = False
                for g, tr in tl:
                    if tr.lastA is None:
                        continue
                    if now - tr.lastT < STATE_TTL:
                        alive = True
                    if 0 <= now - tr.lastT <= ETA_MAX_AGE and tr.moved >= 200:
                        vehicles[tr.ri].append((float(tr.lastA), float(tr.lastT), int(tr.next_k), tr.veh, g[8]))
                if not alive and tl and all(tr.lastA is not None for _, tr in tl):
                    dead.append(vk)
            for vk in dead:
                del self.vstate[vk]
            # keep observations/passings bounded in a long-running service (2 h)
            if not self.keep_history and len(self._obs["t1"]) > 400_000:
                self._trim(now - 7200)
        self.snapshot = Snapshot(now, logr, n_seg, n_area, city, n_obs, dict(vehicles), kf=kfs)
        return self.snapshot

    def _trim(self, tmin):
        for store, col in ((self._obs, "t1"), (self._pass, "t")):
            keep = [i for i, t in enumerate(store[col]) if t >= tmin]
            if store is self._obs and self.kf is not None:   # keep the KF cursor on the same observation
                self.kf.cursor = int(np.searchsorted(np.array(keep, np.int64), self.kf.cursor))
            for k in list(store):
                v = store[k]
                store[k] = [v[i] for i in keep]
        self._obs_np = None

    # ------------------------------------------------------------------ queries
    def flat(self, route_id, from_stop, to_stop):
        return self.seg_key.get((str(route_id), str(from_stop), str(to_stop)))

    def live_ratio(self, route_id, from_stop, to_stop, snap=None):
        """Recent observed / historical travel time on one route segment (1.0 if unknown)."""
        snap = snap or self.snapshot
        f = self.flat(route_id, from_stop, to_stop)
        if snap is None or f is None:
            return 1.0
        return float(snap.ratio[f])

    def route_ratio(self, ri, snap=None):
        snap = snap or self.snapshot
        return snap.ratio[self.off[ri]:self.off[ri + 1]]

    def arrivals(self, ri, k, snap=None, max_eta=3600):
        """Predicted arrival times (unix) at pattern position k of route ri for live vehicles before it."""
        snap = snap or self.snapshot
        if snap is None:
            return []
        out = []
        o = self.off[ri]
        now = snap.now
        for along, t, nk, veh, pdist in snap.vehicles.get(ri, ()):
            if nk > k or along >= pdist[k] or nk < 1:
                continue
            h, dt = ist_hour_dt(t)
            hseg = self.hist[o:o + len(pdist) - 1, dt, h]
            if snap.kf is not None:
                # horizon of each segment = when the bus reaches it (historical cumulative time from its position)
                hz = np.concatenate([[0.0], np.cumsum(hseg[nk - 1:k])])[:-1] + (t - now)
                lr = kfmod.forecast(snap.kf, np.maximum(hz, 0.0), np.arange(o + nk - 1, o + k))[0]
                fac = np.ones(len(hseg)); fac[nk - 1:k] = np.exp(lr)
                segs = hseg * fac
            else:
                segs = hseg * snap.ratio[o:o + len(pdist) - 1]
            frac = (pdist[nk] - along) / max(pdist[nk] - pdist[nk - 1], 1.0)
            rem = segs[nk - 1] * min(max(frac, 0.0), 1.0) + float(segs[nk:k].sum())
            eta = max(t + rem, now)
            if eta - now <= max_eta:
                out.append((eta, veh))
        out.sort()
        return out

    def next_bus_eta(self, stop_id, route_id, snap=None):
        """[(eta_seconds_from_now, vehicle_idx), ...] for live buses of route_id heading to stop_id."""
        snap = snap or self.snapshot
        ri = self.ri_of.get(int(route_id)) if str(route_id).isdigit() else None
        if ri is None or snap is None:
            return []
        k = self.stop_pos.get((ri, str(stop_id)))
        if k is None:
            return []
        return [(round(e - snap.now), v) for e, v in self.arrivals(ri, k, snap)]

    def status(self):
        s = self.snapshot
        d = dict(self.stats, tracked_vehicles=len(self.vstate))
        if s is not None:
            d.update(now=datetime.fromtimestamp(s.now, IST).isoformat(timespec="seconds"),
                     obs_in_window=s.n_obs, city_ratio=round(s.city[0], 3), city_minutes=round(s.city[1]),
                     segments_with_own_obs=int((s.n_seg > 0).sum()),
                     live_mode=self.mode,
                     live_vehicles=int(sum(len(v) for v in s.vehicles.values())),
                     ratio_p10_p50_p90=[round(float(x), 3) for x in np.percentile(s.ratio[s.n_seg > 0], [10, 50, 90])] if (s.n_seg > 0).any() else None)
        return d
