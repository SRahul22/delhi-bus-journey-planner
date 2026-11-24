"""Frequency-based RAPTOR journey planner over GTFS route patterns.

Waits are frequency based (headway/2, capped) rather than following scheduled departures, because
the Delhi feed is not schedule-adherent. Ride times come from TravelTimeModel (observed GPS segment
medians when available, schedule-based fallback otherwise).
"""
import os
import pickle
import re
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.spatial import cKDTree

from ttmodel import TravelTimeModel

# Final ride-time correction: XGBoost model v3 (model/v3/predict.py, trained by model/v3/scripts/xgb_final.py).
# V3_MODE=xgb (default) | lgbm | calib | off. If xgboost or the model files are missing, the plain model is used.
import sys as _sys
os.environ.setdefault("V3_MODE", "xgb")
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "v3"))
try:
    from predict import correct as v3_correct
except Exception as _e:  # pragma: no cover
    print("model v3 not loaded:", _e); v3_correct = None

HERE = os.path.dirname(os.path.abspath(__file__))
NET_PATH = os.path.join(HERE, "cache", "network.pkl")
IST = timezone(timedelta(hours=5, minutes=30))
WALK_SPEED = 1.2
DETOUR = 1.3
ACCESS_RADII = (800, 1500, 2500)
MAX_ACCESS_STOPS = 60
MAX_ROUNDS = 4  # = max 3 transfers
TRANSFER_PENALTY = 600  # s per transfer, ranking only (riders prefer a direct bus)
WAIT_WEIGHT = 0.5  # extra perceived cost per second of waiting (ranking only)
DIRECT_KEEP = 1.2  # always show the best direct bus if within 20% of the fastest
MIN_TRANSFER_GAIN = 180  # s  # seconds, used for ranking only
DIRECT_WALK_MAX = 2500  # metres: offer walk-only option below this
# "Usual time at a quieter hour": each returned option is re-timed at these departure hours (same date, so the
# same weekday/weekend type) and the one with the shortest time on the bus is reported as the quiet-hour reference.
QUIET_HOURS = (6, 11, 15, 22)
INF = float("inf")
LAT0 = 28.6
M_LAT = 111_320.0
M_LON = 111_320.0 * np.cos(np.radians(LAT0))


def _xy(lat, lon):
    return np.column_stack([np.atleast_1d(lat) * M_LAT, np.atleast_1d(lon) * M_LON])


def _dist_m(lat1, lon1, lat2, lon2):
    return float(np.hypot((lat1 - lat2) * M_LAT, (lon1 - lon2) * M_LON))


def _display_name(short):
    """Short label for a route: strip UP/DOWN; sentence-like names (e.g. 'ANAND VIHAR ISBT TO GURGAON UP')
    are title-cased and truncated."""
    name = re.sub(r"[\s-]*(UP|DOWN|Up|Down)$", "", short.strip()).strip() or short
    if " " in name and len(name) > 14:
        name = name.title()
        if len(name) > 22:
            name = name[:21].rstrip(" -/") + "…"
    return name


class Planner:
    def __init__(self, net_path=NET_PATH, model_dir=None):
        t0 = time.time()
        if not os.path.exists(net_path):
            import preprocess
            preprocess.main()
        with open(net_path, "rb") as f:
            self.net = pickle.load(f)
        st = self.net["stops"]
        self.stop_lat = st.lat.to_numpy()
        self.stop_lon = st.lon.to_numpy()
        self.stop_name = st.stop_name.astype(str).tolist()
        self.stop_id = st.stop_id.astype(str).tolist()
        self.N = len(st)
        self.tree = cKDTree(_xy(self.stop_lat, self.stop_lon))
        self.routes = self.net["routes"]
        self.route_stops = [r["stops"].tolist() for r in self.routes]
        self.stop_routes = [[] for _ in range(self.N)]
        for ri, ss in enumerate(self.route_stops):
            for pos, s in enumerate(ss):
                self.stop_routes[s].append((ri, pos))
        self.fp = {s: [(n, w) for n, w, _ in lst] for s, lst in self.net["footpaths"].items()}
        self.route_by_id = {r["route_id"]: ri for ri, r in enumerate(self.routes)}
        self._by_name = {}
        for ri, r in enumerate(self.routes):
            self._by_name.setdefault(_display_name(r["short_name"]), []).append(ri)
        self._names_lower = [n.lower() for n in self.stop_name]
        self.model = TravelTimeModel(self.net, **({"model_dir": model_dir} if model_dir else {}))
        self.load_s = time.time() - t0

    def reload_model(self):
        self.model = TravelTimeModel(self.net, model_dir=self.model.model_dir)

    # ---------------------------------------------------------------- helpers
    def nearby(self, lat, lon):
        p = _xy(lat, lon)[0]
        for r in ACCESS_RADII:
            idx = self.tree.query_ball_point(p, r)
            if idx:
                break
        out = []
        for s in idx:
            d = _dist_m(lat, lon, self.stop_lat[s], self.stop_lon[s]) * DETOUR
            out.append((d, s))
        out.sort()
        return {s: d / WALK_SPEED for d, s in out[:MAX_ACCESS_STOPS]}

    def search_stops(self, q, limit=15):
        q = q.strip().lower()
        if not q:
            return []
        seen = {}
        for i, n in enumerate(self._names_lower):
            if q in n:
                key = self.stop_name[i]
                rank = (0 if n.startswith(q) else 1, len(n))
                if key not in seen:
                    seen[key] = [rank, i, 1]
                else:
                    seen[key][2] += 1
        items = sorted(seen.values(), key=lambda x: x[0])[:limit]
        return [dict(name=self.stop_name[i], stop_id=self.stop_id[i], lat=float(self.stop_lat[i]),
                     lng=float(self.stop_lon[i]), n_stops=c, n_routes=len(self.stop_routes[i])) for _, i, c in items]

    # ---------------------------------------------------------------- RAPTOR
    def raptor(self, access, egress, t0, dt, walk_mult=1.0, banned=frozenset(), max_rounds=MAX_ROUNDS):
        best = [INF] * self.N
        prev = {}
        par0 = {}
        for s, w in access.items():
            t = t0 + w * walk_mult
            prev[s] = t
            best[s] = t
            par0[s] = ("access",)
        parents = [par0]
        bus_parents = [{}]
        marked = set(access)
        target_best = INF
        results = []  # (k, egress_stop, cost)
        route_stops = self.route_stops
        stop_routes = self.stop_routes
        model = self.model
        for k in range(1, max_rounds + 1):
            Q = {}
            for s in marked:
                for ri, pos in stop_routes[s]:
                    if ri in banned:
                        continue
                    if pos < Q.get(ri, 1 << 30):
                        Q[ri] = pos
            arr_k, par_k, bus_k = {}, {}, {}
            for ri, p0 in Q.items():
                ss = route_stops[ri]
                last = len(ss) - 1
                bpos = -1
                t_board = 0.0
                cumr = None
                for i in range(p0, last + 1):
                    s = ss[i]
                    ta = INF
                    if bpos >= 0:
                        ta = t_board + cumr[i] - cumr[bpos]
                        if ta < best[s] and ta < target_best:
                            best[s] = ta
                            arr_k[s] = ta
                            bus_k[s] = (ri, bpos, i)
                            par_k[s] = ("bus", ri, bpos, i)
                    if i < last:
                        pt = prev.get(s)
                        if pt is not None:
                            h = int(pt // 3600) % 24
                            tb = pt + model.waits(dt, h)[ri][i]
                            if tb < ta:
                                t_board = tb
                                bpos = i
                                cumr = model.cum(dt, int(tb // 3600) % 24)[ri]
            # footpaths from bus-reached stops
            for s, (ri, bp, i) in list(bus_k.items()):
                base = arr_k[s] if par_k[s][0] == "bus" else None
                if base is None:
                    continue
                for n, w in self.fp.get(s, ()):
                    tn = base + w * walk_mult
                    if tn < best[n] and tn < target_best:
                        best[n] = tn
                        arr_k[n] = tn
                        par_k[n] = ("walk", s, w)
            for s, w in egress.items():
                ta = arr_k.get(s)
                if ta is not None:
                    c = ta + w * walk_mult
                    if c < target_best:
                        # an extra bus leg must save a meaningful amount of time to be reported
                        if not results or results[-1][0] == k or c < results[-1][2] - MIN_TRANSFER_GAIN:
                            results.append((k, s, c))
                        target_best = c
            parents.append(par_k)
            bus_parents.append(bus_k)
            prev = arr_k
            marked = set(arr_k)
            if not marked:
                break
        # best per round (last improvement in each round)
        per_round = {}
        for k, s, c in results:
            per_round[k] = (s, c)
        journeys = []
        for k, (s, c) in sorted(per_round.items()):
            legs = self._backtrack(parents, k, s)
            if legs is not None:
                journeys.append(legs)
        return journeys

    def _silly(self, skel):
        bus = [l for l in skel if l[0] == "bus"]
        for a, b in zip(bus, bus[1:]):
            ra, rb = self.routes[a[1]], self.routes[b[1]]
            if _display_name(ra["short_name"]) == _display_name(rb["short_name"]):
                return True  # re-boarding the same route
        if len(bus) > 1 and any(l[3] - l[2] <= 1 for l in bus):
            return True  # a 1-stop hop inside a multi-bus journey
        return False

    def _backtrack(self, parents, k, s):
        legs = []
        cur = s
        while k > 0:
            p = parents[k].get(cur)
            if p is None:
                return None
            if p[0] == "walk":
                legs.append(("walk", p[1], cur))
                cur = p[1]
                p = parents[k].get(cur)
                if p is None or p[0] != "bus":
                    return None
            _, ri, bpos, i = p
            legs.append(("bus", ri, bpos, i))
            cur = self.route_stops[ri][bpos]
            k -= 1
        if parents[0].get(cur) is None:
            return None
        legs.append(("access", cur))
        legs.reverse()
        return legs

    # ---------------------------------------------------------------- timing / output
    def _stop_obj(self, s):
        return dict(stop_id=self.stop_id[s], name=self.stop_name[s], lat=float(self.stop_lat[s]), lng=float(self.stop_lon[s]))

    def _polyline(self, ri, i, j):
        r = self.routes[ri]
        if r["shape"] is not None and r["shape_idx"] is not None:
            a, b = int(r["shape_idx"][i]), int(r["shape_idx"][j])
            if b > a:
                pts = r["shape"][a:b + 1]
                if len(pts) > 400:
                    pts = np.vstack([pts[::len(pts) // 300 + 1], pts[-1:]])
                s0, s1 = self.route_stops[ri][i], self.route_stops[ri][j]
                pts = np.vstack([[self.stop_lat[s0], self.stop_lon[s0]], pts, [self.stop_lat[s1], self.stop_lon[s1]]])
                return np.round(pts.astype(float), 6).tolist()
        ss = self.route_stops[ri][i:j + 1]
        return [[round(float(self.stop_lat[s]), 6), round(float(self.stop_lon[s]), 6)] for s in ss]

    def alt_routes(self, ri, i, j, limit=6):
        """Other route patterns that serve board stop -> alight stop in order (any of them can be taken)."""
        s0, s1 = self.route_stops[ri][i], self.route_stops[ri][j]
        out = []
        for rj, p in self.stop_routes[s0]:
            if rj == ri:
                continue
            ss = self.route_stops[rj]
            try:
                q = ss.index(s1, p + 1)
            except ValueError:
                continue
            if q - p <= (j - i) * 1.5 + 3:
                out.append(_display_name(self.routes[rj]["short_name"]))
        return sorted(set(out))[:limit]

    def time_journey(self, skel, o, d, t0, dt, base_date):
        """Exact time-dependent timing of a leg skeleton. o,d = (lat,lon)."""
        model = self.model
        legs = []
        t = t0
        total_walk = 0.0
        p85_extra = 0.0  # ride: summed (traffic is correlated along a trip)
        wait_var = 0.0   # wait: p85 of a uniform wait is 1.7x its mean; independent across legs
        cur_pt = o
        nbus = 0

        def ts(sec):
            return (base_date + timedelta(seconds=sec)).isoformat()

        def walk_leg(a_pt, b_pt, a_name, b_name):
            nonlocal t, total_walk
            dm = _dist_m(a_pt[0], a_pt[1], b_pt[0], b_pt[1]) * DETOUR
            if dm < 1:
                return
            dur = dm / WALK_SPEED
            legs.append(dict(mode="walk", from_name=a_name, to_name=b_name, distance_m=round(dm), duration_s=round(dur),
                             depart=ts(t), arrive=ts(t + dur),
                             polyline=[[a_pt[0], a_pt[1]], [b_pt[0], b_pt[1]]]))
            t += dur
            total_walk += dm

        prev_name = "Origin"
        for leg in skel:
            if leg[0] == "access":
                continue
            if leg[0] == "walk":
                a, b = leg[1], leg[2]
                walk_leg((self.stop_lat[a], self.stop_lon[a]), (self.stop_lat[b], self.stop_lon[b]),
                         self.stop_name[a], self.stop_name[b])
                cur_pt = (self.stop_lat[b], self.stop_lon[b])
                prev_name = self.stop_name[b]
                continue
            _, ri, i, j = leg
            r = self.routes[ri]
            s0, s1 = self.route_stops[ri][i], self.route_stops[ri][j]
            walk_leg(cur_pt, (self.stop_lat[s0], self.stop_lon[s0]), prev_name, self.stop_name[s0])
            wait = model.wait_s(ri, i, t, dt)
            t_dep = t + wait
            ride, ride85 = model.ride(ri, i, j, t_dep, dt)
            global v3_correct
            if v3_correct is not None and os.environ.get("V3_MODE", "xgb") != "off":
                try:
                    new = v3_correct(r["route_id"], self.stop_id[s0], self.stop_id[s1],
                                     base_date + timedelta(seconds=float(t_dep)), ride, None, j - i, None)
                    if new and new > 0:
                        ride85 *= new / max(ride, 1.0); ride = float(new)  # keep the p85 band proportional
                except Exception as e:
                    print("model v3 disabled:", e); v3_correct = None
            dist = float(r["dist"][i:j].sum())
            legs.append(dict(
                mode="bus", route=_display_name(r["short_name"]), route_short_name=r["short_name"],
                route_id=r["route_id"], agency=r["agency"], headsign=r["headsign"], route_desc=r["desc"],
                from_stop=self._stop_obj(s0), to_stop=self._stop_obj(s1), num_stops=j - i,
                wait_s=round(wait), ride_s=round(ride), ride_p85_s=round(ride85), distance_m=round(dist),
                speed_kmh=round(dist / max(ride, 1) * 3.6, 1),
                arrive_at_stop=ts(t), depart=ts(t_dep), arrive=ts(t_dep + ride),
                alt_routes=self.alt_routes(ri, i, j), polyline=self._polyline(ri, i, j)))
            t = t_dep + ride
            p85_extra += ride85 - ride
            wait_var += (0.7 * wait) ** 2
            cur_pt = (self.stop_lat[s1], self.stop_lon[s1])
            prev_name = self.stop_name[s1]
            nbus += 1
        walk_leg(cur_pt, d, prev_name, "Destination")
        bus_legs = [l for l in legs if l["mode"] == "bus"]
        return dict(
            depart=ts(t0), arrive=ts(t), duration_s=round(t - t0), duration_p85_s=round(t - t0 + p85_extra + wait_var ** 0.5),
            walk_m=round(total_walk), transfers=max(nbus - 1, 0),
            wait_s=sum(l["wait_s"] for l in bus_legs), ride_s=sum(l["ride_s"] for l in bus_legs),
            routes=[l["route"] for l in bus_legs],
            summary=" → ".join(["Walk"] + [l["route"] for l in bus_legs] + ["Walk"]) if bus_legs else "Walk",
            legs=legs)

    def _quiet_hour(self, op, o, d, dt, base_date):
        """Re-time the same journey (same stops and routes) at each of QUIET_HOURS and return the one with the
        least time on the bus, so a rider can see how much of the trip is traffic. Walk-only options are
        unaffected by traffic and get None."""
        if not op["routes"]:
            return None
        best = None
        for h in QUIET_HOURS:
            q = self.time_journey(op["_skel"], o, d, h * 3600.0, dt, base_date)
            if best is None or q["ride_s"] < best["ride_s"]:
                best = q
        return dict(depart=best["depart"], duration_s=best["duration_s"], ride_s=best["ride_s"],
                    wait_s=best["wait_s"], traffic_delay_s=op["ride_s"] - best["ride_s"],
                    note="same routes and stops, leaving at the quietest of %s on the same day"
                         % ", ".join("%02d:00" % h for h in QUIET_HOURS))

    def plan(self, from_lat, from_lng, to_lat, to_lng, depart=None, max_options=3, compare_quiet=True, quiet_planner=None):
        """quiet_planner: planner whose (historical) model times the quiet-hour reference; defaults to self.
        The realtime app passes the historical planner so live ratios never leak into the comparison."""
        tstart = time.time()
        if depart is None:
            depart = datetime.now(IST)
        elif depart.tzinfo is None:
            depart = depart.replace(tzinfo=IST)
        else:
            depart = depart.astimezone(IST)
        base_date = depart.replace(hour=0, minute=0, second=0, microsecond=0)
        t0 = (depart - base_date).total_seconds()
        dt = 1 if depart.weekday() >= 5 else 0
        o, d = (from_lat, from_lng), (to_lat, to_lng)
        access = self.nearby(*o)
        egress = self.nearby(*d)
        cands = []
        if access and egress:
            fast = self.raptor(access, egress, t0, dt)
            cands += fast
            cands += self.raptor(access, egress, t0, dt, walk_mult=3.0)
            # diversity: ban the routes of the fastest one-or-two-bus journeys
            # iteratively ban the routes (and same-number variants) of found journeys
            banned = set()
            for _ in range(3):
                found = [l[1] for j in cands for l in j if l[0] == "bus"]
                if not found:
                    break
                names = {_display_name(self.routes[ri]["short_name"]) for ri in found}
                for nm in names:
                    banned.update(self._by_name.get(nm, ()))
                more = self.raptor(access, egress, t0, dt, banned=frozenset(banned), max_rounds=3)
                if not more:
                    break
                cands += more
        options = []
        seen = set()
        for skel in cands:
            key = tuple((l[1], l[2], l[3]) for l in skel if l[0] == "bus")
            if key in seen or self._silly(skel):
                continue
            seen.add(key)
            op = self.time_journey(skel, o, d, t0, dt, base_date)
            op["_skel"] = skel
            options.append(op)
        crow = _dist_m(*o, *d)
        if crow * DETOUR <= DIRECT_WALK_MAX or not options:
            op = self.time_journey([], o, d, t0, dt, base_date)
            op["_skel"] = []
            options.append(op)
        # drop multi-bus journeys containing a trivially short hop (< 4 min ride)
        options = [op for op in options
                   if op["transfers"] == 0 or all(l["ride_s"] >= 240 for l in op["legs"] if l["mode"] == "bus")] or options
        # dedupe by displayed route sequence (keeps the faster)
        options.sort(key=lambda x: x["duration_s"])
        uniq, seen_routes = [], set()
        for op in options:
            k = tuple(op["routes"])
            if k in seen_routes:
                continue
            seen_routes.add(k)
            uniq.append(op)
        options = uniq
        walk_only = [op for op in options if not op["routes"]]
        if walk_only:  # a bus option slower than just walking is pointless
            wd = walk_only[0]["duration_s"]
            options = [op for op in options if not op["routes"] or op["duration_s"] < wd]
        selected = self._select(options, max_options)
        for op in selected:
            if compare_quiet:
                op["quiet_hour"] = (quiet_planner or self)._quiet_hour(op, o, d, dt, base_date)
        for op in options:
            op.pop("_skel", None)
        return dict(
            query=dict(from_lat=from_lat, from_lng=from_lng, to_lat=to_lat, to_lng=to_lng, depart=depart.isoformat(),
                       daytype="weekend" if dt else "weekday", straight_line_m=round(crow)),
            model=dict(source=self.model.source, stats=self.model.stats),
            options=selected, n_candidates=len(options), compute_ms=round((time.time() - tstart) * 1000))

    @staticmethod
    def _select(options, n):
        """Pick up to n options. Ranking uses a generalized cost with a transfer penalty, so a direct bus
        that is a few minutes slower beats a transfer. The best direct option is always kept if it is
        within DIRECT_KEEP of the fastest."""
        if not options:
            return []
        gen = lambda o: (o["duration_s"] + TRANSFER_PENALTY * o["transfers"] + 0.5 * o["walk_m"] / WALK_SPEED  # noqa
                         + WAIT_WEIGHT * o["wait_s"])
        fastest = min(options, key=lambda o: o["duration_s"])
        limit = fastest["duration_s"] * 1.6 + 1200
        pool = [o for o in options if o["duration_s"] <= limit]
        buses = [o for o in pool if o["routes"]]
        direct = [o for o in buses if o["transfers"] == 0]
        best_direct = min(direct, key=gen) if direct else None
        if best_direct is not None and best_direct["duration_s"] > fastest["duration_s"] * DIRECT_KEEP:
            best_direct = None
        for o in pool:
            o["labels"] = []
        picks = [(min(pool, key=gen), "recommended"), (best_direct, "direct"), (fastest, "fastest")]
        if buses:
            picks.append((min(buses, key=lambda o: (o["walk_m"], gen(o))), "least_walking"))
            picks.append((min(buses, key=lambda o: (o["transfers"], gen(o))), "fewest_transfers"))
        chosen = []
        for o, label in picks:
            if o is None:
                continue
            if any(c is o for c in chosen):
                o["labels"].append(label)
            elif len(chosen) < n:
                o["labels"].append(label)
                chosen.append(o)
        for o in sorted(pool, key=gen):
            if len(chosen) >= n:
                break
            if all(c is not o for c in chosen):
                o["labels"].append("alternative")
                chosen.append(o)
        chosen.sort(key=gen)
        return chosen
