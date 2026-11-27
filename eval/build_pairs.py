"""Build the validation test set of origin-destination pairs from ticketing data.

Input:
  od_route.csv   (start,end,route,tickets,hours[24]) aggregated on Windows by agg_tickets.py
  stop_names.csv (name,count) all ticket stop names with ticket counts
  ../data/gtfs   GTFS feed
Output:
  test_pairs.csv, google_pairs_todo.csv, match_report.txt (in eval/)

Run: ../planner/.venv/bin/python build_pairs.py --work <dir with od_route.csv>
"""
import argparse, difflib, math, os, re, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GTFS = os.environ.get("GTFS_DIR") or next((p for p in (os.path.join(HERE, "..", "data", "gtfs"), os.path.join(os.environ.get("DATA_DIR", r"E:\hackathon-data"), "gtfs")) if os.path.exists(os.path.join(p, "stops.txt"))), os.path.join(HERE, "..", "data", "gtfs"))
CP = (28.6315, 77.2167)  # Connaught Place
DATE = "2026-10-28"      # Wednesday in the held-out week (25-31 Oct)
SLOTS = {"am_peak": ("09:00", range(7, 11)), "offpeak": ("13:00", range(11, 16)), "pm_peak": ("18:30", range(17, 21))}

ABBR = [(r"\bsation\b", "station"), (r"\bstn\b", "station"), (r"\bmetro station\b", "metro"), (r"\bms\b", "metro"),
        (r"\bterminal\b", "terminal"), (r"\b(t)\b", "terminal"), (r"\bsec\b", "sector"), (r"\bsect\b", "sector"),
        (r"\bxing\b", "crossing"), (r"\bx ing\b", "crossing"), (r"\brly\b", "railway"), (r"\bextn\b", "extension"),
        (r"\bext\b", "extension"), (r"\bmor\b", "more"), (r"\bhosp\b", "hospital"), (r"\bdepot\b", "depot"),
        (r"\bvill\b", "village"), (r"\bgaon\b", "village"), (r"\bnr\b", "near"), (r"\bph\b", "phase")]


def norm(s):
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    for a, b in ABBR:
        s = re.sub(a, b, s)
    return re.sub(r"\s+", " ", s).strip()


def hav(a1, o1, a2, o2):
    r = 6371000.0
    p1, p2 = math.radians(a1), math.radians(a2)
    dp, dl = p2 - p1, math.radians(o2 - o1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def region(lat, lng):
    if hav(lat, lng, *CP) < 5000:
        return "Central"
    ang = math.degrees(math.atan2(lat - CP[0], (lng - CP[1]) * math.cos(math.radians(CP[0]))))
    return ["East", "North", "West", "South"][int(((ang + 45) % 360) // 90)]


class Gtfs:
    def __init__(self):
        self.stops = pd.read_csv(os.path.join(GTFS, "stops.txt"), dtype={"stop_id": str})
        self.stops["n"] = self.stops.stop_name.map(norm)
        routes = pd.read_csv(os.path.join(GTFS, "routes.txt"), dtype={"route_id": str})
        trips = pd.read_csv(os.path.join(GTFS, "trips.txt"), dtype={"route_id": str, "trip_id": str})
        st = pd.read_csv(os.path.join(GTFS, "stop_times.txt"), dtype={"trip_id": str, "stop_id": str},
                         usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"])
        st["t"] = st.departure_time.map(lambda x: sum(int(a) * b for a, b in zip(x.split(":"), (3600, 60, 1))))
        st = st.merge(trips[["trip_id", "route_id"]], on="trip_id")
        self.st = st
        self.short2rid = routes.groupby(routes.route_short_name.str.upper()).route_id.apply(list).to_dict()
        # stop set per route short name
        rs = st[["route_id", "stop_id"]].drop_duplicates().merge(routes[["route_id", "route_short_name"]])
        self.route_stops = rs.groupby(rs.route_short_name.str.upper()).stop_id.apply(set).to_dict()
        self.by_id = self.stops.set_index("stop_id")
        self.by_norm = self.stops.groupby("n").stop_id.apply(list).to_dict()
        self.by_exact = self.stops.groupby("stop_name").stop_id.apply(list).to_dict()
        self.norm_names = list(self.by_norm)
        # GTFS names are often compound ("Laxmi Nagar X-ing / Shakarpur Crossing"): index each part too
        self.stops["parts"] = self.stops.stop_name.map(lambda x: [norm(p) for p in str(x).split("/") if norm(p)])
        self.part_idx = collections.defaultdict(list)
        for sid, ps in zip(self.stops.stop_id, self.stops.parts):
            for p in ps:
                self.part_idx[p].append(sid); self.part_idx[p.replace(" ", "")].append(sid)
        self.part_names = [p for p in self.part_idx if " " in p or len(p) > 3]
        # normalised stop-name set per route (for 'is there a direct route' checks)
        self.route_names = {r: {self.by_id.n[s] for s in ids} for r, ids in self.route_stops.items()}

    def match(self, name, route=None):
        """Return (stop_id, method). Prefer stops on the ticket's route."""
        n = norm(name)
        cand_route = self.route_stops.get(str(route).upper(), set()) if route else set()
        if cand_route:
            sub = self.by_id.loc[list(cand_route)]
            hit = sub[sub.stop_name == name]
            if len(hit):
                return hit.index[0], "route_exact"
            hit = sub[sub.n == n]
            if len(hit):
                return hit.index[0], "route_norm"
            c = n.replace(" ", "")
            for sid, ps in zip(sub.index, self.stops.set_index("stop_id").parts.loc[sub.index]):
                if any(p == n or p.replace(" ", "") == c for p in ps):
                    return sid, "route_part"
            best = difflib.get_close_matches(n, list(sub.n), n=1, cutoff=0.8)
            if best:
                return sub[sub.n == best[0]].index[0], "route_fuzzy"
            toks = set(n.split())
            for sid, ps in zip(sub.index, self.stops.set_index("stop_id").parts.loc[sub.index]):
                if any(toks <= set(p.split()) or (len(c) > 5 and c in p.replace(" ", "")) for p in ps):
                    return sid, "route_partial"
        if name in self.by_exact:
            return self.by_exact[name][0], "exact"
        if n in self.by_norm:
            return self.by_norm[n][0], "norm"
        if n in self.part_idx or n.replace(" ", "") in self.part_idx:
            return (self.part_idx.get(n) or self.part_idx[n.replace(" ", "")])[0], "part"
        best = difflib.get_close_matches(n, self.part_names, n=1, cutoff=0.85)
        if best:
            return self.part_idx[best[0]][0], "fuzzy"
        return None, "none"

    def sched_ride(self, route, a, b, hour):
        """Median scheduled ride (s) from stop a to b on route short name, trips departing a near hour."""
        rids = self.short2rid.get(str(route).upper(), [])
        st = self.st[self.st.route_id.isin(rids) & self.st.stop_id.isin([a, b])]
        if st.empty:
            return np.nan
        p = st.pivot_table(index="trip_id", columns="stop_id", values="t", aggfunc="first")
        p2 = st.pivot_table(index="trip_id", columns="stop_id", values="stop_sequence", aggfunc="first")
        if a not in p or b not in p:
            return np.nan
        ok = p2[b] > p2[a]
        d = (p[b] - p[a])[ok].dropna()
        if d.empty:
            return np.nan
        dep = p[a][ok].loc[d.index]
        near = d[(dep // 3600 - hour).abs() <= 1]
        return float((near if len(near) else d).median())


def pick_slot(hours):
    tot = {k: sum(hours[h] for h in rng) for k, (_, rng) in SLOTS.items()}
    return max(tot, key=tot.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    g = Gtfs()
    od = pd.read_csv(os.path.join(args.work, "od_route.csv"), keep_default_na=False)
    od = od[(od.start != od.end) & (od.start != "") & (od.end != "")]
    names = pd.read_csv(os.path.join(args.work, "stop_names.csv"), keep_default_na=False)

    # ---- match-rate report (global name match, without route context)
    rep = []
    meth = {nm: g.match(nm)[1] for nm in names.name}
    names["method"] = names.name.map(meth)
    tot_n, tot_w = len(names), names["count"].sum()
    rep.append(f"ticket stop names: {tot_n} unique, {tot_w} stop mentions (boarding + alighting)")
    for m in ["exact", "norm", "part", "fuzzy", "none"]:
        s = names[names.method == m]
        rep.append(f"  {m:6s}: {len(s):5d} names ({len(s)/tot_n:6.1%})  {s['count'].sum()/tot_w:6.1%} of mentions")
    rep.append("top unmatched names: " + "; ".join(names[names.method == "none"].head(15).name))

    # ---- route-aware matching for the top OD x route rows
    top = od.head(6000).copy()
    ms = [g.match(s, r) for s, r in zip(top.start, top.route)]
    me = [g.match(e, r) for e, r in zip(top.end, top.route)]
    top["from_stop_id"], top["from_method"] = zip(*ms)
    top["to_stop_id"], top["to_method"] = zip(*me)
    okm = top.from_stop_id.notna() & top.to_stop_id.notna()
    rep.append(f"top 6000 OD x route rows: both ends matched {okm.mean():.1%} "
               f"({top.tickets[okm].sum()/top.tickets.sum():.1%} of their tickets); "
               f"route-aware methods: {collections.Counter(list(top.from_method)+list(top.to_method)).most_common()}")
    top = top[okm]
    top["hours"] = top.hours.map(lambda s: [int(x) for x in s.split()])

    # aggregate by OD across routes (use the most-ticketed route's matched stops for coordinates)
    rows = []
    for (s, e), grp in top.groupby(["start", "end"], sort=False):
        grp = grp.sort_values("tickets", ascending=False)
        h = np.sum(np.array(list(grp.hours)), axis=0)
        b = grp.iloc[0]
        fa, fo = g.by_id.stop_lat[b.from_stop_id], g.by_id.stop_lon[b.from_stop_id]
        ta, to = g.by_id.stop_lat[b.to_stop_id], g.by_id.stop_lon[b.to_stop_id]
        rows.append(dict(from_name=s, to_name=e, from_stop_id=b.from_stop_id, to_stop_id=b.to_stop_id,
                         from_lat=fa, from_lng=fo, to_lat=ta, to_lng=to, tickets=int(grp.tickets.sum()),
                         routes="|".join(grp.route[:4]), main_route=b.route, hours=h,
                         match=f"{b.from_method}/{b.to_method}"))
    P = pd.DataFrame(rows)
    P["dist_km"] = [hav(*x) / 1000 for x in zip(P.from_lat, P.from_lng, P.to_lat, P.to_lng)]
    P = P[P.dist_km > 1.0]
    P["band"] = pd.cut(P.dist_km, [0, 5, 12, 1e9], labels=["short", "medium", "long"]).astype(str)
    P["region"] = [region((a + c) / 2, (b + d) / 2) for a, b, c, d in zip(P.from_lat, P.from_lng, P.to_lat, P.to_lng)]
    P = P.sort_values("tickets", ascending=False)

    # ---- stratified direct selection: round-robin over regions within each band
    quota = {"short": 18, "medium": 18, "long": 14}
    used, seen_pairs, chosen = collections.Counter(), set(), []
    # stops are deduplicated by location (~600 m cells), so "West Patel Nagar" and
    # "West Patel Nagar / Patel Nagar" or a reversed pair count as the same pair
    ck = lambda lat, lng: (round(lat / 0.006), round(lng / 0.006))
    for band, q in quota.items():
        pool = P[P.band == band]
        regs = list(pool.region.value_counts().index)
        ptr = {r: iter(pool[pool.region == r].itertuples()) for r in regs}
        got = 0
        while got < q and ptr:
            for r in list(ptr):
                if got >= q:
                    break
                for row in ptr[r]:
                    a, b = ck(row.from_lat, row.from_lng), ck(row.to_lat, row.to_lng)
                    key = frozenset((a, b))
                    if key in seen_pairs or used[a] >= 2 or used[b] >= 2:
                        continue
                    seen_pairs.add(key); used[a] += 1; used[b] += 1
                    chosen.append(row._asdict()); got += 1
                    break
                else:
                    del ptr[r]
    D = pd.DataFrame(chosen)
    D["kind"] = "direct"

    # ---- transfer pairs: A->B on r1 and B->C on r2 (both popular), no GTFS route serves both A and C
    base = lambda r: (re.findall(r"\d+", str(r)) or [str(r)])[0]  # 717AUP/717BDOWN -> 717
    T = top.copy()
    T["from_n"], T["to_n"] = T.start.map(norm), T.end.map(norm)
    legs1 = T[T.tickets >= 40]
    by_start = legs1.groupby("from_n")
    cands = []
    for r1 in legs1.itertuples():
        if r1.to_n not in by_start.groups:
            continue
        for r2 in by_start.get_group(r1.to_n).itertuples():
            if base(r2.route) == base(r1.route) or r2.to_n == r1.from_n:
                continue
            a, c = g.by_id.loc[r1.from_stop_id], g.by_id.loc[r2.to_stop_id]
            dac = hav(a.stop_lat, a.stop_lon, c.stop_lat, c.stop_lon) / 1000
            dab = hav(a.stop_lat, a.stop_lon, *g.by_id.loc[r1.to_stop_id, ["stop_lat", "stop_lon"]]) / 1000
            if dac < 4 or dac < 0.8 * dab:  # transfer must make progress
                continue
            if any(r1.from_n in ns and r2.to_n in ns for ns in g.route_names.values()):
                continue
            cands.append(dict(from_name=r1.start, to_name=r2.end, via=r1.end, r1=r1.route, r2=r2.route,
                              from_stop_id=r1.from_stop_id, to_stop_id=r2.to_stop_id,
                              via_stop_id=r1.to_stop_id, via2_stop_id=r2.from_stop_id,
                              from_lat=a.stop_lat, from_lng=a.stop_lon, to_lat=c.stop_lat, to_lng=c.stop_lon,
                              tickets=int(min(r1.tickets, r2.tickets)), t1=r1.tickets, t2=r2.tickets,
                              dist_km=dac, hours=np.array(r1.hours) + np.array(r2.hours)))
    C = pd.DataFrame(cands).sort_values("tickets", ascending=False)
    tsel, used_via = [], collections.Counter()
    for row in C.itertuples():
        a, b = ck(row.from_lat, row.from_lng), ck(row.to_lat, row.to_lng)
        v = ck(*g.by_id.loc[row.via_stop_id, ["stop_lat", "stop_lon"]])
        key = frozenset((a, b))
        if key in seen_pairs or used_via[v] >= 2 or used[a] >= 2 or used[b] >= 2:
            continue
        seen_pairs.add(key); used_via[v] += 1; used[a] += 1; used[b] += 1
        tsel.append(row._asdict())
        if len(tsel) >= 10:
            break
    X = pd.DataFrame(tsel)
    X["kind"] = "transfer"
    X["routes"] = X.r1 + ">" + X.r2
    X["main_route"] = X.r1
    X["match"] = "route"
    X["band"] = pd.cut(X.dist_km, [0, 5, 12, 1e9], labels=["short", "medium", "long"]).astype(str)
    X["region"] = [region((a + c) / 2, (b + d) / 2) for a, b, c, d in zip(X.from_lat, X.from_lng, X.to_lat, X.to_lng)]

    A = pd.concat([D, X], ignore_index=True)
    # departure slot: mix of ~25 am peak / ~15 off-peak / ~20 pm peak, each pair gets the slot where
    # its own tickets are relatively strongest (and always a slot where that pair has real demand)
    sh = pd.DataFrame({k: A.hours.map(lambda h: sum(h[x] for x in rng) / max(1, sum(h))) for k, (_, rng) in SLOTS.items()})
    A["slot"] = "am_peak"
    for k, q in [("offpeak", 15), ("pm_peak", 20)]:
        free = A.slot == "am_peak"
        idx = sh[free & (sh[k] >= 0.15)][k].nlargest(q).index
        A.loc[idx, "slot"] = k
    A["depart"] = [f"{DATE}T{SLOTS[s][0]}" for s in A.slot]
    # scheduled GTFS ride time on the ticket route(s) (pure in-vehicle, no walk/wait)
    sched = []
    for r in A.itertuples():
        h = int(r.depart[11:13])
        if r.kind == "direct":
            sched.append(g.sched_ride(r.main_route, r.from_stop_id, r.to_stop_id, h))
        else:
            s1 = g.sched_ride(r.r1, r.from_stop_id, r.via_stop_id, h)
            s2 = g.sched_ride(r.r2, r.via2_stop_id, r.to_stop_id, h + 1)
            sched.append(s1 + s2)
    A["gtfs_sched_ride_s"] = np.round(sched)
    A = A.sort_values(["kind", "band", "tickets"], ascending=[True, False, False]).reset_index(drop=True)
    A.insert(0, "pair_id", [f"P{i+1:02d}" for i in range(len(A))])
    A["dist_km"] = A.dist_km.round(2)
    A["peak_hour"] = A.hours.map(lambda h: int(np.argmax(h)))
    A["via"] = A.get("via", pd.Series(dtype=str)).fillna("")
    cols = ["pair_id", "kind", "band", "region", "from_name", "from_stop_id", "from_lat", "from_lng",
            "to_name", "to_stop_id", "to_lat", "to_lng", "dist_km", "depart", "slot", "peak_hour", "tickets",
            "routes", "via", "gtfs_sched_ride_s", "match"]
    A[cols].to_csv(os.path.join(HERE, "test_pairs.csv"), index=False, float_format="%.6f")

    # ---- Google todo: 30 most important (top tickets per band/kind, keep the mix)
    imp = pd.concat([A[A.kind == "transfer"].head(6)] +
                    [A[(A.kind == "direct") & (A.band == b)].nlargest(8, "tickets") for b in ["short", "medium", "long"]])
    imp = imp.sort_values("pair_id")
    G = imp[["pair_id", "from_name", "to_name", "from_lat", "from_lng", "to_lat", "to_lng", "depart"]].copy()
    G["google_url"] = [f"https://www.google.com/maps/dir/?api=1&origin={a:.6f},{b:.6f}&destination={c:.6f},{d:.6f}&travelmode=transit"
                       for a, b, c, d in zip(G.from_lat, G.from_lng, G.to_lat, G.to_lng)]
    G["google_s"] = ""
    G["google_notes"] = ""
    G.to_csv(os.path.join(HERE, "google_pairs_todo.csv"), index=False, float_format="%.6f")

    rep.append(f"selected: {len(A)} pairs; " + str(A.groupby(['kind', 'band']).size().to_dict()))
    rep.append("regions: " + str(A.region.value_counts().to_dict()) + "  slots: " + str(A.slot.value_counts().to_dict()))
    open(os.path.join(HERE, "match_report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
