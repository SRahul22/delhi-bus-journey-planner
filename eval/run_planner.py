"""Sanity-run the planner API on eval/test_pairs.csv and flag implausible results.

Start the planner first (e.g. `cd ../planner && PORT=8010 ./run.sh`), then:
  ../planner/.venv/bin/python run_planner.py --url http://localhost:8010
Writes eval/planner_results.csv.
"""
import argparse, json, os, time, urllib.parse, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GTFS = os.path.join(HERE, "..", "data", "gtfs")


def load_sched():
    trips = pd.read_csv(os.path.join(GTFS, "trips.txt"), dtype={"route_id": str, "trip_id": str}, usecols=["route_id", "trip_id"])
    st = pd.read_csv(os.path.join(GTFS, "stop_times.txt"), dtype={"trip_id": str, "stop_id": str},
                     usecols=["trip_id", "departure_time", "stop_id", "stop_sequence"])
    st["t"] = st.departure_time.map(lambda x: sum(int(a) * b for a, b in zip(x.split(":"), (3600, 60, 1))))
    return st.merge(trips, on="trip_id").drop(columns="departure_time")


def sched_leg(st_by_route, route_id, a, b, hour):
    st = st_by_route.get(route_id)
    if st is None:
        return np.nan
    st = st[st.stop_id.isin([a, b])]
    t = st.pivot_table(index="trip_id", columns="stop_id", values="t", aggfunc="first")
    q = st.pivot_table(index="trip_id", columns="stop_id", values="stop_sequence", aggfunc="first")
    if a not in t or b not in t:
        return np.nan
    ok = q[b] > q[a]
    d = (t[b] - t[a])[ok].dropna()
    if d.empty:
        return np.nan
    near = d[((t[a][ok].loc[d.index] // 3600) - hour).abs() <= 1]
    return float((near if len(near) else d).median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8010")
    ap.add_argument("--pairs", default=os.path.join(HERE, "test_pairs.csv"))
    args = ap.parse_args()
    P = pd.read_csv(args.pairs, dtype={"from_stop_id": str, "to_stop_id": str})
    st = load_sched()
    st_by_route = {k: v for k, v in st.groupby("route_id")}
    out = []
    for r in P.itertuples():
        q = dict(from_lat=r.from_lat, from_lng=r.from_lng, to_lat=r.to_lat, to_lng=r.to_lng, depart=r.depart, max_options=3)
        t0 = time.time()
        try:
            with urllib.request.urlopen(f"{args.url}/journey?{urllib.parse.urlencode(q)}", timeout=60) as f:
                res = json.load(f)
            err = ""
        except Exception as e:  # noqa
            res, err = {"options": []}, str(e)[:80]
        ms = (time.time() - t0) * 1000
        opts = res.get("options", [])
        row = dict(pair_id=r.pair_id, kind=r.kind, band=r.band, region=r.region, from_name=r.from_name, to_name=r.to_name,
                   dist_km=r.dist_km, depart=r.depart, tickets=r.tickets, ticket_routes=r.routes,
                   ticket_sched_ride_s=r.gtfs_sched_ride_s, n_options=len(opts), api_ms=round(ms), error=err)
        if opts:
            o = min(opts, key=lambda o: o["duration_s"])
            bus = [l for l in o["legs"] if l["mode"] == "bus"]
            sched = []
            for l in bus:
                h = int(l["depart"][11:13])
                sched.append(sched_leg(st_by_route, str(l["route_id"]), str(l["from_stop"]["stop_id"]), str(l["to_stop"]["stop_id"]), h))
            ticket_bases = {x.replace("UP", "").replace("DOWN", "") for x in str(r.routes).replace(">", "|").split("|")}
            row.update(duration_s=o["duration_s"], duration_p85_s=o.get("duration_p85_s"), ride_s=o["ride_s"], wait_s=o["wait_s"],
                       walk_m=o["walk_m"], transfers=o["transfers"], routes=">".join(l["route_short_name"] for l in bus),
                       labels="|".join(o.get("labels", [])),
                       legs_board_alight=" ; ".join(f'{l["from_stop"]["name"][:25]}->{l["to_stop"]["name"][:25]}' for l in bus),
                       sched_ride_same_legs_s=float(np.sum(sched)) if sched and not np.isnan(sched).any() else np.nan,
                       uses_ticket_route=any(l["route_short_name"].replace("UP", "").replace("DOWN", "") in ticket_bases for l in bus),
                       best_option_by_transfers=min(x["transfers"] for x in opts))
            # predicted ride on the ticket route itself (first 0-transfer option, incl. alt_routes, on a ticket route)
            for x in opts:
                xb = [l for l in x["legs"] if l["mode"] == "bus"]
                names = [xb[0]["route_short_name"]] + list(xb[0].get("alt_routes", [])) if len(xb) == 1 else []
                if any(n.replace("UP", "").replace("DOWN", "") in ticket_bases or n in ticket_bases for n in names):
                    row.update(ticket_route_ride_s=x["ride_s"], ticket_route_duration_s=x["duration_s"],
                               ticket_route_used=xb[0]["route_short_name"])
                    break
        out.append(row)
        print(r.pair_id, row.get("duration_s"), row.get("routes"), f"{ms:.0f}ms", err)
    R = pd.DataFrame(out)
    R["door_kmh"] = (R.dist_km / (R.duration_s / 3600)).round(1)
    R["ref_fast_s"] = (R.dist_km / 18 * 3600).round()   # 18 km/h straight-line door-to-door
    R["ref_slow_s"] = (R.dist_km / 10 * 3600).round()   # 10 km/h
    R["ride_vs_sched"] = (R.ride_s / R.sched_ride_same_legs_s).round(2)
    R["bus_ride_kmh"] = (R.dist_km / (R.ride_s / 3600)).round(1)

    def flags(x):
        f = []
        if not x.n_options:
            f.append("NO_RESULT")
            return "NO_RESULT"
        if x.duration_s > 3 * 3600: f.append("OVER_3H")
        if x.door_kmh > 25: f.append("TOO_FAST_>25kmh")
        elif x.door_kmh > 18: f.append("fast_18-25kmh")
        if x.door_kmh < 6: f.append("SLOW_<6kmh")
        elif x.door_kmh < 10: f.append("slow_6-10kmh")
        if x.ride_vs_sched < 0.6: f.append("RIDE<0.6xSCHED")
        if x.ride_vs_sched > 2.0: f.append("RIDE>2xSCHED")
        if x.walk_m > 2000: f.append("WALK>2km")
        if x.wait_s > 30 * 60: f.append("WAIT>30min")
        if x.kind == "direct" and x.best_option_by_transfers > 0: f.append("NO_DIRECT_OPTION")
        if not x.uses_ticket_route: f.append("not_ticket_route")
        return "|".join(f)
    R["flags"] = R.apply(flags, axis=1)
    R["implausible"] = R["flags"].str.contains(r"NO_RESULT|OVER_3H|TOO_FAST|SLOW_<6|RIDE<0.6|RIDE>2x|WALK>2km|WAIT>30|NO_DIRECT")
    R.to_csv(os.path.join(HERE, "planner_results.csv"), index=False)
    ok = R[R.n_options > 0]
    print(f"\nresults: {len(ok)}/{len(R)} with options; implausible: {R.implausible.sum()}")
    print("door-to-door km/h (straight line):", ok.door_kmh.describe().round(1).to_dict())
    print("ride / GTFS schedule on same legs:", ok.ride_vs_sched.describe().round(2).to_dict())
    print("by band:\n", ok.groupby("band")[["door_kmh", "ride_vs_sched", "duration_s"]].median())
    print("by slot hour:\n", ok.groupby(ok.depart.str[11:16])[["door_kmh", "ride_vs_sched"]].median())
    print(R[R["flags"] != ""][["pair_id", "kind", "dist_km", "depart", "duration_s", "ride_s", "wait_s", "walk_m", "transfers", "routes",
                            "ticket_routes", "door_kmh", "ride_vs_sched", "flags"]].to_string())


if __name__ == "__main__":
    main()
