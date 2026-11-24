"""Compare door-to-door realized_val.csv files from several planner configs (integration rerun A/B/C).

  python eval/rerun_compare.py NAME=path/realized_val.csv [NAME=...]  > table.md
Per config on its own scored journeys, and on the common subset (OD-times scored in every config). Metrics from
eval/metrics.py. Ride-only leg error: plan leg ride_s vs realized leg ride (real_legs), split by realized leg length.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402

KEY = ["od_id", "day", "depart_s"]


def load(p):
    R = pd.read_csv(p, dtype={"plan_route_ids": str, "plan_stops": str, "kind": str})
    R["key"] = R.od_id.astype(str) + "|" + R.day.astype(str) + "|" + R.depart_s.astype(str)
    R["scored"] = R.scored.astype(bool)
    return R


def summary(S):
    a = S.real_total_s.to_numpy()
    e = M.error(S.plan_total_s, a); b = M.bias(S.plan_total_s, a)
    cal = M.calibration(S.plan_total_s, S.plan_p85_s, a)
    T = S[S.plan_n_bus >= 2]; Dd = S[S.plan_n_bus == 1]
    return dict(n=len(S), median_ape=e["median_ape"], within20=e["within20"], bias_total=b["median_bias_s"],
                bias_wait=M.bias(S.plan_wait_s, S.real_wait_s)["median_bias_s"],
                bias_ride=M.bias(S.plan_ride_s, S.real_ride_s)["median_bias_s"],
                bias_transfer=M.bias(T.plan_transfer_s, T.real_transfer_s)["median_bias_s"],
                n_direct=len(Dd), ape_direct=M.error(Dd.plan_total_s, Dd.real_total_s)["median_ape"],
                n_transfer=len(T), ape_transfer=M.error(T.plan_total_s, T.real_total_s)["median_ape"],
                under_p50=cal["under_p50"], under_p85=cal["under_p85"])


def legs(S):
    rows = []
    for r in S.itertuples():
        pl = [l for l in json.loads(r.plan_legs_json) if l["mode"] == "bus"]
        rl = str(r.real_legs).split(";")
        if len(pl) != len(rl):
            continue
        if any("ride" not in l for l in pl):  # csv rebuilt with --plans-from-csv has no leg rides: direct journeys only
            if len(pl) != 1:
                continue
            pl = [dict(pl[0], ride=r.plan_ride_s)]
        for k, (l, x) in enumerate(zip(pl, rl)):
            rid, wr = x.split(":"); ride = float(wr.split("+")[1])
            if ride <= 0:
                continue
            rows.append(dict(key=r.key, k=k, leg=f"{l['route_id']}:{l['board']}>{l['alight']}", plan=l["ride"], real=ride))
    return pd.DataFrame(rows)


def leg_table(L):
    out = {}
    for nm, m in [("all", np.ones(len(L), bool)), ("<=60", L.real <= 3600), (">60", L.real > 3600)]:
        e = M.error(L.plan[m], L.real[m]); b = M.bias(L.plan[m], L.real[m])
        out[nm] = dict(n=e["n"], ape=e["median_ape"], w20=e["within20"], bias=b["median_bias_s"], bias_pct=b["median_bias_pct"])
    return out


def fmt_row(name, s):
    p = M.fmt_pct
    return (f"| {name} | {s['n']} | {p(s['median_ape'])} | {p(s['within20'])} | {s['bias_total']:+.0f} | {s['bias_wait']:+.0f} | "
            f"{s['bias_ride']:+.0f} | {s['bias_transfer']:+.0f} | {p(s['ape_direct'])} ({s['n_direct']}) | "
            f"{p(s['ape_transfer'])} ({s['n_transfer']}) | {p(s['under_p50'])} | {p(s['under_p85'])} |")


HDR = ("| config | n | d2d median APE | within ±20% | median bias total (s) | first wait (s) | ride (s) | transfer (s) | "
       "direct APE (n) | transfer APE (n) | under p50 | under p85 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    cfg = dict(a.split("=", 1) for a in sys.argv[1:])
    D = {k: load(v) for k, v in cfg.items()}
    print("Scored / total:", ", ".join(f"{k} {int(v.scored.sum())}/{len(v)}" for k, v in D.items()), "\n")
    print("### Each config on its own scored journeys\n"); print(HDR)
    for k, v in D.items():
        print(fmt_row(k, summary(v[v.scored])))
    common = set.intersection(*[set(v.key[v.scored]) for v in D.values()])
    print(f"\n### Common subset (scored in all configs, n={len(common)})\n"); print(HDR)
    for k, v in D.items():
        print(fmt_row(k, summary(v[v.key.isin(common)])))
    print("\n### Plans that changed vs the first config (all 5,240 OD-times)\n")
    names = list(D); base = D[names[0]].set_index("key")
    for k in names[1:]:
        o = D[k].set_index("key").reindex(base.index)
        same = (o.plan_stops.fillna("") == base.plan_stops.fillna("")) & (o.plan_route_ids.fillna("") == base.plan_route_ids.fillna(""))
        dur = (o.plan_total_s - base.plan_total_s).abs() > 0.5
        print(f"- {k}: itinerary changed {int((~same).sum())}, same itinerary but different duration {int((same & dur).sum())}")
    print("\n### Ride-only leg error (plan leg ride_s vs realized leg ride), by realized leg length\n")
    print("| config | subset | bucket | legs | median APE | within ±20% | median bias (s) | median bias % |\n|---|---|---|---|---|---|---|---|")
    Ls = {k: legs(v[v.scored]) for k, v in D.items()}
    ck = set.intersection(*[set(L.key + "#" + L.k.astype(str) + "#" + L.leg) for L in Ls.values()])
    for sub in ("own scored", "identical legs, common"):
        for k, L in Ls.items():
            if sub != "own scored":
                L = L[(L.key + "#" + L.k.astype(str) + "#" + L.leg).isin(ck)]
            for b, s in leg_table(L).items():
                print(f"| {k} | {sub} | {b} | {s['n']} | {M.fmt_pct(s['ape'])} | {M.fmt_pct(s['w20'])} | {s['bias']:+.0f} | {M.fmt_pct(s['bias_pct'])} |")


if __name__ == "__main__":
    main()
