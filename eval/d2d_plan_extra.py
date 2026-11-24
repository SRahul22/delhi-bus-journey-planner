"""Extra config-C plans for the d2d floor/bias/live analysis (eval/d2d_*.py).

  python eval/d2d_plan_extra.py calib   -> 2,000 random OD-times on days 13-18 (calibration fit, days 1-18 only)
  python eval/d2d_plan_extra.py live22  -> random OD-times on 22 Oct, departures 06:00-21:00 (live-mode test)
Config C: V3_MODE=xgb, V3_MIN_RIDE_S=3600, V2_WAIT=pooled with the days 1-18 index and headways (no val leakage).
Plans are made in-process with eval/realized_journeys.py and replayed with its realize() (strict route_id, same
rules as realized_val.csv). Output: E:\\hackathon-work\\v2\\analysis\\plans_<set>.pkl and realized_<set>.csv.
"""
import os
import sys

os.environ.update(V3_MODE="xgb", V3_MIN_RIDE_S="3600", V2_WAIT="pooled", REALTIME="0",
                  POOLED_INDEX=r"E:\hackathon-work\v2\planner\pooled_index_visits_1-18.npz",
                  V2_HEADWAYS=r"E:\hackathon-work\v2\data\stop_headways_v2_1_18.csv",
                  MODEL_DIR=r"E:\hackathon-project\model")
import json  # noqa: E402
import pickle  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realized_journeys as RJ  # noqa: E402

OUT = Path(r"E:\hackathon-work\v2\analysis")
SETS = {"calib": dict(days=[13, 14, 15, 16, 17, 18], n=2000, seed=1318, max_dep=22 * 3600),
        "live22": dict(days=[22], n=2200, seed=2222, max_dep=21 * 3600)}


def main():
    name = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cfg = SETS[name]
    assert all(d <= 24 for d in cfg["days"])  # never test days 25-31
    Q = [q for q in RJ.random_ods(cfg["days"], cfg["n"], seed=cfg["seed"]) if q["depart_s"] < cfg["max_dep"]]
    RJ.log(name, len(Q), "OD-times")
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(workers, initializer=RJ._init) as pool:
        plans = []
        for i, p in enumerate(pool.imap(RJ._plan, Q, chunksize=4)):
            plans.append(p)
            if (i + 1) % 250 == 0:
                RJ.log("planned", i + 1)
    pickle.dump(plans, open(OUT / f"plans_{name}.pkl", "wb"))
    fams = RJ.route_families()
    rows = []
    for d in cfg["days"]:
        V = RJ.DayVisits(d, fams)
        ep = RJ.day_epoch(d)
        for p in plans:
            if p["day"] != d:
                continue
            row = {k: v for k, v in p.items() if k != "plan_legs"}
            row.update(RJ.plan_columns(p))
            row["plan_legs_json"] = json.dumps(p.get("plan_legs", []))
            row.update(RJ.realize(p, V, ep, family=False))
            rows.append(row)
        RJ.log("realized day", d)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / f"realized_{name}.csv", index=False)
    RJ.log("wrote", len(R), "scored", int(R.scored.sum()), "DONE")


if __name__ == "__main__":
    main()

