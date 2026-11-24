"""T0.5 One-command v2 report.

  python eval/report.py --set val      # days 19-24 (use this for every check and gate, A4)
  python eval/report.py --set test     # days 25-31: only once at the end, for T5.1 (A4). Not run by agents.

Writes docs/results/v2_<date>_<set>.md with:
  1. ride error by ride-length bucket for GTFS schedule, sum of medians, sum of means, blend and XGBoost
     (val: blend/XGBoost only where available leak-free, see notes in the file), with n per bucket;
  2. band metrics (coverage, width, pinball@0.85, calibration) for the v1 ride band, and for the door-to-door band;
  3. door-to-door error for direct and transfer journeys from realized_<set>.csv (eval/realized_journeys.py; run it
     first), including wait, ride and transfer bias, and the share of journeys scored;
  4. the noise floor (eval/noise_floor.py).
Inputs: E:\\hackathon-work\\v3\\{fold4,test}.parquet, E:\\hackathon-work\\pass, E:\\hackathon-work\\v2\\eval\\realized_<set>.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402
import ride_baselines as RB  # noqa: E402

OUT = Path(os.environ.get("V2_EVAL_OUT", r"E:\hackathon-work\v2\eval"))
t0 = time.time()


def pct(x, nd=1):
    return M.fmt_pct(x, nd)


def d2d_section(which: str) -> list[str]:
    f = OUT / f"realized_{which}.csv"
    L = ["## 3. Door-to-door (realized journeys)", ""]
    if not f.exists():
        return L + [f"`{f}` not found: run `python eval/realized_journeys.py --set {which}` first.", ""]
    R = pd.read_csv(f)
    L.append(f"Source: `{f.name}` ({len(R):,} OD-times, written {datetime.fromtimestamp(f.stat().st_mtime):%Y-%m-%d %H:%M}). "
             "Plan = v1 planner recommended option; real = the same itinerary replayed on the day's real buses "
             "(stop visits, A1). Wait = first boarding wait; transfer = transfer walks + later waits.")
    L.append("")
    L.append("| Sample | OD-times | Scored | Share scored |")
    L.append("|---|---|---|---|")
    for smp, g in R.groupby("sample"):
        L.append(f"| {smp} | {len(g):,} | {int(g.scored.sum()):,} | {pct(g.scored.mean())} |")
    pr = R[R["sample"] == "pairs"]
    if len(pr):
        per = pr.groupby("od_id").scored.sum()
        L.append("")
        L.append(f"Ticket pairs scored on >= 3 days: {(per >= 3).sum()} of {per.size} "
                 f"(transfer pairs: {(pr[pr.kind == 'transfer'].groupby('od_id').scored.sum() >= 3).sum()} of "
                 f"{pr[pr.kind == 'transfer'].od_id.nunique()}). T0.2 target: >= 90% of the 60 pairs.")
    reasons = R.loc[~R.scored, "unscored_reason"].fillna("").str.replace(r"^leg\d:", "", regex=True).value_counts()
    if len(reasons):
        L.append("Unscored reasons: " + ", ".join(f"{k} {v}" for k, v in reasons.items()) + ".")
    S = R[R.scored].copy()
    S["journey"] = np.where(S.plan_n_bus >= 2, "transfer", "direct")
    L += ["", "Error of the planned door-to-door time vs realized (median APE, within ±20%, median bias):", "",
          "| Sample | Journey | n | Median error | Within ±20% | Median bias total | Median bias wait | Median bias ride | Median bias transfer | Mean real wait |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for smp in ["pairs", "random", "all"]:
        for j in ["direct", "transfer", "all"]:
            g = S if smp == "all" else S[S["sample"] == smp]
            g = g if j == "all" else g[g.journey == j]
            if not len(g):
                continue
            e = M.error(g.plan_total_s, g.real_total_s)
            L.append(f"| {smp} | {j} | {len(g):,} | {pct(e['median_ape'])} | {pct(e['within20'])} | "
                     f"{np.median(g.plan_total_s - g.real_total_s):+.0f} s | {np.median(g.plan_wait_s - g.real_wait_s):+.0f} s | "
                     f"{np.median(g.plan_ride_s - g.real_ride_s):+.0f} s | "
                     f"{np.median(g.plan_transfer_s - g.real_transfer_s) if j != 'direct' else 0:+.0f} s | {g.real_wait_s.mean():.0f} s |")
    L += ["", f"Mean wait bias (plan - real, all scored): {np.mean(S.plan_wait_s - S.real_wait_s):+.0f} s "
              f"(G2 target: within ±60 s).", ""]
    # door-to-door band (v1: duration_p85_s); p50 = plan_total_s
    L += ["Door-to-door band (v1 `duration_p85_s`; p90 from the log-normal of the band):", "",
          "| Journey | n | <p50 | <p85 | <p90 | Width | Pinball@0.85 (s) |", "|---|---|---|---|---|---|---|"]
    for j in ["direct", "transfer", "all"]:
        g = S if j == "all" else S[S.journey == j]
        if not len(g):
            continue
        c = M.calibration(g.plan_total_s, g.plan_p85_s, g.real_total_s)
        L.append(f"| {j} | {len(g):,} | {pct(c['under_p50'])} | {pct(c['under_p85'])} | {pct(c['under_p90'])} | "
                 f"{pct(M.width(g.plan_total_s, g.plan_p85_s))} | {M.pinball(g.plan_p85_s, g.real_total_s):.0f} |")
    L += ["", "G2 targets (for v2 bands, judged by humans): 47-53% / 82-88% / 87-93%.", ""]
    # by bucket x peak, all scored
    tab = M.by_bucket_peak(S.real_total_s, S.peak.astype(bool), {"plan": S.plan_total_s}, {"plan": S.plan_p85_s})
    tab = tab[tab.n > 0]
    L += ["Door-to-door by length bucket × peak (realized total):", "",
          "| Bucket | Peak | n | Median error | Within ±20% | Median bias | Coverage p85 |", "|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        L.append(f"| {r.bucket} | {r.peak} | {r.n:,} | {pct(r.median_ape)} | {pct(r.within20)} | {r.median_bias_s:+.0f} s | {pct(r.coverage)} |")
    L.append("")
    return L


def noise_section(which: str) -> list[str]:
    f = OUT / f"noise_floor_{which}.json"
    env = dict(os.environ)
    subprocess.run([sys.executable, str(HERE / "noise_floor.py"), "--set", which], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    nf = json.load(open(f))
    L = ["## 4. Noise floor", "", nf["definition"] + f"; {nf['n_keys_sampled']:,} sampled ride keys, "
         f"{nf['n_cells']:,} cells, {nf['n_traversals']:,} traversals, days {nf['days'][0]}-{nf['days'][-1]}.", "",
         f"Median error vs own cell median: **{pct(nf['median_ape'], 2)}** (within ±20%: {pct(nf['within20'])}).", "",
         "| Bucket | n | Median error | Within ±20% |", "|---|---|---|---|"]
    for b, v in nf["by_bucket"].items():
        L.append(f"| {b} | {v['n']:,} | {pct(v['median_ape'], 2)} | {pct(v['within20'])} |")
    L += ["", "A model that only knows route, stops and hour cannot beat this on average; the gap between a model's "
              "ride error and this number is what is left to win.", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["val", "test"], required=True)
    ap.add_argument("--out", default=str(REPO / "docs" / "results"))
    a = ap.parse_args()
    which = a.set
    L = [f"# v2 evaluation report: {which} set ({'19-24' if which == 'val' else '25-31'} Oct)", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python eval/report.py --set {which}`"
         f" (git {subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}). "
         "All numbers measure v1 behaviour unless a v2 flag is named.", ""]
    R, tab, nf_in, info = RB.compute(which)
    tab.to_csv(OUT / f"rides_{which}_metrics.csv", index=False)
    L += ["## 1. Ride error (in-vehicle, stop to stop)", ""]
    if which == "val":
        L += ["Rides: fold-4 rides of days 19-24 with lookup features from days 1-18 (`E:\\hackathon-work\\v3\\fold4.parquet`). "
              "Blend (600 s) on val comes from `eval/val_blend.py` (1-18 tables rebuilt; reproduces fold4 base_med exactly). "
              "XGBoost is not shown on val: the final model was refit on days 1-24 and has seen these rides; the ml "
              "agent's 1-18 model (T0.4/T1.1) should be scored here instead.", ""]
    L.append(RB.summary_md(which, tab, nf_in, info).replace("### ", "### ").split("\nIn-sample noise floor")[0])
    L.append("")
    L += d2d_section(which)
    L += noise_section(which)
    L.append(f"_Report runtime {time.time() - t0:.0f} s._")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"v2_{datetime.now():%Y-%m-%d}_{which}.md"
    f.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("wrote", f)


if __name__ == "__main__":
    main()
