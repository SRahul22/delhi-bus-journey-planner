"""v2 T2.1 validation on 19-24 Oct (no test days). Truth = random-arrival waits implied by the 19-24 stop-visit gaps
(same visit definition). Predictions from tables built on 01-18 only:
  v1:  min(median_headway/2, 1200) from tables_1_18/stop_headways.csv (map-matched passings, v1 method)
  v2:  min(mean(H^2)/(2 mean H), 1200) from stop_headways_v2_1_18.csv where n >= 20 (else v1 value, else skip)
Weights: total gap time in the cell (= share of random arrivals). Cells: (stop, route, hour, daytype) present in truth."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_headways_v2 import day_gaps
W = r"E:\hackathon-work\v2\data"; CAP = 1200
G = pd.concat([day_gaps(d) for d in ["19", "20", "21", "22", "23", "24"]], ignore_index=True)
G["daytype"] = np.where(G.wkend, "weekend", "weekday")
keys = ["stop_id", "route_id", "hour", "daytype"]
v1 = pd.read_csv(os.path.join(W, "tables_1_18", "stop_headways.csv"))
v1 = v1[v1.n >= 3].assign(w_v1=lambda d: np.minimum(d.median_headway_s / 2, CAP))[keys + ["w_v1"]]
v2 = pd.read_csv(os.path.join(W, "stop_headways_v2_1_18.csv"))
v2 = v2[v2.n >= 20].assign(w_v2=lambda d: np.minimum(d.expected_wait_s, CAP), p85_v2=lambda d: d.wait_p85_s)[keys + ["w_v2", "p85_v2"]]
G = G.merge(v1, on=keys, how="left").merge(v2, on=keys, how="left")
G["w_v2f"] = G.w_v2.fillna(G.w_v1)
G["p85_v1"] = G.w_v1 * 1.7
G["p85_v2f"] = G.p85_v2.fillna(G.p85_v1)
# per cell truth: mean wait = sum(H^2)/(2 sum H) ; P(W <= w) = 1 - sum(max(H-w,0))/sum(H)
def summarize(pred, p85, label, mask):
    g = G[mask & G[pred].notna()]
    H = g.H.to_numpy(); w = np.minimum(g[pred].to_numpy(), CAP)
    # random arrival in gap H: wait ~ U(0,H). Per-gap expected error terms weighted by H.
    tw = H / 2  # true mean wait inside this gap
    cell = g.assign(H2=H**2, err=(w - tw) * H, over=np.maximum(H - g[p85].to_numpy(), 0)).groupby(keys).agg(
        sH=("H", "sum"), sH2=("H2", "sum"), pred=(pred, "first"), over=("over", "sum"), n=("H", "size"))
    cell["truth"] = cell.sH2 / (2 * cell.sH); cell["pred"] = np.minimum(cell.pred, CAP)
    cell = cell[cell.n >= 5]
    wts = cell.sH
    bias = np.average(cell.pred - np.minimum(cell.truth, CAP), weights=wts)
    ae = (cell.pred - np.minimum(cell.truth, CAP)).abs()
    mae_w = np.average(ae, weights=wts)
    cov85 = 1 - cell.over.sum() / cell.sH.sum()
    print(f"{label:34s} cells={len(cell):7d} arrival-weighted: mean truth wait={np.average(np.minimum(cell.truth,CAP),weights=wts):6.1f}s "
          f"bias={bias:+6.1f}s MAE={mae_w:6.1f}s median|err|={ae.median():6.1f}s  P(wait<=p85)={cov85:.3f}")
both = G.w_v1.notna() & G.w_v2.notna()
print("== cells where both v1 and v2 (n>=20) have a value ==")
summarize("w_v1", "p85_v1", "v1 median/2 (passings)", both)
summarize("w_v2", "p85_v2", "v2 gaps (visits)", both)
print("== all cells with a v1 value; v2 falls back to v1 where n<20 ==")
summarize("w_v1", "p85_v1", "v1 median/2 (passings)", G.w_v1.notna())
summarize("w_v2f", "p85_v2f", "v2 gaps + v1 fallback", G.w_v1.notna())
print("share of truth gap-time covered: v1", round(G.H[G.w_v1.notna()].sum() / G.H.sum(), 3), " v2(n>=20)", round(G.H[G.w_v2.notna()].sum() / G.H.sum(), 3))
