"""All-options benchmark: every option we return (C = main, ALT = ALT_V2=1) vs every bus-only option Google lists.
Inputs: allopts_{C,ALT}.csv (bench_allopts.py), google_bus_options.csv (5 per pair), comparison.csv + test_pairs.csv.
Like for like: ours = total - first wait (Google's time runs from its first departure)."""
import pandas as pd, numpy as np, re
G = pd.read_csv("google_bus_options.csv")
def norm(r): return frozenset(re.sub(r"(UP|DOWN)\d*$|\(.*?\)|STL|\s*EXT\.?$|^0+", "", x.strip().upper()).strip() for x in re.split(r"\+", str(r)) if x.strip())
G["key"] = G.routes.map(norm)
pairs = sorted(set(G.pair_id))
def pct(x): return f"{100*x:4.0f}%"
for cfg in ["C", "ALT"]:
    O = pd.read_csv(f"allopts_{cfg}.csv"); O = O[O.pair_id.isin(pairs)].copy()
    O["key"] = O.routes.map(norm); O["ours_nowait"] = O.total_s - O.first_wait_s
    gbest = G[G.option_rank == 1].set_index("pair_id").duration_s
    print(f"\n=== {cfg}: {len(O)} of our options over {O.pair_id.nunique()} pairs (Google: {len(G)} options)")
    print(" rank | n  | route set also in Google's top 5 | time vs SAME Google option (median diff / bias) | vs Google's best (median ratio)")
    for k in [1, 2, 3]:
        s = O[O["rank"] == k]
        if not len(s): print(f"   {k}  |  0 |"); continue
        hit, diffs = [], []
        for r in s.itertuples():
            g = G[(G.pair_id == r.pair_id) & (G.key == r.key)]
            hit.append(len(g) > 0)
            if len(g): diffs.append((r.ours_nowait - g.duration_s.min()) / g.duration_s.min())
        diffs = np.array(diffs); ratio = (s.ours_nowait.values / gbest.reindex(s.pair_id).values)
        print(f"   {k}  | {len(s):2d} | {sum(hit):2d} ({pct(np.mean(hit))})" + (f"                   | {np.median(np.abs(diffs))*100:5.1f}% / {np.median(diffs)*100:+5.1f}%" if len(diffs) else " | –") + f"                   | {np.median(ratio):.2f}x (p90 {np.quantile(ratio,.9):.2f}x)")
    cov = np.mean([len(set(G[G.pair_id == p].key) & set(O[O.pair_id == p].key)) > 0 for p in pairs])
    g1 = np.mean([any(O[O.pair_id == p].key == G[(G.pair_id == p) & (G.option_rank == 1)].key.iloc[0]) for p in pairs])
    print(f" pairs where any of our options shares a route set with any Google option: {pct(cov)}; Google's #1 appears among ours: {pct(g1)}")
    print(f" options per pair: ours {len(O)/len(pairs):.1f}, Google 5.0")
# Google's own spread: do its 5 options differ in time?
sp = G.groupby("pair_id").duration_s.agg(lambda x: x.max() / x.min())
print(f"\nGoogle: options 2-5 vs its own #1, median {sp.median():.2f}x; pairs where all 5 have the same duration: {(G.groupby('pair_id').duration_s.nunique()==1).sum()}/{len(pairs)}")
