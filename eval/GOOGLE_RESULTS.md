# Google Maps comparison (30 pairs)

Google times: `google_pairs_todo.csv`. Filled from Google Maps transit on Wed 2026-11-25 at the same clock time as each pair's departure. Google can't return past dates; the held-out data is 26–30 Oct. Google's time runs from its first departure to arrival, so it excludes the initial wait. Some of Google's fastest options use the Metro.

Script: `../planner` venv, `gcmp.py` logic. Per-pair table: `google_comparison.csv` (full model, blend mode).

## 1. Planner vs Google (all 30 pairs)

| Comparison | Median error | Within ±20% | Median bias (ours − Google) |
|---|---|---|---|
| As-is (ours includes the first wait, Google doesn't) | 30.6% | 30% | +484 s |
| **Like for like (our first wait removed)** | **19.1%** | **53%** | **+242 s** |
| Like for like, 09:00 | 12.0% | 62% | +242 s |
| Like for like, 13:00 | 18.1% | 71% | −116 s |
| Like for like, 18:30 | 44.9% | 14% | +896 s |

We are higher than Google, most of all in the evening peak.

## 2. Who is right? Both checked against actual GPS rides (18 pairs with held-out rides, 26–30 Oct)

| Estimate | Median error vs actual | Within ±20% | Median bias |
|---|---|---|---|
| **Our planner (ride time)** | **9.6%** | **89%** | −2.8% |
| Google Maps (door to door, without the wait) | 16.3% | 56% | −10.6% |
| GTFS schedule | 21.2% | 44% | −16.9% |

Google's time includes walking, yet it is still shorter than the actual bus ride on these pairs. **Google underestimates Delhi bus travel times, as the official schedule does**; it likely uses the same timetable. Evening example, P22 Mahipal Pur → Andheria More at 18:30: Google 24 min, ours 36 min, actual median 42 min.

## Caveats
- Small sample: 30 pairs, 18 with actual rides. Google was queried for a November weekday, while the actual rides are from late October.
- In part 2, ours is ride time only; Google is door to door. The comparison favours Google, and it still comes out less accurate.
- P08 (airport T1 → T3): Google 32 min vs ours 11 min. Our route (AIRPORT-STL) probably misses the walk or shuttle detail. Needs checking.
