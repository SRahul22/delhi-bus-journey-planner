# v2 waits: plan (branch v2/waits, from v2/integration 6c9441f)

Written 2026-11-24 ~22:20 IST, before coding. Analysis scripts: `E:\hackathon-work\v2\waits\a1.py`–`a4.py`
(input: `E:\hackathon-work\v2\eval\realized_val.csv` = config A = v1 defaults, 2,983 scored OD-times, days 19–24).

## 1. Why door-to-door plans are ~21 min too short

Rides are fine (median bias −6 s). The gap is waits, and it has three causes, all in how the planner *predicts*
waits, not in the evaluation:

1. **Wrong formula.** v1 plans `median headway / 2`. Delhi buses bunch, so the median gap is much shorter than the
   gap a random rider lands in. For the first-leg cells with ≥20 observed gaps, mean(H²)/(2·mean H) is **2.5×**
   median/2. Planned median first wait 275 s; same-day real median 629 s; the v2 gap table (built from days 1–18)
   says 684 s. So the v2 "gaps" formula is already right where it has data.
2. **Sparse cells fall back to v1.** `V2_WAIT=gaps` only uses cells with n ≥ 20 gaps; the rest fall back to v1 /
   GTFS (~600 s). But sparse cells are exactly the infrequent routes, where real waits are long:

   | gaps in cell (days 1–18) | journeys | plan (v1) | v2 gap table | real median |
   |---|---|---|---|---|
   | none | 86 | 600 | – | 3,419 |
   | 1–4 | 139 | 600 | 2,050 | 2,829 |
   | 5–9 | 226 | 754 | 1,892 | 1,976 |
   | 10–19 | 442 | 588 | 1,386 | 1,235 |
   | 20–39 | 903 | 398 | 928 | 828 |
   | 40+ | 1,187 | 224 | 560 | 498 |

   Worse, once frequent routes get honest waits, RAPTOR *prefers* the sparse routes that still look cheap. This is
   why config B (gaps) barely moved the first-wait bias (−462 → −444 s).
3. **20-min cap.** All waits are capped at 1,200 s, but 39% of real first waits are longer.

Transfers use the same wait arrays, so transfer waits (bias −598 s) have the same causes.

**Is the realized-wait definition fair?** Mostly yes:
- Same route number vs exact route_id makes almost no difference (median 834 vs 845 s).
- The rule rejects ~19% of stop visits as wrong-direction or not reaching the alighting stop. That is correct.
- GPS sees ~1.27× longer gaps than the GTFS timetable. Either buses are missing from GPS or service runs below
  timetable; we can't tell which from this data. Both the planner's tables and the realized truth use the same
  GPS, so they are consistent, but absolute waits may be overstated if buses are missing. Stated as a caveat.
- Counting any parallel route (T2.2 "pooled") is a different, more generous definition. It is kept as a
  sensitivity number only; the official metric is not changed.

## 2. The fix (behind flags, defaults = v1)

On top of `V2_WAIT=gaps`, with the 1–18 gap table for validation (`V2_HEADWAYS=...stop_headways_v2_1_18.csv`):
- `V2_WAIT_MIN_N` (exists, default 20): use the gap table from n ≥ 1–3 instead of 20.
- `V2_WAIT_CAP` (new, default 1200): cap for v2 gap waits, try 3,600 s.
- `V2_WAIT_FILL` (new, default 0): hours where a stop–route has gap data on other hours but none at this hour
  (no bus within 3 h) get the cap, instead of the timetable's wait.
- Wait p85 for the band comes from the gap table (exists).
- If the band is still badly off after this, a door-to-door p85 calibration (T2.6 method, leave-one-day-out).

## 3. How to prove it on validation

- Rerun `eval/realized_journeys.py --set val --workers 9` with the new flags, compare against config A with
  `eval/rerun_compare.py` (own scored journeys and the common subset). One clean gaps run (1–18 table, v2
  defaults) as an extra reference.
- Report: door-to-door median error, within ±20%, median bias, first-wait and transfer bias, under p50/p85/p90,
  ride leg error (must not regress), share scored.
- Pick settings on validation only. Test week (25–31) once at the end, only if validation clearly beats v1.

## 4. Can it be done in a few hours?

Yes. The code change is small (a few lines in `ttmodel.py`); each validation run takes ~30–45 min. I expect the
first-wait bias to drop from −462 s to around −100 s and door-to-door median error to fall a few points. It will
not reach ±60 s on mean wait: real waits have a long tail (19% over 40 min), and the mean is dominated by it.
