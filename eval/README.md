# Evaluation: test pairs, planner sanity run, actual and Google comparison

Goal (judging criterion): estimates should hold up against **known actual travel times** for sample location pairs and against **Google Maps**.

## Files

| File | What |
|---|---|
| `test_pairs.csv` | 60 origin–destination pairs from ticketing data (50 direct, 10 needing a transfer), with stop names, GTFS stop ids, coordinates, straight-line km, departure, ticket count, ticket route(s), GTFS scheduled ride on the ticket route |
| `google_pairs_todo.csv` | 30 most important pairs with a ready-to-click Google Maps transit URL and an empty `google_s` column. **To fill in by hand** (no API key was available) |
| `planner_results.csv` | Planner `/journey` result for every pair (fastest option: time, transfers, walk, wait, routes), GTFS schedule for the same legs, speed band, flags |
| `actual_rides.csv` | Observed GPS bus ride times on held-out weekdays (26–30 Oct) for direct pairs, on the ticket routes, ±1 h of the departure |
| `comparison.csv` | Planner predicted ride vs observed held-out ride vs GTFS schedule, per pair |
| `match_report.txt` | Ticket stop name → GTFS match rate |
| `agg_tickets.py` | Runs on Windows: aggregates the 2.56 M tickets to (start, end, route) counts with hour histograms |
| `build_pairs.py` | Name matching, stratified pair selection, departure slots, Google todo |
| `run_planner.py` | Calls the running planner for each pair and flags implausible results |
| `actual_rides.py` | Ships a small job to Windows that reads `E:\hackathon-work\pass\pass_DD.npz` (the data agent's stop passings) for days 25–31 |
| `compare.py` | Joins planner results with actual rides |

## How the pairs were chosen

1. `agg_tickets.py` (Windows, ~40 s): 2,559,899 tickets → 402k (boarding stop, alighting stop, route) combinations, with tickets per hour. Ticket peaks: 09:00 (351k) and 18:00 (235k); midday ~70k/h.
2. **Name matching** to GTFS stops. Ticket names are short ("Ashram"); GTFS names are often compound ("Maharani Bagh / Ashram"). Order: exact name, normalised name (case, punctuation, common abbreviations like Xing/Sation/Extn/(T)), match to one part of a compound "A / B" name, fuzzy (difflib ≥ 0.85). For OD pairs we first match **within the stops of the ticket's route** (route_short_name = ticket `route`, e.g. `8UP`), which also picks the correct side of the road.
   - All 7,552 ticket stop names, without route context: 82.8% matched (exact 34.9%, normalised 20.6%, part 12.2%, fuzzy 15.1%); **92.1% of stop mentions** weighted by tickets.
   - Top 6,000 (start, end, route) rows, route-aware: **both ends matched for 93.2%** of rows (93.9% of their tickets). Most common unmatched names: "Mahipal Pur", "Badarpur Border", "Jahangir Puri Metro Station" when the ticket's route doesn't list them.
3. Pairs are aggregated over routes. Bands by straight-line distance: short < 5 km, medium 5–12 km, long > 12 km. Within each band, pairs are taken by ticket count in round-robin over 5 regions (Central = within 5 km of Connaught Place; N/E/S/W by bearing). Duplicates are removed by location (~600 m cells), so name variants and reversed pairs don't repeat, and no stop area appears more than twice. Result: 18 short, 18 medium, 14 long direct; regions South 17, West 14, Central 10, North 10, East 9.
4. **Transfer pairs** (10): chains A→B on route r1 and B→C on route r2, both with ≥ 40 tickets, different route numbers, A→C at least 4 km and further than A→B×0.8 (the transfer makes progress), and **no GTFS route serves both A and C** by name. Ranked by min(tickets of the two legs).
5. **Departure** on Wed 2026-10-28 (held-out week): 25 × 09:00 (AM peak), 15 × 13:00 (off-peak), 20 × 18:30 (PM peak). Each pair gets the slot where its own tickets are relatively strongest (≥ 15% of its tickets in that window).

Limitations: the South/West airport corridor (Mahipalpur, Kapashera, Badarpur) dominates ticket volume, so it is over-represented, especially among transfer pairs. Origins and destinations are bus-stop coordinates, so walking access is near zero; that is optimistic compared with real door-to-door trips. Weekend is not covered.

## Filling in Google Maps times

No Google Maps API key is available (`GOOGLE_MAPS_API_KEY` is not set), and nothing was scraped. To fill in by hand (~15 min for 30 pairs):

1. Open each `google_url` in `google_pairs_todo.csv`. In Google Maps choose **Depart at**, a **weekday** (Wednesday) and the time in `depart` (09:00 / 13:00 / 18:30).
2. Under transit options choose **Bus** only (Google otherwise prefers the Metro), then take the fastest bus option's total door-to-door time.
3. Enter it in seconds in `google_s` (e.g. 42 min → 2520). Put notes such as "only metro offered" in `google_notes`.
4. Run:
   ```bash
   cd planner && .venv/bin/python evaluate.py --pairs ../eval/google_pairs_todo.csv
   ```
   Extra columns are ignored and empty `google_s` rows are skipped. Output goes to `planner/eval_out/pairs_predictions.csv`.

## Re-running

Current, OS-independent commands: [`../REPRODUCE.md`](../REPRODUCE.md) section 6. `actual_rides.py` now runs locally when
the pass files are on the same machine (`--pass-dir`, default `$WORK_DIR/pass`); `--ssh` keeps the old Mac→Windows mode.
The commands below are the original Mac workflow.

```bash
# Windows aggregation (only needed once)
cat eval/agg_tickets.py | $S "python -"   # writes E:\hackathon-work\eval\od_route.csv, stop_names.csv
# copy od_route.csv + stop_names.csv to a local dir, then
cd eval && ../planner/.venv/bin/python build_pairs.py --work <dir>
(cd ../planner && PORT=8010 ./run.sh &) ; ../planner/.venv/bin/python run_planner.py --url http://localhost:8010
../planner/.venv/bin/python actual_rides.py      # needs pass_26..30.npz on Windows
../planner/.venv/bin/python compare.py
```

## Findings

Run at 17:30 IST on 2026-11-24 against model **v1.5** (GPS days 01–10 and 15; the full 1–24 Oct tables were not in `model/` yet). Re-run `run_planner.py` and `compare.py` when the full model lands.

### 1. Against actual held-out travel times (the key result)

`actual_rides.py` measured real bus rides on held-out weekdays 26–30 Oct for the direct pairs: trips on the pair's ticket routes that pass the boarding stop and then the alighting stop, departing within ±1 h of the slot. 38 of 50 direct pairs have ≥ 3 observations (median 63 rides per pair).

| In-vehicle ride vs actual median | median abs. error | within ±20% | median bias |
|---|---|---|---|
| **Planner (GPS model)** | **5.8%** | **84%** | −2.0% |
| GTFS schedule on the same route | 20.7% | 49% | −12.4% (schedule too optimistic) |

By band (median abs. error, model vs schedule): short 4.8% vs 24.3%, medium 6.6% vs 20.7%, long 6.1% vs 9.6%. 76% of predictions fall inside the observed 25–75% range. Details: `comparison.csv`.

Caveats: (a) "actual" uses the data agent's stop-passing extraction (`pass_DD.npz`), so it checks the model out of sample in time, not the GPS-to-stop matching itself. (b) The prediction is the in-vehicle time of the planner's option on the ticket route, which may board or alight one stop away from the ticket stop. (c) Wait and walk are not checked here; only Google gives a door-to-door check.

Worst cases (model too fast):
- **P25** CRPF Camp → Old Delhi Rly Stn, 227A, 13:00: predicted 1489 s vs actual 2222 s (−33%, n = 22). Old Delhi / Chandni Chowk midday congestion is underestimated. Likely cause: sparse cells for this route-hour, so the fallback (schedule × hour factor) is used.
- **P07** Samalkha → Kapashera Border (terminal), 09:00: −31%, but the actual IQR is very wide (515–1333 s). The terminal approach includes queueing/dwell before the terminal.
- **P01** West Patel Nagar → Karol Bagh Metro, 13:00: −25% (n = 144). **P14** Alipur → Jahangirpuri Metro: −22%. Both are short trips through busy markets.
- P17 (−58%) has only 3 observations; ignore.
Model too slow: **P50** Mori Gate → Mayur Vihar III (211UP, 18:30) +20%, **P33** Scindia House → Defence Colony (18:30) +19%. The PM-peak factor (×1.33 at 18:00) may be too strong for some routes.

### 2. Plausibility of all 60 pairs (`planner_results.csv`)

- **60/60 pairs got options**, `/journey` 115 ms median (max 307 ms). **No hard flags**: none over 3 h, none faster than 25 km/h door-to-door, no wait over 30 min, no walk over 2 km, no ride < 0.6× or > 2× schedule.
- Door-to-door speed (straight-line km / total time): median 11.6 km/h (IQR 9.6–13.9), inside the 10–18 km/h reference band for 67% of pairs. By slot: 09:00 13.0, 13:00 11.8, 18:30 9.9 km/h. Planner ride / GTFS schedule on the same legs: median 1.04, PM peak 1.20.
- Soft flags:
  - 18 pairs are slow (6–10 km/h), mostly short or PM-peak trips. The held-out actuals confirm these (e.g. P16 Zakhira → Azadpur: 2134 s predicted vs 2120 s actual; P11: 1382 vs 1399 s), so this is real traffic, not a bug.
  - 2 pairs are fast (18–25 km/h): P28 Dhaula Kuan → Rangpuri on NH-8 (actual ride 1141 s vs 1006 s predicted, plausible) and P46 Jahangirpuri → Singhu Border on GT Karnal Road.
  - Transfer pairs are the slowest: P57 6.5 km/h (49 min for 5.3 km), P52 7.2, P53 7.6 (86 min for 10.9 km, Shiv Murti → Uttam Nagar). These are plausible for 2 buses, but they are the first ones to check against Google.

Routing observations for the planner agent:
1. **11 of 50 single-route ticket pairs return a transfer journey as the fastest option** (e.g. P49 Nehru Place → Anand Vihar: 534 + 543A rather than the direct 534 that riders use; P33 440 + 623 rather than the direct 440). Each pair also offers a direct option. Riders, like Google, prefer a direct bus that is a few minutes slower. Suggest a transfer penalty (~5–10 min) in "recommended"/fastest ranking.
2. 21 pairs' fastest option uses a route that is not the ticket route. Often this is an equivalent parallel route (fine), but check that `alt_routes` lists the ticket route.
3. 2 of 10 "transfer" pairs were answered with a single bus (P55 764, P58 473) because those routes serve nearby stops with different names. This is fine and shows walking access works.
4. Display bug: one route's `route_short_name` is a whole sentence ("VASANT VIHAR METRO STATION - ANDHERIA BAGH MORE / CHHATARPUR METRO STATION UP", P52). Truncate it or use route_long_name in the UI.
5. P38 (18:30) and P47 (09:00) have waits of 20–21 min across two transfers, which inflates total time. This is plausible but worth showing as a range (p85).

For the data agent:
- Biggest error source is congested CBD/market segments at midday (P25, P01, P14), where the model is up to 25–33% too fast. Check cell coverage (n) for these routes and hours once the full 1–24 Oct tables are in. Fallback cells (schedule × factor) are a likely culprit.
- Segments approaching terminals (Kapashera Border, Badarpur Border) have very wide spreads, probably from pre-terminal dwell or queueing. p85 is a better guide there.

### 3. Against Google Maps
Pending: fill `google_s` in `google_pairs_todo.csv` (see above) and run `evaluate.py --pairs`.
