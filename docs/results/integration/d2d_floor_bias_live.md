# Door-to-door: floor, bias fix, live mode (validation 19-24 Oct)

Common subset of the A/B/C rerun (journeys scored in every config). Config C = XGBoost only on rides over 60 min + pooled waits (now live on port 8000). Test days 25-31 were never used.
Scripts: `eval/d2d_common.py`, `d2d_plan_extra.py`, `d2d_oracle.py`, `d2d_calib.py`, `d2d_live.py`. Outputs: `E:\hackathon-work\v2\analysis\`.

| Row | d2d median error | within ±20% | median bias | n |
|---|---|---|---|---|
| **C, historical (live now)** | **24.4%** | **42.1%** | −869 s | 1,993 |
| Floor: typical-day oracle (mean wait at the same clock time over days 19-24 + typical ride) | 16.4% | 57.3% | +117 s | 1,993 |
| Floor: typical-day oracle waits + our rides | 17.6% | 54.7% | +68 s | 1,993 |
| Floor: same-day oracle (that day's expected wait + typical ride; optimistic) | 11.8% | 70.7% | +22 s | 1,993 |
| Same-day oracle waits + our rides | 13.3% | 66.2% | −8 s | 1,993 |
| Our waits + oracle rides | 22.9% | 45.8% | −789 s | 1,993 |
| C + wait calibration, additive per leg (fit on days 13-18) | 23.7% | 44.0% | −306 s | 1,996 |
| C + wait calibration, ratio per leg (fit on days 13-18) | 23.5% | 43.4% | −328 s | 1,996 |
| C + one factor on the total (days 13-18) | 25.2% | 40.3% | +153 s | 1,996 |
| C, median bias removed on validation itself (upper bound for any pure shift) | 24.9% | 40.9% | 0 s | 1,996 |
| 22 Oct only: C, historical (same journeys as the live rows) | 27.5% | 37.7% | −1,610 s | 1,017 |
| 22 Oct: C + live first wait (V2_LIVE=kf) | 26.4% | 39.6% | −1,611 s | 1,017 |
| 22 Oct: C + live first wait + live transfer waits (snapshot at arrival) | 26.4% | 40.6% | −1,552 s | 1,017 |
| 22 Oct, journeys where a live bus was predicted for the first leg (55%): C historical | 22.6% | 47.2% | −913 s | 564 |
| same, C + live first + transfer waits | **18.8%** | **52.0%** | −908 s | 564 |

## Conclusion 
1. **Waits are the problem, not rides.** Perfect waits with our rides give 13.3%; perfect rides with our waits stay at 22.9%.
2. **Without live data the realistic floor is about 16% (57% within ±20%).** A perfect typical-day model gets 16.4%; we are at 24.4%, about 8 points above it.
3. **Calibration is not the fix (under 1 point).** Real waits are heavy-tailed (median real 893 s vs planned 596 s; 90th percentile real wait 4,049 s), so a constant shift or factor cannot correct them. The gap needs a better per-route wait distribution (irregularity, bunching, missing buses) and wider honest bands.
4. **Live data helps where it can see a bus, but it doesn't fix the bias.** Where a live bus was predicted for the first leg (55% of journeys), the error drops from 22.6% to 18.8% and the first-wait error halves (441 s → 224 s). Over all journeys it moves only 27.5% → 26.4%, because 45% have no live bus to track and the live wait is capped at 2× the (too optimistic) planned wait, so the −27 min bias on 22 Oct remains. Live data works best combined with better historical waits (the v2/waits fix, config D), which is untested together.

## Caveats
- The same-day oracle is optimistic: for routes with gaps over 30 min its wait equals the rider's real wait. The typical-day row is the fair floor for a historical model.
- The oracle waits use the scoring rules of `realized_journeys.py` (planned route_id, a bus that really reaches the alighting stop, departure at `t_last`), including buses parked at a stop.
- Typical rides use all of days 19-24, including the realized day; the median cell has 11 rides.
- The calibration fit is partly in-sample for the pooled wait model (built from days 1-18). Small cells fall back to peak/off-peak offsets.
