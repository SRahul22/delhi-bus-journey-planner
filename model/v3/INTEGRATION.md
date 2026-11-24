# Integrating model v3 into the planner (not applied; optional)

Recommendation: the planner's current median→mean blend is already about as good (see README.md). Wire this in
only if you want the small extra gain (mainly on rides with sparse data). Pure python + numpy; no LightGBM needed.

Artefacts needed in `model/v3/`: `predict.py`, `v3_lookup.npz`, `lgbm_model.json`, `calib.json`
(on Windows they are produced in `E:\hackathon-work\v3`; copy them into `E:\hackathon-project\model\v3`).
Check: `python model/v3/predict.py` (self-test against LightGBM's own predictions; needs `check_rides.json`).

## Change in `planner/router.py`, `time_journey()` (one place, per bus leg)

```python
# top of router.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model", "v3"))
try:
    from predict import correct as v3_correct
except Exception as e:  # artefacts missing -> keep the plain model
    print("model v3 not loaded:", e); v3_correct = None

# in time_journey(), right after:   ride, ride85 = model.ride(ri, i, j, t_dep, dt)
            if v3_correct is not None and os.environ.get("V3_MODE", "lgbm") != "off":
                try:
                    dep_dt = base_date + timedelta(seconds=t_dep)          # IST, naive
                    new = v3_correct(r["route_id"], s0_id, s1_id, dep_dt, ride, None, j - i, None)
                    if new:
                        ride85 *= new / max(ride, 1.0); ride = new        # keep the p85 band proportional
                except Exception:
                    pass
```
where `s0_id, s1_id` are the GTFS stop_id values of `s0, s1` (e.g. `self.net["stops"].stop_id.iloc[s0]`; the
router uses internal stop indices). `route_id` and stop ids may be str or int.

Notes
- predict.py recomputes its own base (sum of 1–24 Oct medians/means) so serving matches training. The planner's
  `ride` is only used as a fallback (× bucket calibration) when the route/stops aren't in v3's patterns.
- Only `time_journey()` (final timing of the chosen options) changes; RAPTOR search still uses the table, so
  option ranking is unchanged. Cost ~1–5 ms per bus leg.
- `V3_MODE=calib` uses only the calibration table; `V3_MODE=off` disables it.
- Only weekday/weekend + hour are modelled; holidays are not.
