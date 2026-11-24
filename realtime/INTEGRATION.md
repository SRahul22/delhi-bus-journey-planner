# Integrating the real-time layer into the planner app

`realtime/planner_patch.diff` changes only `planner/app.py`. It adds about 45 lines and leaves the router
and the model untouched. Apply it from the repo root:

```bash
patch -p1 --dry-run < realtime/planner_patch.diff && patch -p1 < realtime/planner_patch.diff
```

The patch was tested on a copy of the planner (as of 17:15): `/journey` with and without `realtime=1`,
`/realtime/status` and `/realtime/clock`. Startup adds about 3.5 s and roughly 250 MB of RAM. A
realtime `/journey` call takes about 280–310 ms, against 230–240 ms without it.

If `app.py` has changed since and the patch no longer applies, make these edits by hand:

1. After `planner = Planner()`, load the service, which is optional and fails soft:
   ```python
   RT = None
   if os.environ.get("REALTIME", "1") != "0":
       try:
           import sys
           sys.path.insert(0, os.path.join(os.path.dirname(HERE), "realtime"))
           from service import RealtimeService
           RT = RealtimeService(planner)
       except Exception as e:
           print("realtime layer disabled:", repr(e))
   ```
2. In `/journey`, add the parameter `realtime: bool = Query(False)`, then replace the final `return` with:
   ```python
   p = planner
   if realtime and RT is not None:
       dep = dep or RT.now_dt()             # no depart -> replay clock "now"
       p = RT.planner_for(dep) or planner   # live only near the replay clock on the replay day
   res = p.plan(from_lat, from_lng, to_lat, to_lng, dep, max_options=max_options)
   if realtime:
       res["realtime"] = dict(applied=p is not planner, clock=RT.feed.now_iso() if RT else None)
   return res
   ```
3. Add the new endpoints:
   - `GET /realtime/status` returns `RT.status()`: clock, number of observations in the window, city
     ratio, live vehicles, and the spread of live ratios.
   - `POST /realtime/clock?start=2026-10-28T18:00&speed=10` calls `RT.set_clock(start, speed)` to move
     the replay clock for demos.
   - `GET /realtime/eta?stop_id=&route_id=` returns `RT.live.next_bus_eta(stop_id, route_id)`.
4. In `/admin/reload-model`, add `if RT is not None: RT.reload()` after `planner.reload_model()`, so that
   the live ratios use the new historical model.

Environment variables:

| variable | effect |
|---|---|
| `REALTIME=0` | skips loading the layer |
| `RT_START` | start of the replay clock, in IST (default `2026-10-28T08:30`) |
| `RT_SPEED` | speed multiplier for the replay clock |
| `RT_TICK` | update interval in seconds (default 15) |
| `RT_REPLAY` | path to the replay file |

On Windows, the replay file is found automatically at `E:\hackathon-work\realtime\replay_28.npz`. The
route shapes are read from `model/work/route_shapes.csv`, falling back to
`E:\hackathon-work\route_shapes.csv`.

Thread safety: the background thread swaps in a new immutable `Snapshot` on each update. Requests read
only snapshots, plus the `LiveState` arrays that never change after construction. The base `Planner` and
its `TravelTimeModel` are shared with the realtime copy and never modified.

UI idea (optional, not done): a "live traffic" checkbox in `static/index.html` that adds `&realtime=1`,
plus a small label showing the replay clock taken from the response's `realtime.clock`.
