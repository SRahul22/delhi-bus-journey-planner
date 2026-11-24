"""Pluggable travel-time model.

Segment ride-time lookup order (per route segment, hour 0-23 IST, daytype weekday|weekend):
  1. observed route segment median from segment_times.csv (n >= MIN_N)
  2. observed median for the same stop pair on any route (n-weighted) at that hour/daytype
  3. GTFS scheduled segment time x hour-of-day congestion factor
     (factor fitted from observed/scheduled ratios when model tables exist, else a prior profile)
  4. distance / fallback speed
Wait at a stop = min(headway/2, WAIT_CAP) from observed stop_headways.csv, else GTFS trips/hour.

Everything is materialised into dense numpy arrays per route at load time, so the router
does O(1) lookups:  seg[r]  float32 [nseg, 2, 24]   (daytype 0=weekday 1=weekend)
                    p85[r]  float32 [nseg, 2, 24]
                    wait[r] float32 [nstops, 2, 24]
"""
import os
import time

import numpy as np
import pandas as pd

from paths import model_dir

MODEL_DIR = model_dir()
MIN_N = 3
WAIT_CAP = 20 * 60
NO_SERVICE_WAIT = 45 * 60  # wait if no trip in that hour at all (penalised, router may still use it)
FALLBACK_SPEED = 15 / 3.6  # m/s

# Prior congestion multiplier on GTFS scheduled times (GTFS is flat ~19.7 km/h all day,
# observed Delhi bus speeds are ~12-18 km/h with AM/PM peaks). Replaced by fitted factors.
PRIOR_FACTOR = np.array([0.95, 0.9, 0.9, 0.9, 0.9, 0.95, 1.05, 1.15, 1.3, 1.35, 1.3, 1.2,
                         1.2, 1.2, 1.2, 1.2, 1.25, 1.35, 1.4, 1.4, 1.3, 1.2, 1.1, 1.0], dtype=np.float32)
WEEKEND_PRIOR = 0.92


class TravelTimeModel:
    def __init__(self, net, model_dir=MODEL_DIR):
        self.net = net
        self.model_dir = model_dir
        self.source = "gtfs_schedule+prior_hour_factor"
        self.stats = {}
        self.seg_stat = "median_s"
        self._has_mean = False
        self.factor = np.stack([PRIOR_FACTOR, PRIOR_FACTOR * WEEKEND_PRIOR])  # [2,24]
        # SEG_MODE: median | mean | blend (median for the first BLEND_S s of a ride, mean after; sums of
        # per-segment medians under-predict long rides because segment times are right-skewed)
        self.mode = os.environ.get("SEG_MODE", "blend")
        self.blend_s = float(os.environ.get("BLEND_S", 600))  # best on holdout (see tests/eval_holdout_modes.txt)
        self.seg_mean = None
        self.load("median_s")
        if self.mode in ("mean", "blend") and self._has_mean:
            med = (self.seg, self.p85, self.factor, self.stats, self.source)
            self.factor = np.stack([PRIOR_FACTOR, PRIOR_FACTOR * WEEKEND_PRIOR])
            self.stats = {}
            self.load("mean_s")
            self.seg_mean = self.seg
            if self.mode == "blend":
                self.seg, self.p85, self.factor, stats, self.source = med
                stats["segment_cells_mean"] = self.stats["segment_cells"]
                self.stats = stats
            self._cum_cache = {}
        else:
            self.mode = "median"
        self.stats["seg_mode"] = self.mode if self.mode != "blend" else f"blend(median<{self.blend_s:.0f}s, then mean)"

    # ------------------------------------------------------------------ loading
    def load(self, stat="median_s"):
        t0 = time.time()
        routes = self.net["routes"]
        stop_ids = self.net["stops"].stop_id.to_numpy()
        seg_path = os.path.join(self.model_dir, "segment_times.csv")
        hw_path = os.path.join(self.model_dir, "stop_headways.csv")
        segdf = None
        if os.path.exists(seg_path):
            try:
                segdf = pd.read_csv(seg_path, dtype={"route_id": str, "from_stop": str, "to_stop": str})
                segdf["dt"] = (segdf.daytype.astype(str).str.lower() == "weekend").astype(np.int8)
                # point estimate per segment: mean if provided (sums of means are unbiased for multi-segment
                # rides; sums of medians under-predict skewed segment times), else median. Override SEG_STAT.
                self._has_mean = "mean_s" in segdf.columns
                self.seg_stat = stat if stat in segdf.columns else "median_s"
                if self.seg_stat != "median_s":
                    segdf["median_s"] = segdf[self.seg_stat].where(segdf[self.seg_stat] > 0, segdf.median_s)
                segdf = segdf[(segdf.median_s > 0) & segdf.hour.between(0, 23)]
            except Exception as e:  # noqa
                print("segment_times.csv unreadable:", e)
                segdf = None
        hwdf = None
        if os.path.exists(hw_path):
            try:
                hwdf = pd.read_csv(hw_path, dtype={"route_id": str, "stop_id": str})
                hwdf["dt"] = (hwdf.daytype.astype(str).str.lower() == "weekend").astype(np.int8)
                hwdf = hwdf[(hwdf.median_headway_s > 0) & hwdf.hour.between(0, 23)]
            except Exception as e:  # noqa
                print("stop_headways.csv unreadable:", e)
                hwdf = None

        # flat table of all route segments
        rows_r, rows_i, rows_a, rows_b = [], [], [], []
        for ri, r in enumerate(routes):
            s = r["stops"]
            n = len(s) - 1
            rows_r.append(np.full(n, ri, np.int32))
            rows_i.append(np.arange(n, dtype=np.int32))
            rows_a.append(s[:-1])
            rows_b.append(s[1:])
        R = np.concatenate(rows_r); I = np.concatenate(rows_i)
        A = np.concatenate(rows_a); B = np.concatenate(rows_b)
        nseg = len(R)
        sched = np.concatenate([r["sched_seg"] for r in routes])  # [nseg,24]
        dist = np.concatenate([r["dist"] for r in routes])
        sched = np.where(sched <= 0, dist[:, None] / FALLBACK_SPEED, sched)

        obs = np.full((nseg, 2, 24), np.nan, np.float32)
        obs85 = np.full((nseg, 2, 24), np.nan, np.float32)
        level = np.zeros(4, np.int64)
        has = np.zeros(1, bool)
        if segdf is not None and len(segdf):
            rid = np.array([r["route_id"] for r in routes])
            key = pd.DataFrame(dict(seg=np.arange(nseg), route_id=rid[R], from_stop=stop_ids[A], to_stop=stop_ids[B]))
            good = segdf[segdf.n >= MIN_N]
            # 1. exact route segment
            m1 = key.merge(good, on=["route_id", "from_stop", "to_stop"])
            obs[m1.seg.values, m1.dt.values, m1.hour.values] = m1.median_s.values
            if "p85_s" in m1:
                obs85[m1.seg.values, m1.dt.values, m1.hour.values] = m1.p85_s.values
            # 2. same stop pair, any route (n-weighted mean of medians)
            segdf = segdf.assign(w=segdf.median_s * segdf.n, w85=segdf.get("p85_s", segdf.median_s * 1.25) * segdf.n)
            pair = segdf.groupby(["from_stop", "to_stop", "dt", "hour"], as_index=False)[["w", "w85", "n"]].sum()
            pair = pair[pair.n >= MIN_N]
            pair["med"] = pair.w / pair.n
            pair["p85"] = pair.w85 / pair.n
            m2 = key.merge(pair, on=["from_stop", "to_stop"])
            sel = np.isnan(obs[m2.seg.values, m2.dt.values, m2.hour.values])
            m2 = m2[sel]
            level[0] = int(np.isfinite(obs).sum())
            obs[m2.seg.values, m2.dt.values, m2.hour.values] = m2.med.values
            obs85[m2.seg.values, m2.dt.values, m2.hour.values] = m2.p85.values
            level[1] = int(np.isfinite(obs).sum()) - level[0]
            # fit hour-of-day factor: observed / scheduled, n-weighted, on matched segments
            fin = np.isfinite(obs)
            fac = np.array(self.factor)
            fitted = np.zeros((2, 24), bool)
            for dt in range(2):
                for h in range(24):
                    m = fin[:, dt, h]
                    if m.sum() >= 50:
                        ratio = obs[m, dt, h].sum() / max(sched[m, h].sum(), 1)
                        fac[dt, h] = float(np.clip(ratio, 0.5, 3.0))
                        fitted[dt, h] = True
            # borrow the other daytype's fitted factor where one is missing
            for h in range(24):
                if fitted[0, h] and not fitted[1, h]:
                    fac[1, h] = fac[0, h] * WEEKEND_PRIOR
                elif fitted[1, h] and not fitted[0, h]:
                    fac[0, h] = fac[1, h] / WEEKEND_PRIOR
            self.factor = fac
            self.source = "observed_gps_segments (fallback: schedule x fitted hour factor)"
        # 3. schedule x hour factor x per-segment relative slowness (learned from the hours we did observe)
        base = sched[:, None, :] * self.factor[None, :, :]
        fin = np.isfinite(obs)
        rel = np.ones(nseg, np.float32)
        rel85 = np.full(nseg, 1.25, np.float32)
        if fin.any():
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(fin, obs / base, np.nan)
                r85 = np.where(fin & np.isfinite(obs85), obs85 / np.maximum(obs, 1), np.nan)
                cnt = fin.sum(axis=(1, 2))
                has = cnt > 0
                rel[has] = np.clip(np.nanmedian(r[has].reshape(has.sum(), -1), axis=1), 0.5, 2.5)
                has85 = np.isfinite(r85).any(axis=(1, 2))
                rel85[has85] = np.clip(np.nanmedian(r85[has85].reshape(has85.sum(), -1), axis=1), 1.0, 2.5)
            level[3] = int((~fin & has[:, None, None]).sum())
        self.stats["segments_with_any_observation"] = int((rel != 1).sum()) if fin.any() else 0
        fallback = base * rel[:, None, None]
        miss = ~fin
        level[2] = int(miss.sum()) - int(level[3])
        obs = np.where(miss, fallback, obs).astype(np.float32)
        obs85 = np.where(np.isfinite(obs85), obs85, obs * rel85[:, None, None]).astype(np.float32)
        obs85 = np.maximum(obs85, obs)
        self.stats["segment_cells"] = dict(route_exact=int(level[0]), stop_pair=int(level[1]),
                                           schedule_x_segment_ratio=int(level[3]), schedule_scaled=int(level[2]))

        # waits
        tph = np.stack([r["trips_per_hour"] for r in routes]).astype(np.float32)  # [nr,24]
        gtfs_wait = np.where(tph > 0, np.minimum(3600.0 / np.maximum(tph, 1) / 2, WAIT_CAP), NO_SERVICE_WAIT)
        stop_off = np.cumsum([0] + [len(r["stops"]) for r in routes])
        nst = stop_off[-1]
        SR = np.concatenate([np.full(len(r["stops"]), ri, np.int32) for ri, r in enumerate(routes)])
        SS = np.concatenate([r["stops"] for r in routes])
        # shift GTFS trips/hour by scheduled offset from origin to this stop
        OFF = np.concatenate([r["offset"] for r in routes])
        wait = np.empty((nst, 2, 24), np.float32)
        for h in range(24):
            hh = ((h * 3600 - OFF) // 3600 % 24).astype(int)  # hour the bus left its terminal
            wait[:, 0, h] = gtfs_wait[SR, hh]
            wait[:, 1, h] = gtfs_wait[SR, hh]
        n_obs_wait = 0
        if hwdf is not None and len(hwdf):
            rid = np.array([r["route_id"] for r in routes])
            key = pd.DataFrame(dict(pos=np.arange(nst), route_id=rid[SR], stop_id=stop_ids[SS]))
            m = key.merge(hwdf[hwdf.n >= MIN_N], on=["route_id", "stop_id"])
            vals = np.minimum(m.median_headway_s.values / 2, WAIT_CAP)
            wait[m.pos.values, m.dt.values, m.hour.values] = vals
            n_obs_wait = len(m)
        self.stats["wait_cells_observed"] = n_obs_wait

        self.seg, self.p85, self.wait = [], [], []
        off = 0
        for ri, r in enumerate(routes):
            n = len(r["stops"]) - 1
            self.seg.append(obs[off:off + n])
            self.p85.append(obs85[off:off + n])
            off += n
            self.wait.append(wait[stop_off[ri]:stop_off[ri + 1]])
        self._cum_cache = {}
        self.stats["factor_weekday"] = [round(float(x), 2) for x in self.factor[0]]
        self.stats["load_s"] = round(time.time() - t0, 2)
        print(f"TravelTimeModel: {self.source}; {self.stats}")

    # ------------------------------------------------------------------ lookups
    def cum(self, dt, h):
        """Per route cumulative ride seconds along pattern for (daytype,hour), as python lists."""
        k = (dt, h)
        c = self._cum_cache.get(k)
        if c is None:
            c = [np.concatenate([[0.0], np.cumsum(s[:, dt, h])]).tolist() for s in self.seg]
            self._cum_cache[k] = c
        return c

    def waits(self, dt, h):
        k = ("w", dt, h)
        c = self._cum_cache.get(k)
        if c is None:
            c = [w[:, dt, h].tolist() for w in self.wait]
            self._cum_cache[k] = c
        return c

    def ride(self, ri, i, j, t_sec, dt):
        """Exact time-dependent ride from pattern pos i to j starting at t_sec (seconds since midnight).
        Returns (point_s, p85_s). In blend mode, segments after the first blend_s seconds use the mean."""
        seg, p85 = self.seg[ri], self.p85[ri]
        segm = self.seg_mean[ri] if (self.mode == "blend" and self.seg_mean is not None) else None
        t = t_sec
        tot = tot85 = 0.0
        for k in range(i, j):
            h = int(t // 3600) % 24
            v = float(seg[k, dt, h])
            if segm is not None and tot >= self.blend_s:
                v = float(segm[k, dt, h])
            tot += v
            tot85 += max(float(p85[k, dt, h]), v)
            t += v
        return tot, tot85

    def wait_s(self, ri, i, t_sec, dt):
        return float(self.wait[ri][i, dt, int(t_sec // 3600) % 24])
