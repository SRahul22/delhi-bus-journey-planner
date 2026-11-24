"""Per-segment Kalman update of live log travel-time ratios (v2, T3.1; used when V2_LIVE=kf).

State per flat route segment f:   x_f = log(observed / historical), stored as x_f = x_c[cell(f)] + d_f,
with variance P_f. Each segment belongs to a 3x3 km cell c with its own state x_c, variance P_c.

Between measurements (time step dt):
    x_c <- x_c * exp(-dt / tau_c)                  (cell relaxes toward 0, i.e. the historical model)
    d_f <- d_f * exp(-dt / tau_s)                  (segment relaxes toward its cell)
    P   <- min(P + q * dt, P0)
Measurement z (one observed segment traversal, log ratio as in live_state.observations()) with noise
R = sigma^2, sigma = ln(p85 / p50) / 1.0364 of that segment and hour (AGENTS.md section 4), floored at
SIG_MIN. Measurements that land in the same STEP-second bin are combined in information form (exactly
equivalent to sequential updates for simultaneous measurements). The cell is updated with every
measurement in it (noise R + RC_ADD, the between-segment spread), then the segment.

Forecast at horizon h (seconds after the state time):
    x(h) = x_c * exp(-h / tau_c) + d * exp(-h / tau_s),   var(h) = min(P + q * h, P0)
exp(x) is clipped to [0.6, 2.5] as in v1.

tau_s, tau_c and q are fitted on 22 Oct only (realtime/v2_live_eval.py --fit), written to
E:\\hackathon-work\\v2\\realtime\\kf_params.json and copied into realtime/kf_params.json.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = (0.6, 2.5)
LCLIP = (math.log(CLIP[0]), math.log(CLIP[1]))
SIG_MIN, SIG_MAX = 0.05, 1.0
DEFAULTS = dict(tau_s=1200.0, tau_c=3600.0, q=2e-5, P0=0.09, Pc0=0.04, rc_add=0.04, step=60.0)


def load_params(path=None):
    p = dict(DEFAULTS)
    for cand in (path, os.environ.get("KF_PARAMS"), os.path.join(HERE, "kf_params.json")):
        if cand and os.path.exists(cand):
            with open(cand) as f:
                p.update({k: float(v) for k, v in json.load(f).get("params", {}).items() if k in DEFAULTS})
            break
    return p


def seg_sigma(model):
    """[n_flat, 2, 24] float32 sigma of log travel time per segment, from the model's p50/p85."""
    p50 = np.concatenate(model.seg).astype(np.float32)
    p85 = np.concatenate(model.p85).astype(np.float32)
    s = np.log(np.maximum(p85, p50 + 1e-3) / np.maximum(p50, 1.0)) / 1.0364
    return np.clip(s, SIG_MIN, SIG_MAX).astype(np.float32)


class SegmentKF:
    def __init__(self, n_flat, gx, gy, sigma, **params):
        p = dict(DEFAULTS); p.update(params)
        self.p = p
        self.tau_s, self.tau_c, self.q = p["tau_s"], p["tau_c"], p["q"]
        self.P0, self.Pc0, self.rc_add, self.step = p["P0"], p["Pc0"], p["rc_add"], p["step"]
        self.n = n_flat
        self.cell = (np.asarray(gx) // 3) * 50 + (np.asarray(gy) // 3)   # 1 km grid (150x150) -> 3 km cells
        self.nc = 2500
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.d = np.zeros(self.n)
        self.P = np.full(self.n, self.P0)
        self.xc = np.zeros(self.nc)
        self.Pc = np.full(self.nc, self.Pc0)
        self.t = None          # state time (unix)
        self.cursor = 0        # observations consumed

    # ------------------------------------------------------------------ dynamics
    def _relax(self, dt):
        if dt <= 0:
            return
        self.xc *= math.exp(-dt / self.tau_c)
        self.d *= math.exp(-dt / self.tau_s)
        np.minimum(self.P + self.q * dt, self.P0, out=self.P)
        np.minimum(self.Pc + self.q * dt, self.Pc0, out=self.Pc)

    def _update(self, f, z, R):
        c = self.cell[f]
        # cell
        wc = 1.0 / (R + self.rc_add)
        uc, ic = np.unique(c, return_inverse=True)
        Ic = np.bincount(ic, weights=wc); Sc = np.bincount(ic, weights=wc * z)
        xc_old = self.xc[uc].copy()
        Pc_new = 1.0 / (1.0 / self.Pc[uc] + Ic)
        self.xc[uc] = Pc_new * (xc_old / self.Pc[uc] + Sc)
        self.Pc[uc] = Pc_new
        # segment (prior mean = updated cell + deviation)
        uf, jf = np.unique(f, return_inverse=True)
        wf = 1.0 / R
        If = np.bincount(jf, weights=wf); Sf = np.bincount(jf, weights=wf * z)
        xprior = self.xc[self.cell[uf]] + self.d[uf]
        P_new = 1.0 / (1.0 / self.P[uf] + If)
        xpost = P_new * (xprior / self.P[uf] + Sf)
        self.d[uf] = xpost - self.xc[self.cell[uf]]
        self.P[uf] = P_new

    def advance(self, obs, now, n_max=None):
        """Consume observations [cursor, n_max) (emission order; all known by `now`) and move the state to
        `now`. obs: dict from LiveState.observations() (f, t0, t1, logr, ...)."""
        n = len(obs["f"]) if n_max is None else n_max
        if self.t is None:
            t0_ = float(now) if n <= self.cursor else float(min(obs["t1"][self.cursor:n].min(), now))
            self.t = math.floor(t0_ / self.step) * self.step   # bins on an absolute grid (incremental == bulk)
        if n > self.cursor:
            sl = slice(self.cursor, n)
            f = obs["f"][sl]; z = obs["logr"][sl]; t1 = np.maximum(obs["t1"][sl], self.t)
            t0 = obs["t0"][sl]
            loc = t0 + 19800
            h = ((loc % 86400) // 3600).astype(np.int64)
            dt = (((loc // 86400) + 3) % 7 >= 5).astype(np.int64)
            R = self.sigma[f, dt, h].astype(np.float64) ** 2
            # time bins
            b = np.floor(t1 / self.step).astype(np.int64)
            o = np.argsort(b, kind="stable")
            b, f, z, R = b[o], f[o], z[o], R[o]
            cuts = np.flatnonzero(np.diff(b)) + 1
            starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [len(b)]])
            for a, e in zip(starts, ends):
                tb = min((b[a] + 1) * self.step, max(now, self.t))
                self._relax(tb - self.t); self.t = max(self.t, tb)
                self._update(f[a:e], z[a:e], R[a:e])
            self.cursor = n
        self._relax(now - self.t)
        self.t = max(self.t, float(now))

    # ------------------------------------------------------------------ outputs
    def state(self):
        """Arrays for a snapshot: per flat segment cell level xc, deviation d, variance P (copies)."""
        return dict(xc=self.xc[self.cell].astype(np.float32), d=self.d.astype(np.float32), P=self.P.astype(np.float32),
                    tau_s=self.tau_s, tau_c=self.tau_c, q=self.q, P0=self.P0)


def forecast(kfs, h, idx=None):
    """Log ratio and variance at horizon h (s, scalar or array broadcastable to idx) from a state() dict."""
    xc = kfs["xc"] if idx is None else kfs["xc"][idx]
    d = kfs["d"] if idx is None else kfs["d"][idx]
    P = kfs["P"] if idx is None else kfs["P"][idx]
    h = np.maximum(np.asarray(h, dtype=np.float64), 0.0)
    x = xc * np.exp(-h / kfs["tau_c"]) + d * np.exp(-h / kfs["tau_s"])
    return np.clip(x, *LCLIP), np.minimum(P + kfs["q"] * h, kfs["P0"])
