"""pytest for model/v3/v2/journey_band.py (A3 combination)."""
import math, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["V2_D2D_SHIFTS"] = str(Path(__file__).with_name("no_such_file.json"))   # uncalibrated
import journey_band as J


def test_single_component_is_identity():
    p50, p85, s = J.combine([(1000.0, 1300.0)])
    assert abs(p50 - 1000) < 1e-9 and abs(p85 - 1300) < 1e-6


def test_deterministic_components_add():
    p50, p85, s = J.combine([(300, 300), (600, None)])
    assert p50 == 900 and p85 == 900 and s == 0


def test_sum_is_narrower_than_sum_of_p85s():
    comps = [(600, 900), (1200, 1500), (300, 500)]
    p50, p85, _ = J.combine(comps)
    assert p50 == 2100 and p50 < p85 < sum(c[1] for c in comps)


def test_against_monte_carlo():
    comps = [(600, 900), (1200, 1500), (300, 500)]
    rng = np.random.default_rng(0)
    tot = sum(np.exp(math.log(a) + math.log(b / a) / 1.0364 * rng.standard_normal(200000)) for a, b in comps)
    _, p85, _ = J.combine(comps)
    # A3 centres on the sum of medians, which is below the true median of a sum of skewed legs, so P85 runs ~3% low
    # here (exact quantile would need the Fenton-Wilkinson centre); T2.6 shifts absorb this on real journeys.
    assert -0.05 < p85 / np.quantile(tot, 0.85) - 1 < 0.0


def test_on_time_probability():
    assert abs(J.on_time_probability(1000, 1000, 0.2) - 0.5) < 1e-9
    assert abs(J.on_time_probability(1000 * math.exp(1.0364 * 0.2), 1000, 0.2) - 0.85) < 1e-3
