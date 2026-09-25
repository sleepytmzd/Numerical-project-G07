"""Tests for importance sampling (Stage 5, Tamzeed Mahfuz, 2105012)."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from src.benchmark import ESTIMATORS
from src.config import N_STEPS, SCENARIOS
from src.estimators import make_seed_seq, plain_mc
from src.payoffs import payoff_for
from src.vr_importance import (choose_theta, effective_sample_size, heuristic_theta,
                               importance_mc, likelihood_ratio, weighted_samples)

# 252-date (discretely monitored) references from Stages 2-4.
PAPER_REF = 7.0941
DEEP_REF = 3.3857


def test_smoke_tail_probability():
    """Task 0: P(Z > 4) with a proposal shifted to mean 4 matches norm.sf(4).

    Checked over 50 independent runs (not one seed): the pooled estimate is
    within 3 SE and the per-run z-scores are centred with unit spread.
    """
    exact = norm.sf(4.0)
    n, theta = 20_000, 4.0
    ests, ses = [], []
    for seed in range(50):
        z_tilde = np.random.default_rng(seed).standard_normal(n)
        x = z_tilde + theta                  # a draw from N(4, 1)
        samples = (x > 4.0) * likelihood_ratio(z_tilde, theta, T=1.0)
        ests.append(samples.mean())
        ses.append(samples.std(ddof=1) / math.sqrt(n))
    ests, ses = np.array(ests), np.array(ses)
    pooled_se = math.sqrt(np.mean(ses**2) / len(ests))
    assert abs(ests.mean() - exact) < 3 * pooled_se
    z = (ests - exact) / ses
    assert abs(z.mean()) < 0.5 and 0.7 < z.std(ddof=1) < 1.3
    assert np.mean(ses) / exact < 0.03       # plain MC would see ~0.6 hits in 2e4 draws


def test_likelihood_ratio_has_unit_mean():
    rng = np.random.default_rng(1)
    w = likelihood_ratio(rng.standard_normal(200_000), 1.5, T=1.0)
    assert abs(w.mean() - 1.0) < 4 * w.std(ddof=1) / math.sqrt(len(w))


def test_heuristic_theta_puts_log_price_on_barrier():
    sc = SCENARIOS["deep_barrier"]
    theta0 = heuristic_theta(sc)
    mean_log_ST = math.log(sc.S0) + (sc.r + sc.sigma * theta0 - 0.5 * sc.sigma**2) * sc.T
    assert mean_log_ST == pytest.approx(math.log(sc.B))
    with pytest.raises(ValueError):
        heuristic_theta(SCENARIOS["asian_arith"])


def test_theta_zero_reproduces_plain_mc():
    sc = SCENARIOS["paper_barrier"]
    ss = make_seed_seq("test_is", 0)
    a = importance_mc(sc, 2048, N_STEPS, ss, option="call", theta=0.0)
    b = plain_mc(sc, 2048, N_STEPS, make_seed_seq("test_is", 0), option="call")
    assert a.price == pytest.approx(b.price, rel=1e-12)
    assert a.extra["ESS_frac"] == pytest.approx(1.0)


@pytest.mark.parametrize("name, ref", [("deep_barrier", DEEP_REF),
                                       ("paper_barrier", PAPER_REF)])
def test_importance_unbiased_and_reduces_variance(name, ref):
    sc = SCENARIOS[name]
    n = 32_768
    res = importance_mc(sc, n, N_STEPS, make_seed_seq("test_is", 1), option="call")
    plain = plain_mc(sc, n, N_STEPS, make_seed_seq("test_is", 2), option="call")
    assert abs(res.price - ref) < 3 * res.std_error
    assert res.std_error < plain.std_error
    for key in ("theta", "theta0", "ESS", "ESS_frac", "max_weight", "hit_rate_proposal"):
        assert key in res.extra
    assert 0 < res.extra["ESS"] <= n
    assert res.extra["total_paths_simulated"] == n + res.extra["pilot_paths"]


def test_pilot_grid_prefers_a_shift_on_the_deep_barrier():
    sc = SCENARIOS["deep_barrier"]
    payoff = payoff_for(sc, "call")
    theta, theta0, table = choose_theta(sc, N_STEPS, np.random.default_rng(3), payoff,
                                        pilot_paths=4096)
    assert len(table) == 7 and table[0][0] == 0.0
    assert theta > 0
    moments = dict(table)
    assert moments[theta] < moments[0.0]


def test_likelihood_ratio_uses_the_same_normals():
    """w must be exp(-theta*W~_T - theta^2 T/2) with W~_T built from the path's own normals."""
    sc = SCENARIOS["deep_barrier"]
    payoff = payoff_for(sc, "call")
    Y, w, w_tilde_T = weighted_samples(sc, 500, 52, np.random.default_rng(5), 1.2, payoff)
    Z = np.random.default_rng(5).standard_normal((500, 52))
    np.testing.assert_allclose(w_tilde_T, math.sqrt(sc.T / 52) * Z.sum(axis=1))
    np.testing.assert_allclose(w, np.exp(-1.2 * w_tilde_T - 0.5 * 1.2**2 * sc.T))


def test_effective_sample_size():
    assert effective_sample_size(np.ones(10)) == pytest.approx(10.0)
    assert effective_sample_size(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_registered():
    assert ESTIMATORS["importance"] is importance_mc
