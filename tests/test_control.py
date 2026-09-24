"""Stage-3 tests for arithmetic payoffs and control variates."""

import math

import numpy as np
import pytest

from src.config import SCENARIOS
from src.estimators import make_seed_seq
from src.payoffs import arithmetic_asian_payoff
from src.vr_control import control_variate_mc, optimal_coefficient


def test_arithmetic_asian_payoff_call_and_put():
    paths = np.array([
        [100.0, 90.0, 110.0, 130.0],
        [100.0, 80.0, 90.0, 100.0],
    ])
    np.testing.assert_allclose(
        arithmetic_asian_payoff(paths, 100.0, "call"), [10.0, 0.0]
    )
    np.testing.assert_allclose(
        arithmetic_asian_payoff(paths, 100.0, "put", avg_start_idx=2), [0.0, 5.0]
    )


def test_synthetic_correlated_normals_recover_known_b_and_vrf():
    """WORKPLAN Stage-3 task 0: coefficient and achievable VRF smoke test."""
    rng = np.random.default_rng(402)
    n = 250_000
    true_b = 1.7
    noise_sd = 0.55
    control = rng.standard_normal(n)
    target = true_b * control + noise_sd * rng.standard_normal(n)

    b_hat = optimal_coefficient(target, control)
    rho = np.corrcoef(target, control)[0, 1]
    adjusted = target - b_hat * (control - control.mean())
    observed_vrf = target.var(ddof=1) / adjusted.var(ddof=1)
    known_vrf = 1.0 + true_b**2 / noise_sd**2

    assert b_hat == pytest.approx(true_b, abs=0.005)
    assert 1.0 / (1.0 - rho**2) == pytest.approx(known_vrf, rel=0.015)
    assert observed_vrf == pytest.approx(known_vrf, rel=0.015)


def test_arithmetic_asian_control_reduces_standard_error():
    result = control_variate_mc(
        SCENARIOS["asian_arith"],
        n_paths=8192,
        n_steps=64,
        seed_seq=make_seed_seq("test_control_asian"),
        pilot_paths=2048,
    )
    assert math.isfinite(result.price)
    assert result.extra["rho"] > 0.99
    assert result.extra["observed_vrf"] > 50.0
    assert result.std_error < result.extra["plain_std_error"]
    assert result.extra["total_paths_simulated"] == 10_240


def test_barrier_control_estimate_is_consistent_with_continuous_price():
    scenario = SCENARIOS["paper_barrier"]
    result = control_variate_mc(
        scenario,
        n_paths=32_768,
        n_steps=252,
        seed_seq=make_seed_seq("test_control_barrier"),
        pilot_paths=4096,
    )
    # The simulation is discretely monitored; allow for that small bias when
    # checking consistency with the continuous closed form.
    from src.analytic import barrier_closed_form

    continuous = barrier_closed_form(
        scenario.S0,
        scenario.K,
        scenario.B,
        scenario.r,
        scenario.sigma,
        scenario.T,
        scenario.barrier_kind,
        "call",
    )
    assert abs(result.price - continuous) < 0.03 + 4.0 * result.std_error
    assert result.extra["observed_vrf"] > 20.0
