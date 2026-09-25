# Engine sanity tests — GBM path generation and payoffs.

import math

import numpy as np
import pytest

from src.analytic import geometric_asian_closed_form
from src.benchmark import _discrete_geometric_asian_price, exact_price
from src.config import N_STEPS, SCENARIOS
from src.paths import generate_paths
from src.payoffs import barrier_payoff, geometric_asian_payoff, payoff_for

SCENARIO = SCENARIOS["paper_barrier"]
ASIAN = SCENARIOS["paper_asian_geo"]


class TestGeneratePaths:
    def test_shape_and_initial_value(self):
        n, m = 1000, 50
        Z = np.random.default_rng(0).standard_normal((n, m))
        paths = generate_paths(SCENARIO, n, m, Z, scheme="exact")
        assert paths.shape == (n, m + 1)
        assert np.allclose(paths[:, 0], SCENARIO.S0)

    def test_bad_normals_shape_raises(self):
        Z = np.zeros((10, 5))
        with pytest.raises(ValueError):
            generate_paths(SCENARIO, 10, 10, Z, scheme="exact")

    def test_unknown_scheme_raises(self):
        Z = np.zeros((10, 5))
        with pytest.raises(ValueError):
            generate_paths(SCENARIO, 10, 5, Z, scheme="bogus")

    def test_terminal_mean_and_log_variance(self):
        n, m = 200_000, N_STEPS
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((n, m))
        paths = generate_paths(SCENARIO, n, m, Z, scheme="exact")
        ST = paths[:, -1]

        expected_mean = SCENARIO.S0 * math.exp(SCENARIO.r * SCENARIO.T)
        se_mean = ST.std(ddof=1) / math.sqrt(n)
        assert abs(ST.mean() - expected_mean) < 4 * se_mean

        log_var = np.log(ST / SCENARIO.S0).var(ddof=1)
        expected_var = SCENARIO.sigma**2 * SCENARIO.T
        # variance of a sample-variance estimator ~ 2*var^2/(n-1)
        se_var = expected_var * math.sqrt(2 / (n - 1))
        assert abs(log_var - expected_var) < 4 * se_var

    def test_drift_override_shifts_mean(self):
        n, m = 100_000, 50
        rng = np.random.default_rng(2)
        Z = rng.standard_normal((n, m))
        mu = 0.10
        paths = generate_paths(SCENARIO, n, m, Z, scheme="exact", drift_override=mu)
        ST = paths[:, -1]
        expected = SCENARIO.S0 * math.exp(mu * SCENARIO.T)
        se = ST.std(ddof=1) / math.sqrt(n)
        assert abs(ST.mean() - expected) < 4 * se

    @pytest.mark.parametrize("scheme", ["euler_maruyama", "milstein"])
    def test_schemes_agree_with_exact_at_fixed_seed(self, scheme):
        # Same normals => compare with a PAIRED standard error.
        n, m = 50_000, 252
        Z = np.random.default_rng(3).standard_normal((n, m))
        diff = (generate_paths(SCENARIO, n, m, Z, scheme=scheme)[:, -1]
                - generate_paths(SCENARIO, n, m, Z, scheme="exact")[:, -1])
        se = diff.std(ddof=1) / math.sqrt(n)
        assert abs(diff.mean()) < 4 * se + 0.01   # 0.01: O(dt) weak bias allowance

    def test_strong_error_milstein_beats_euler(self):
        # Pathwise (strong) error vs exact: Euler O(sqrt dt), Milstein O(dt).
        # A wrong-signed Milstein correction would be WORSE than Euler here.
        n, m = 20_000, 252
        Z = np.random.default_rng(7).standard_normal((n, m))
        exact = generate_paths(SCENARIO, n, m, Z, scheme="exact")[:, -1]
        err_em = np.abs(generate_paths(SCENARIO, n, m, Z, "euler_maruyama")[:, -1] - exact).mean()
        err_mil = np.abs(generate_paths(SCENARIO, n, m, Z, "milstein")[:, -1] - exact).mean()
        assert err_mil < 0.05 * err_em

    def test_strong_order_euler_half(self):
        # Halving dt 4x should roughly halve Euler's strong error (order 0.5).
        n = 20_000
        rng = np.random.default_rng(8)
        Zf = rng.standard_normal((n, 1024))
        errs = []
        for m in (64, 256, 1024):
            k = 1024 // m
            Z = Zf.reshape(n, m, k).sum(axis=2) / math.sqrt(k)   # same Brownian path
            exact = generate_paths(SCENARIO, n, m, Z, "exact")[:, -1]
            errs.append(np.abs(generate_paths(SCENARIO, n, m, Z, "euler_maruyama")[:, -1]
                               - exact).mean())
        slope = np.polyfit(np.log([1 / 64, 1 / 256, 1 / 1024]), np.log(errs), 1)[0]
        assert 0.4 < slope < 0.6


class TestPayoffs:
    def test_barrier_hit_gives_vanilla_payoff(self):
        # Path that touches the barrier, ends ITM.
        paths = np.array([[100.0, 108.0, 112.0, 106.0]])
        payoff = barrier_payoff(paths, K=105.0, B=110.0, kind="up_and_in", option="call")
        assert payoff[0] == pytest.approx(1.0)

    def test_barrier_no_hit_gives_zero_for_in(self):
        paths = np.array([[100.0, 102.0, 104.0, 106.0]])
        payoff = barrier_payoff(paths, K=105.0, B=110.0, kind="up_and_in", option="call")
        assert payoff[0] == pytest.approx(0.0)

    def test_up_in_up_out_complementarity(self):
        rng = np.random.default_rng(4)
        n, m = 5000, 20
        Z = rng.standard_normal((n, m))
        paths = generate_paths(SCENARIO, n, m, Z, scheme="exact")
        vanilla = np.maximum(paths[:, -1] - SCENARIO.K, 0.0)
        p_in = barrier_payoff(paths, SCENARIO.K, SCENARIO.B, "up_and_in", "call")
        p_out = barrier_payoff(paths, SCENARIO.K, SCENARIO.B, "up_and_out", "call")
        assert np.allclose(p_in + p_out, vanilla)

    def test_geometric_asian_matches_kv_discrete_full_average(self):
        rng = np.random.default_rng(5)
        n, m = 300_000, N_STEPS
        Z = rng.standard_normal((n, m))
        paths = generate_paths(ASIAN, n, m, Z, scheme="exact")
        payoff = geometric_asian_payoff(paths, ASIAN.K, option="call", avg_start_idx=1)
        disc = math.exp(-ASIAN.r * ASIAN.T)
        mc_price = disc * payoff.mean()
        se = disc * payoff.std(ddof=1) / math.sqrt(n)

        exact = geometric_asian_closed_form(ASIAN.S0, ASIAN.K, ASIAN.r, ASIAN.sigma,
                                            ASIAN.T, option="call", m=m)
        assert abs(mc_price - exact) < 3 * se

    def test_30day_window_matches_closed_form(self):
        scen = SCENARIOS["asian_30d"]
        assert N_STEPS - scen.avg_start_idx + 1 == 30
        n, m = 200_000, N_STEPS
        Z = np.random.default_rng(6).standard_normal((n, m))
        payoff = payoff_for(scen)(generate_paths(scen, n, m, Z, scheme="exact"))
        disc = math.exp(-scen.r * scen.T)
        mc, se = disc * payoff.mean(), disc * payoff.std(ddof=1) / math.sqrt(n)
        assert abs(mc - exact_price(scen, "call", m)) < 3 * se

    def test_arithmetic_asian_dispatcher(self):
        scen = SCENARIOS["asian_arith"]
        paths = np.array([
            [100.0, 110.0, 120.0],
            [100.0, 80.0, 90.0],
        ])
        np.testing.assert_allclose(payoff_for(scen)(paths), [10.0, 0.0])


class TestGeneralDiscreteGeometricFormula:
    def test_matches_kemna_vorst_at_full_averaging(self):
        m = N_STEPS
        general = _discrete_geometric_asian_price(ASIAN, "call", m)
        kv = geometric_asian_closed_form(ASIAN.S0, ASIAN.K, ASIAN.r, ASIAN.sigma,
                                         ASIAN.T, option="call", m=m)
        assert general == pytest.approx(kv, rel=1e-9)

    def test_put_matches_kemna_vorst_at_full_averaging(self):
        m = N_STEPS
        general = _discrete_geometric_asian_price(ASIAN, "put", m)
        kv = geometric_asian_closed_form(ASIAN.S0, ASIAN.K, ASIAN.r, ASIAN.sigma,
                                         ASIAN.T, option="put", m=m)
        assert general == pytest.approx(kv, rel=1e-9)

    def test_30day_window_differs_from_full_average(self):
        scen30 = SCENARIOS["asian_30d"]
        p30 = exact_price(scen30, "call", N_STEPS)
        pfull = exact_price(ASIAN, "call", N_STEPS)
        # Shorter averaging window -> less variance reduction -> price closer
        # to vanilla, i.e. higher than the full-average geometric Asian.
        assert p30 > pfull
