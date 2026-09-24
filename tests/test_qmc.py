"""
Stage-4 tests: scrambled Sobol, Brownian bridge and the RQMC estimator.

Stage 4 (Zaki Rehnoom Unmona, 2105016).
"""

import math

import numpy as np
import pytest
from scipy.special import ndtri

import src.results as results_mod
from src.benchmark import ESTIMATORS, exact_price, fit_loglog_slope, run_sweep
from src.config import SCENARIOS
from src.estimators import make_seed_seq, plain_mc
from src.vr_qmc import bridge_matrix, rqmc_mc, sobol_uniforms

ASIAN = SCENARIOS["paper_asian_geo"]


# ---------------------------------------------------------------------------
# Task 0 smoke test: known integrals, before touching options
# ---------------------------------------------------------------------------

def _rmse_curve(f, dim, exact, n_grid, R, sampler):
    rmse = []
    for n in n_grid:
        est = np.array([f(sampler(n, dim, rep)).mean() for rep in range(R)])
        rmse.append(math.sqrt(np.mean((est - exact) ** 2)))
    return np.array(rmse)


def _rqmc_normals(n, dim, rep):
    rng = np.random.default_rng(make_seed_seq("test_qmc_smoke", n, rep))
    return ndtri(sobol_uniforms(n, dim, rng))


def _mc_normals(n, dim, rep):
    return np.random.default_rng(make_seed_seq("test_qmc_smoke_mc", n, rep)).standard_normal((n, dim))


SMOOTH_A = 0.5 / np.sqrt(np.arange(1, 9))
ORTHANT_RHO = 0.5
SMOKE_CASES = {
    # E[exp(a.Z)] = exp(|a|^2 / 2): smooth, 8-dimensional
    "smooth_exponential": (lambda z: np.exp(z @ SMOOTH_A), 8,
                           math.exp(0.5 * SMOOTH_A @ SMOOTH_A)),
    # P(X1 > 0, X2 > 0) for corr(X1, X2) = rho: discontinuous, boundary not axis-aligned
    "gaussian_orthant": (lambda z: ((z[:, 0] > 0) & (ORTHANT_RHO * z[:, 0]
                                    + math.sqrt(1 - ORTHANT_RHO**2) * z[:, 1] > 0)).astype(float),
                         2, 0.25 + math.asin(ORTHANT_RHO) / (2 * math.pi)),
}


@pytest.mark.parametrize("case", SMOKE_CASES)
def test_smoke_rqmc_beats_n_minus_half(case):
    f, dim, exact = SMOKE_CASES[case]
    n_grid = [2**k for k in range(6, 15, 2)]
    rq = _rmse_curve(f, dim, exact, n_grid, 32, _rqmc_normals)
    mc = _rmse_curve(f, dim, exact, n_grid, 32, _mc_normals)
    assert fit_loglog_slope(n_grid, rq)[0] < -0.65      # clearly steeper than N^-1/2
    assert fit_loglog_slope(n_grid, mc)[0] > -0.65      # the control: plain MC is not
    assert mc[-1] / rq[-1] > 5


# ---------------------------------------------------------------------------
# Sobol draw
# ---------------------------------------------------------------------------

class TestSobolUniforms:
    @pytest.mark.parametrize("n", [0, 3, 1000, 1025])
    def test_rejects_non_power_of_two(self, n):
        with pytest.raises(ValueError, match="power-of-2"):
            sobol_uniforms(n, 4, np.random.default_rng(0))

    def test_open_unit_cube_and_deterministic(self):
        a = sobol_uniforms(4096, 16, np.random.default_rng(make_seed_seq("t", 1)))
        b = sobol_uniforms(4096, 16, np.random.default_rng(make_seed_seq("t", 1)))
        assert a.shape == (4096, 16)
        assert np.array_equal(a, b)
        assert 0.0 < a.min() and a.max() < 1.0
        assert np.isfinite(ndtri(a)).all()

    def test_distinct_seeds_give_distinct_scrambles(self):
        a = sobol_uniforms(256, 4, np.random.default_rng(make_seed_seq("t", 1)))
        b = sobol_uniforms(256, 4, np.random.default_rng(make_seed_seq("t", 2)))
        assert not np.allclose(a, b)

    def test_each_coordinate_is_stratified(self):
        # A (0, m, s)-net property per coordinate: every interval [k/N, (k+1)/N)
        # holds exactly one point. Pseudo-random points would leave gaps.
        n = 1024
        U = sobol_uniforms(n, 8, np.random.default_rng(make_seed_seq("t", 3)))
        for j in range(8):
            assert np.array_equal(np.sort(np.floor(U[:, j] * n)), np.arange(n))


# ---------------------------------------------------------------------------
# Brownian bridge
# ---------------------------------------------------------------------------

class TestBridgeMatrix:
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 64, 252])
    def test_orthogonal(self, n):
        M = bridge_matrix(n)
        assert np.allclose(M @ M.T, np.eye(n))
        assert np.allclose(M.T @ M, np.eye(n))

    @pytest.mark.parametrize("n", [7, 252])
    def test_dimension_zero_sets_terminal_value_alone(self, n):
        # W_n = sum of all per-step normals = sqrt(n) * z_0, with no other dimension.
        row_sums = bridge_matrix(n).sum(axis=1)
        expected = np.zeros(n)
        expected[0] = math.sqrt(n)
        assert np.allclose(row_sums, expected)

    def test_midpoint_uses_only_first_two_dimensions(self):
        n = 252
        W = np.cumsum(bridge_matrix(n), axis=1)      # W[k, j]: weight of z_k in W_{j+1}
        mid = W[:, n // 2 - 1]
        assert np.allclose(mid[2:], 0.0)
        # Var(W_{n/2}) = n/2 in step units, split between the two dimensions
        assert mid[0] ** 2 + mid[1] ** 2 == pytest.approx(n / 2)

    def test_bridge_preserves_path_law(self):
        # Orthogonality => i.i.d. normals in, i.i.d. normals out; check the terminal moments.
        n_paths, m = 100_000, 64
        z = np.random.default_rng(11).standard_normal((n_paths, m))
        normals = z @ bridge_matrix(m)
        assert abs(normals.mean()) < 0.002
        assert normals.var() == pytest.approx(1.0, abs=0.003)


# ---------------------------------------------------------------------------
# RQMC estimator
# ---------------------------------------------------------------------------

class TestRqmcEstimator:
    def test_registered(self):
        assert "rqmc" in ESTIMATORS and "rqmc_incremental" in ESTIMATORS

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError):
            rqmc_mc(ASIAN, 1000, 16, make_seed_seq("t", 4))

    def test_unknown_construction_raises(self):
        with pytest.raises(ValueError):
            rqmc_mc(ASIAN, 256, 16, make_seed_seq("t", 5), construction="pca")

    def test_within_run_se_is_not_reported(self):
        res = ESTIMATORS["rqmc"](ASIAN, 256, 16, make_seed_seq("t", 6))
        assert math.isnan(res.std_error) and math.isnan(res.ci_low) and math.isnan(res.ci_high)
        assert res.extra["se_valid"] is False
        assert res.extra["construction"] == "bridge"
        assert res.extra["naive_iid_se"] > 0

    def test_chunking_does_not_change_the_estimate(self):
        a = rqmc_mc(ASIAN, 4096, 32, make_seed_seq("t", 7), chunk=512)
        b = rqmc_mc(ASIAN, 4096, 32, make_seed_seq("t", 7), chunk=65536)
        assert a.price == pytest.approx(b.price, rel=1e-12)

    @pytest.mark.parametrize("construction", ["bridge", "incremental"])
    @pytest.mark.parametrize("option", ["call", "put"])
    def test_unbiased_for_geometric_asian(self, construction, option):
        # Mean of R independent scrambles vs the discrete closed form, with the
        # ACROSS-scramble standard error (the only valid RQMC error bar).
        m, R = 64, 16
        est = np.array([rqmc_mc(ASIAN, 4096, m, make_seed_seq("t_unbiased", rep),
                                option=option, construction=construction).price
                        for rep in range(R)])
        se = est.std(ddof=1) / math.sqrt(R)
        assert abs(est.mean() - exact_price(ASIAN, option, m)) < 4 * se

    def test_bridge_beats_incremental_beats_plain_on_asian(self):
        m, R, n = 64, 12, 4096
        def spread(fn):
            return np.std([fn(rep).price for rep in range(R)], ddof=1)
        sd_bridge = spread(lambda r: rqmc_mc(ASIAN, n, m, make_seed_seq("t_sd_b", r)))
        sd_incr = spread(lambda r: rqmc_mc(ASIAN, n, m, make_seed_seq("t_sd_i", r),
                                           construction="incremental"))
        sd_plain = spread(lambda r: plain_mc(ASIAN, n, m, make_seed_seq("t_sd_p", r)))
        assert sd_bridge < sd_incr < sd_plain
        assert sd_plain / sd_bridge > 5


def test_run_sweep_logs_rqmc_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(results_mod, "_RESULTS_DIR", tmp_path)
    run_sweep(person="test", experiment_id="qmc_smoke", cases=[("paper_barrier", "call")],
              methods=["rqmc", "rqmc_incremental"], n_grid=[64, 128], R=3, n_steps=8,
              verbose=False)
    df = results_mod.load_all_results(tmp_path)
    assert len(df) == 2 * 2 * 3
    assert df["std_error"].isna().all()
    assert df["seed"].is_unique
