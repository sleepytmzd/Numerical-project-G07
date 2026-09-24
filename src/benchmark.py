"""
Generic benchmark runner and analysis helpers.

Stage 2 (Nafis Nahian, 2105007).

``ESTIMATORS`` is a registry: Stages 3–5 register their own estimator under a
new name and ``run_sweep`` picks it up automatically — nothing here is
hard-coded to plain/antithetic.
"""

import math
import time
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.analytic import barrier_closed_form, barrier_closed_form_bgk
from src.config import N_GRID, N_STEPS, R, SCENARIOS
from src.estimators import antithetic_mc, make_seed_seq, plain_mc, seed_int
from src.results import log_result

# ---------------------------------------------------------------------------
# Estimator registry
# ---------------------------------------------------------------------------

ESTIMATORS = {}


def register(name):
    """Decorator: ``@register("control_variate")`` adds an estimator to the sweep."""
    def deco(fn):
        ESTIMATORS[name] = fn
        return fn
    return deco


register("plain")(plain_mc)
register("antithetic")(antithetic_mc)


# ---------------------------------------------------------------------------
# Analytic benchmarks
# ---------------------------------------------------------------------------

def _discrete_geometric_asian_price(scenario, option, n_steps):
    """General discrete geometric-Asian price over the ACTUAL averaging window
    ``scenario.avg_start_idx .. n_steps`` (not just full-window m=252).

    log G = log S0 + (r - sigma^2/2)*mean(t_i) + sigma*mean(W_{t_i}), so G is
    lognormal with mean_logG / var_logG below; at full averaging this reduces
    exactly to Kemna–Vorst discrete (checked in tests/test_paths.py).
    """
    S0, r, sigma, T = scenario.S0, scenario.r, scenario.sigma, scenario.T
    dt = T / n_steps
    idx = np.arange(scenario.avg_start_idx, n_steps + 1)
    t = idx * dt
    n = len(t)

    mean_t = t.mean()
    cov = np.minimum.outer(t, t)
    var_logG = sigma**2 * cov.sum() / n**2
    mean_logG = math.log(S0) + (r - 0.5 * sigma**2) * mean_t

    sq = math.sqrt(var_logG)
    K = scenario.K
    d1 = (mean_logG - math.log(K) + var_logG) / sq
    d2 = d1 - sq
    EG = math.exp(mean_logG + 0.5 * var_logG)
    disc = math.exp(-r * T)

    if option == "call":
        return disc * (EG * norm.cdf(d1) - K * norm.cdf(d2))
    if option == "put":
        return disc * (K * norm.cdf(-d2) - EG * norm.cdf(-d1))
    raise ValueError(f"Unsupported option type: {option!r}")


def exact_price(scenario, option=None, n_steps=N_STEPS):
    """Analytic benchmark price, or ``None`` if none exists (arithmetic Asian)."""
    option = option or scenario.option_type
    kind = scenario.exotic_type

    if kind == "barrier":
        return barrier_closed_form(scenario.S0, scenario.K, scenario.B, scenario.r,
                                   scenario.sigma, scenario.T,
                                   kind=scenario.barrier_kind, option=option)
    if kind == "geometric_asian":
        return _discrete_geometric_asian_price(scenario, option, n_steps)
    if kind == "arithmetic_asian":
        return None
    raise ValueError(f"Unsupported exotic_type: {kind!r}")


def exact_price_bgk(scenario, option=None, n_steps=N_STEPS):
    """BGK continuity-corrected barrier benchmark (only meaningful for barriers)."""
    option = option or scenario.option_type
    if scenario.exotic_type != "barrier":
        raise ValueError("BGK correction only applies to barrier options")
    return barrier_closed_form_bgk(scenario.S0, scenario.K, scenario.B, scenario.r,
                                   scenario.sigma, scenario.T, n_steps,
                                   kind=scenario.barrier_kind, option=option)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(person, experiment_id, cases, methods, *, n_grid=N_GRID, R=R,
              n_steps=N_STEPS, schemes=("exact",), method_params=None, verbose=True):
    """Sweep (case x scheme x method x N x replicate), logging one row each.

    cases : list of (scenario_name, option) — option=None uses scenario.option_type
    methods : list of estimator names registered in ESTIMATORS
    method_params : {method_name: {param: value}}, extra kwargs per estimator
    """
    method_params = method_params or {}
    t_start = time.perf_counter()
    n_cells = len(cases) * len(schemes) * len(methods) * len(n_grid)
    cell = 0

    for scen_name, option in cases:
        scenario = SCENARIOS[scen_name]
        opt = option or scenario.option_type
        exact = exact_price(scenario, opt, n_steps)

        for scheme in schemes:
            for method in methods:
                fn = ESTIMATORS[method]
                params = method_params.get(method, {})
                cell += 1
                if verbose:
                    print(f"[{cell}/{n_cells}] {scen_name}/{opt} {scheme} {method}")
                for n_paths in n_grid:
                    for rep in range(R):
                        # Keyed by names via crc32 (hash() is salted per process;
                        # list positions collide across separate run_sweep calls).
                        # Scheme is deliberately NOT in the key: all schemes see the
                        # same normals, so scheme comparisons are paired (CRN).
                        ss = make_seed_seq(experiment_id,
                                          zlib.crc32(scen_name.encode()),
                                          zlib.crc32(opt.encode()),
                                          zlib.crc32(method.encode()), n_paths, rep)
                        result = fn(scenario, n_paths, n_steps, ss, option=opt,
                                   scheme=scheme, **params)
                        log_result(
                            person=person, experiment_id=experiment_id,
                            scenario=scen_name, option_type=opt, scheme=scheme,
                            method=method, method_params=str(params),
                            n_paths=n_paths, n_steps=n_steps, replicate_id=rep,
                            seed=seed_int(ss), price=result.price,
                            std_error=result.std_error, ci_low=result.ci_low,
                            ci_high=result.ci_high, exact_price=exact,
                            runtime_sec=result.runtime_sec, extra=result.extra,
                        )
    if verbose:
        print(f"run_sweep done in {time.perf_counter() - t_start:.1f}s")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

GROUP_COLS = ["scenario", "option_type", "scheme", "method", "n_paths"]


def summarize(df):
    """Per-(scenario, option, scheme, method, n_paths) cell summary.

    Variance and bias are measured ACROSS replicates (never from a single
    estimator's own std_error), per WORKPLAN §1.7.
    """
    rows = []
    for keys, g in df.groupby(GROUP_COLS):
        exact = g["exact_price"].dropna()
        exact_val = float(exact.iloc[0]) if len(exact) else float("nan")
        mean_price = g["price"].mean()
        var_across = g["price"].var(ddof=1) if len(g) > 1 else float("nan")
        bias = mean_price - exact_val if not math.isnan(exact_val) else float("nan")
        rmse = math.sqrt(bias**2 + var_across) if not math.isnan(bias) else float("nan")
        mean_time = g["runtime_sec"].mean()
        efficiency = 1.0 / (var_across * mean_time) if var_across and mean_time else float("nan")
        coverage = ((g["ci_low"] <= exact_val) & (exact_val <= g["ci_high"])).mean() \
            if not math.isnan(exact_val) else float("nan")

        rows.append(dict(zip(GROUP_COLS, keys), **{
            "n_reps": len(g), "mean_price": mean_price, "exact_price": exact_val,
            "bias": bias, "var_across_reps": var_across, "rmse": rmse,
            "mean_std_error": g["std_error"].mean(), "mean_runtime_sec": mean_time,
            "efficiency": efficiency, "ci_coverage": coverage,
        }))
    return pd.DataFrame(rows)


def bootstrap_efficiency_ratio(df, group_keys_a, group_keys_b, group_cols=GROUP_COLS,
                               n_boot=2000, seed=402):
    """Bootstrap CI for efficiency(method_a) / efficiency(method_b) on matched cells.

    group_keys_a / group_keys_b select rows via ``df[group_cols] == keys`` (dict).
    Resamples replicates (with replacement) independently for each group.
    Returns (ratio, ci_low, ci_high).
    """
    def _select(keys):
        mask = np.ones(len(df), dtype=bool)
        for k, v in keys.items():
            mask &= (df[k] == v)
        return df.loc[mask]

    a = _select(group_keys_a)
    b = _select(group_keys_b)
    # Pooling replicates across different N (or scenarios) mixes variance scales
    # and makes the ratio meaningless — require each side to be ONE cell.
    for name, sel in (("a", a), ("b", b)):
        if sel.empty:
            raise ValueError(f"group_keys_{name} selects no rows")
        mixed = [c for c in group_cols if sel[c].nunique() > 1]
        if mixed:
            raise ValueError(f"group_keys_{name} spans several cells (varying {mixed}); "
                             "include those columns in the keys")
    rng = np.random.default_rng(seed)

    def _eff(sample_prices, sample_times):
        var = np.var(sample_prices, ddof=1)
        t = np.mean(sample_times)
        return 1.0 / (var * t) if var > 0 and t > 0 else np.nan

    pa, ta = a["price"].to_numpy(), a["runtime_sec"].to_numpy()
    pb, tb = b["price"].to_numpy(), b["runtime_sec"].to_numpy()
    point = _eff(pa, ta) / _eff(pb, tb)

    ratios = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, len(pa), len(pa))
        ib = rng.integers(0, len(pb), len(pb))
        ratios[i] = _eff(pa[ia], ta[ia]) / _eff(pb[ib], tb[ib])
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def fit_loglog_slope(n, y):
    """Least-squares slope of log(y) vs log(n). Returns (slope, intercept)."""
    n, y = np.asarray(n, dtype=float), np.asarray(y, dtype=float)
    mask = (n > 0) & (y > 0)
    slope, intercept = np.polyfit(np.log(n[mask]), np.log(y[mask]), 1)
    return float(slope), float(intercept)
