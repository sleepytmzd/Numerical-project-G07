import math
import time
from dataclasses import replace

import numpy as np

from src.analytic import bs_call, bs_put
from src.benchmark import exact_price, register
from src.estimators import DEFAULT_CHUNK, summarize_samples
from src.paths import generate_paths
from src.payoffs import geometric_asian_payoff, payoff_for


def optimal_coefficient(target_samples, control_samples):
    #Return Cov(Y, X) / Var(X) for paired one-dimensional samples.
    y = np.asarray(target_samples, dtype=float)
    x = np.asarray(control_samples, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or y.shape != x.shape:
        raise ValueError("target_samples and control_samples must be equal-length 1-D arrays")
    if len(x) < 2:
        raise ValueError("at least two pilot samples are required")
    x_centered = x - x.mean()
    variance_x = float(np.dot(x_centered, x_centered) / (len(x) - 1))
    if not np.isfinite(variance_x) or variance_x <= 0.0:
        raise ValueError("control sample variance must be positive and finite")
    covariance = float(np.dot(y - y.mean(), x_centered) / (len(x) - 1))
    return covariance / variance_x


def _control_definition(scenario, option, n_steps):
    #Return (control payoff, known discounted mean, label).
    if scenario.exotic_type == "arithmetic_asian":
        geometric_scenario = replace(scenario, exotic_type="geometric_asian")
        mean = exact_price(geometric_scenario, option, n_steps)
        payoff = lambda paths: geometric_asian_payoff(paths, scenario.K, option=option, avg_start_idx=scenario.avg_start_idx )
        return payoff, float(mean), "geometric_asian"

    if scenario.exotic_type == "barrier":
        if option == "call":
            mean = bs_call(scenario.S0, scenario.K, scenario.r, scenario.sigma, scenario.T)
            payoff = lambda paths: np.maximum(paths[:, -1] - scenario.K, 0.0)
        elif option == "put":
            mean = bs_put( scenario.S0, scenario.K, scenario.r, scenario.sigma, scenario.T )
            payoff = lambda paths: np.maximum(scenario.K - paths[:, -1], 0.0)
        else:
            raise ValueError(f"Unsupported option type: {option!r}")
        
        return payoff, float(mean), "european_terminal"

    raise ValueError("control_variate supports arithmetic_asian and barrier scenarios, "f"not {scenario.exotic_type!r}")


def _paired_samples(
    scenario,
    n_paths,
    n_steps,
    rng,
    target_payoff,
    control_payoff,
    scheme,
    chunk,
):
    #Generate paired discounted target/control samples in bounded memory.
    target = np.empty(n_paths)
    control = np.empty(n_paths)
    discount = math.exp(-scenario.r * scenario.T)
    pos = 0
    while pos < n_paths:
        size = min(chunk, n_paths - pos)
        normals = rng.standard_normal((size, n_steps))
        paths = generate_paths(scenario, size, n_steps, normals, scheme)
        target[pos:pos + size] = discount * target_payoff(paths)
        control[pos:pos + size] = discount * control_payoff(paths)
        pos += size
    return target, control


@register("control_variate")
def control_variate_mc(
    scenario,
    n_paths,
    n_steps,
    seed_seq,
    *,
    option=None,
    scheme="exact",
    pilot_paths=None,
    pilot_fraction=0.10,
    min_pilot_paths=256,
    chunk=DEFAULT_CHUNK,
    **_,
):
    #Monte Carlo with an analytic-mean control and an independent pilot fit.

    #n_paths counts production paths, consistently with the shared estimator
    #interface.  Pilot paths are additional work: they are included in runtime
    #and recorded in extra as pilot_paths and total_paths_simulated.

    if n_paths < 2:
        raise ValueError("control_variate_mc needs at least two production paths")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if pilot_paths is None:
        if pilot_fraction <= 0:
            raise ValueError("pilot_fraction must be positive")
        pilot_paths = max(min_pilot_paths, int(math.ceil(pilot_fraction * n_paths)))
    pilot_paths = int(pilot_paths)
    if pilot_paths < 2:
        raise ValueError("pilot_paths must be at least two")

    start = time.perf_counter()
    option = option or scenario.option_type
    target_payoff = payoff_for(scenario, option)
    control_payoff, control_mean, control_name = _control_definition(
        scenario, option, n_steps
    )

    pilot_seed, production_seed = seed_seq.spawn(2)
    pilot_y, pilot_x = _paired_samples(
        scenario,
        pilot_paths,
        n_steps,
        np.random.default_rng(pilot_seed),
        target_payoff,
        control_payoff,
        scheme,
        chunk,
    )
    b_hat = optimal_coefficient(pilot_y, pilot_x)
    pilot_rho = float(np.corrcoef(pilot_y, pilot_x)[0, 1])

    target, control = _paired_samples(
        scenario,
        n_paths,
        n_steps,
        np.random.default_rng(production_seed),
        target_payoff,
        control_payoff,
        scheme,
        chunk,
    )
    adjusted = target - b_hat * (control - control_mean)

    rho = float(np.corrcoef(target, control)[0, 1])
    target_variance = float(np.var(target, ddof=1))
    adjusted_variance = float(np.var(adjusted, ddof=1))
    observed_vrf = target_variance / adjusted_variance
    theoretical_vrf = 1.0 / max(1.0 - rho**2, np.finfo(float).eps)
    plain_std_error = math.sqrt(target_variance / n_paths)

    extra = {
        "control": control_name,
        "control_mean": control_mean,
        "b_hat": float(b_hat),
        "rho": rho,
        "pilot_rho": pilot_rho,
        "pilot_paths": pilot_paths,
        "total_paths_simulated": n_paths + pilot_paths,
        "plain_std_error": plain_std_error,
        "observed_vrf": observed_vrf,
        "theoretical_vrf": theoretical_vrf,
    }
    runtime = time.perf_counter() - start
    return summarize_samples(adjusted, runtime, n_paths, extra)
