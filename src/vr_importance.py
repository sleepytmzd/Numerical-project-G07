"""Importance sampling for barrier options (Stage 5, Tamzeed Mahfuz, 2105012).

Change of measure (Girsanov).  Paths are simulated with the drift shifted from
``r`` to ``r + sigma*theta`` by passing ``drift_override`` to the frozen engine.
With proposal Brownian motion ``W~`` (the one the normals actually build), the
risk-neutral Brownian motion is ``W = W~ + theta*t``, and the likelihood ratio
telescopes to a function of the terminal value only (WORKPLAN trap #8):

    L = dQ/dP_theta = exp(-theta * W~_T - 0.5 * theta**2 * T),
    W~_T = sqrt(dt) * sum_j Z_j        (the SAME normals fed to generate_paths)

Drift selection.  The heuristic ``theta0`` puts the expected terminal
log-price on the barrier:

    log S0 + (r + sigma*theta0 - sigma^2/2) T = log B
    =>  theta0 = [log(B/S0)/T - (r - sigma^2/2)] / sigma

(The WORKPLAN formula omits the ``/sigma``: it is the log-price drift, not the
Brownian shift that ``drift_override = r + sigma*theta`` expects.)  A 7-point
grid ``theta0 * {0, 0.25, ..., 1.5}`` is then scored on ONE independent pilot
drawn under ``theta0``: because ``L`` depends only on ``W_T``, the second
moment under any other ``theta`` is a reweighting of the same pilot,

    E_theta[(L_theta Y)^2] = E_Q[L_theta Y^2] = E_theta0[L_theta0 L_theta Y^2],

so all seven candidates cost one pilot simulation.  ``theta = 0`` is on the
grid, so the search can fall back to plain MC when the shift does not help.
"""

import math
import time

import numpy as np

from src.benchmark import register
from src.estimators import DEFAULT_CHUNK, _chunks, summarize_samples
from src.paths import generate_paths
from src.payoffs import payoff_for

GRID_FACTORS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
DEFAULT_PILOT_PATHS = 10_000


def likelihood_ratio(w_tilde_T, theta, T):
    """``dQ/dP_theta`` for a drift shift ``theta``, from the proposal-measure ``W~_T``."""
    return np.exp(-theta * np.asarray(w_tilde_T, dtype=float) - 0.5 * theta**2 * T)


def heuristic_theta(scenario):
    """Brownian shift that puts ``E[log S_T]`` on the barrier."""
    if scenario.B is None:
        raise ValueError("importance sampling needs a barrier scenario (B is None)")
    S0, r, sigma, T, B = scenario.S0, scenario.r, scenario.sigma, scenario.T, scenario.B
    return (math.log(B / S0) / T - (r - 0.5 * sigma**2)) / sigma


def weighted_samples(scenario, n_paths, n_steps, rng, theta, payoff, *,
                     scheme="exact", chunk=DEFAULT_CHUNK):
    """Simulate under the ``theta``-shifted drift.

    Returns ``(Y, w, w_tilde_T)``: discounted payoffs of the shifted paths,
    their likelihood ratios (so ``Y * w`` are unbiased price samples) and the
    proposal-measure terminal Brownian values that the ratios were built from.
    """
    disc = math.exp(-scenario.r * scenario.T)
    sqdt = math.sqrt(scenario.T / n_steps)
    mu = scenario.r + scenario.sigma * theta
    Y = np.empty(n_paths)
    w_tilde_T = np.empty(n_paths)
    pos = 0
    for k in _chunks(n_paths, chunk):
        Z = rng.standard_normal((k, n_steps))
        paths = generate_paths(scenario, k, n_steps, Z, scheme, drift_override=mu)
        Y[pos:pos + k] = payoff(paths)
        w_tilde_T[pos:pos + k] = sqdt * Z.sum(axis=1)
        pos += k
    Y *= disc
    return Y, likelihood_ratio(w_tilde_T, theta, scenario.T), w_tilde_T


def choose_theta(scenario, n_steps, rng, payoff, *, pilot_paths=DEFAULT_PILOT_PATHS,
                 grid_factors=GRID_FACTORS, scheme="exact", chunk=DEFAULT_CHUNK):
    """Pick ``theta`` from ``theta0 * grid_factors`` by minimising the pilot's
    empirical second moment of ``L*Y``.  Returns ``(theta, theta0, table)``,
    where ``table`` lists ``(theta, second_moment)`` for every candidate."""
    theta0 = heuristic_theta(scenario)
    Y, w0, w_tilde_T = weighted_samples(scenario, pilot_paths, n_steps, rng, theta0,
                                        payoff, scheme=scheme, chunk=chunk)
    T = scenario.T
    W_T = w_tilde_T + theta0 * T          # risk-neutral terminal value of each pilot path
    table = []
    for f in grid_factors:
        theta = theta0 * f
        # L_theta evaluated at the Q-Brownian terminal value: exp(-theta W_T + theta^2 T/2)
        L_theta = np.exp(-theta * W_T + 0.5 * theta**2 * T)
        table.append((float(theta), float(np.mean(w0 * L_theta * Y**2))))
    theta = min(table, key=lambda row: row[1])[0]
    return theta, float(theta0), table


def effective_sample_size(w):
    """Kish ESS ``(sum w)^2 / sum w^2``."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.dot(w, w))


@register("importance")
def importance_mc(scenario, n_paths, n_steps, seed_seq, *, option=None, scheme="exact",
                  theta=None, pilot_paths=DEFAULT_PILOT_PATHS, grid_factors=GRID_FACTORS,
                  chunk=DEFAULT_CHUNK, **_):
    """Drift-shifted importance sampling with a pilot-tuned ``theta``.

    ``theta=None`` tunes on an independent pilot (its time is included in
    ``runtime_sec``, its paths recorded in ``extra``).  A fixed ``theta`` skips
    the pilot and draws from ``seed_seq`` directly, so ``theta=0`` reproduces
    ``plain_mc`` exactly.
    """
    if n_paths < 2:
        raise ValueError("importance_mc needs at least two paths")
    t0 = time.perf_counter()
    option = option or scenario.option_type
    payoff = payoff_for(scenario, option)

    if theta is None:
        pilot_seed, production_seed = seed_seq.spawn(2)
        theta, theta0, grid = choose_theta(
            scenario, n_steps, np.random.default_rng(pilot_seed), payoff,
            pilot_paths=pilot_paths, grid_factors=grid_factors, scheme=scheme, chunk=chunk)
        rng = np.random.default_rng(production_seed)
        used_pilot = int(pilot_paths)
    else:
        theta0, grid, used_pilot = heuristic_theta(scenario), [], 0
        rng = np.random.default_rng(seed_seq)

    Y, w, _ = weighted_samples(scenario, n_paths, n_steps, rng, float(theta), payoff,
                               scheme=scheme, chunk=chunk)
    samples = Y * w
    ess = effective_sample_size(w)
    extra = {
        "theta": float(theta),
        "theta0": float(theta0),
        "ESS": ess,
        "ESS_frac": ess / n_paths,
        "max_weight": float(w.max()),
        "max_weight_share": float(w.max() / w.sum()),
        "mean_weight": float(w.mean()),
        "hit_rate_proposal": float(np.mean(Y > 0)),
        "pilot_paths": used_pilot,
        "total_paths_simulated": n_paths + used_pilot,
        "theta_grid": grid,
    }
    return summarize_samples(samples, time.perf_counter() - t0, n_paths, extra)
