"""
Option payoffs on simulated paths.

All payoff functions take ``paths`` of shape (n_paths, n_steps + 1) — as returned
by ``paths.generate_paths`` — and return UNDISCOUNTED payoffs of shape (n_paths,).
Estimators apply the exp(-rT) discount.
"""

import numpy as np


def _vanilla(S, K, option):
    if option == "call":
        return np.maximum(S - K, 0.0)
    if option == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"Unsupported option type: {option!r}")


def barrier_payoff(paths, K, B, kind="up_and_in", option="call"):
    """Knock-in / knock-out payoff with DISCRETE monitoring on the simulated grid.

    The barrier is checked at every grid point (including t=0), so with
    n_steps=252 this is a daily-monitored barrier. Its price is ~0.012 below the
    continuous Reiner–Rubinstein value (7.0941 vs 7.1055); see
    ``analytic.barrier_closed_form_bgk``.

    kind : "up_and_in" | "up_and_out" | "down_and_in" | "down_and_out"
    """
    if kind.startswith("up"):
        hit = paths.max(axis=1) >= B
    elif kind.startswith("down"):
        hit = paths.min(axis=1) <= B
    else:
        raise ValueError(f"Unsupported barrier kind: {kind!r}")

    vanilla = _vanilla(paths[:, -1], K, option)
    if kind.endswith("_in"):
        return np.where(hit, vanilla, 0.0)
    if kind.endswith("_out"):
        return np.where(hit, 0.0, vanilla)
    raise ValueError(f"Unsupported barrier kind: {kind!r}")


def geometric_asian_payoff(paths, K, option="call", avg_start_idx=1):
    """Geometric-average-rate payoff over ``paths[:, avg_start_idx:]``.

    avg_start_idx=1 averages all n_steps monitoring dates t_1..t_n (excludes S0),
    which matches Kemna–Vorst discrete with m = n_steps. For the last-30-day
    variant, ``Scenario.avg_start_idx = n_steps - 29``.
    """
    G = np.exp(np.log(paths[:, avg_start_idx:]).mean(axis=1))
    return _vanilla(G, K, option)


def payoff_for(scenario, option=None):
    """Return a callable ``paths -> payoffs`` for a ``config.Scenario``.

    ``option`` overrides ``scenario.option_type`` (e.g. to price the put leg of
    ``paper_asian_geo``). ``arithmetic_asian_payoff`` is looked up by name at
    call time, so Stage 3 only needs to append that function to this module.
    """
    option = option or scenario.option_type
    kind = scenario.exotic_type

    if kind == "barrier":
        return lambda p: barrier_payoff(p, scenario.K, scenario.B,
                                        kind=scenario.barrier_kind, option=option)
    if kind == "geometric_asian":
        return lambda p: geometric_asian_payoff(p, scenario.K, option=option,
                                                avg_start_idx=scenario.avg_start_idx)
    if kind == "arithmetic_asian":
        fn = globals().get("arithmetic_asian_payoff")
        if fn is None:
            raise NotImplementedError("arithmetic_asian_payoff is added in Stage 3")
        return lambda p: fn(p, scenario.K, option=option,
                            avg_start_idx=scenario.avg_start_idx)
    raise ValueError(f"Unsupported exotic_type: {kind!r}")


def arithmetic_asian_payoff(paths, K, option="call", avg_start_idx=1):
    """Arithmetic-average-rate payoff over ``paths[:, avg_start_idx:]``.

    ``avg_start_idx=1`` averages the simulated monitoring dates while excluding
    ``S0``, matching the convention used by :func:`geometric_asian_payoff`.
    """
    average = np.mean(paths[:, avg_start_idx:], axis=1)
    return _vanilla(average, K, option)
