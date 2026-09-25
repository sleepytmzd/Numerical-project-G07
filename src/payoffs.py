# Option payoffs on simulated paths.
import numpy as np


def _vanilla(S, K, option):
    if option == "call":
        return np.maximum(S - K, 0.0)
    if option == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"Unsupported option type: {option!r}")


def barrier_payoff(paths, K, B, kind="up_and_in", option="call"):
    # Knock-in / knock-out payoff with DISCRETE monitoring on the simulated grid.

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
    # Geometric-average-rate payoff over ``paths[:, avg_start_idx:]``.

    G = np.exp(np.log(paths[:, avg_start_idx:]).mean(axis=1))
    return _vanilla(G, K, option)


def payoff_for(scenario, option=None):
    # Return a callable ``paths -> payoffs`` for a ``config.Scenario``.

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
    # arithmetic-average-rate payoff over ``paths[:, avg_start_idx:]``.
    average = np.mean(paths[:, avg_start_idx:], axis=1)
    return _vanilla(average, K, option)
