"""
Frozen interface (WORKPLAN §1.4): randomness is ALWAYS passed in as a pre-built
``normals`` array of shape (n_paths, n_steps), column j == time step j. This lets
antithetic (pairs [Z, -Z]), RQMC (Sobol + Brownian bridge) and importance
sampling (shifted drift) all reuse the same engine.



(The base paper's "Euler" and "Euler–Maruyama" are the same scheme — trap #7.)
"""
# GBM path generation COde
# antithetic (pairs [Z, -Z]), RQMC (Sobol + Brownian bridge) and importance
# sampling (shifted drift) all **REUSE** the same engine.


# Schemes for dS = mu*S dt + sigma*S dW, with dt = T/n_steps and dW = sqrt(dt)*Z:

#   exact           S_{j+1} = S_j * exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z_j)
#   euler_maruyama  S_{j+1} = S_j * (1 + mu dt + sigma sqrt(dt) Z_j)
#   milstein        S_{j+1} = S_j * (1 + mu dt + sigma sqrt(dt) Z_j
                                    #  + 0.5 sigma^2 dt (Z_j^2 - 1))



import numpy as np

SCHEMES = ("exact", "euler_maruyama", "milstein")


def generate_paths(
    scenario,
    n_paths: int,
    n_steps: int,
    normals: np.ndarray,
    scheme: str = "exact",
    drift_override: float | None = None,
) -> np.ndarray:
    """Simulate GBM paths on a uniform grid of ``n_steps`` steps over [0, T].

    Parameters
    ----------
    scenario : config.Scenario
        Supplies S0, r, sigma, T.
    n_paths, n_steps : int
        Output grid size.
    normals : np.ndarray, shape (n_paths, n_steps)
        Standard normal increments; column j drives step j -> j+1.
    scheme : {"exact", "euler_maruyama", "milstein"}
    drift_override : float or None
        The simulation drift mu that REPLACES r (default: mu = r, risk-neutral).
        For importance sampling with W~ = W + theta*t, pass ``r + sigma*theta``.

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps + 1), with paths[:, 0] == S0.
    """
    normals = np.asarray(normals, dtype=float)
    if normals.shape != (n_paths, n_steps):
        raise ValueError(
            f"normals must have shape {(n_paths, n_steps)}, got {normals.shape}"
        )
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}; expected one of {SCHEMES}")

    S0, sigma, T = scenario.S0, scenario.sigma, scenario.T
    mu = scenario.r if drift_override is None else drift_override
    dt = T / n_steps
    sq = sigma * np.sqrt(dt)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0

    if scheme == "exact":
        log_incr = (mu - 0.5 * sigma**2) * dt + sq * normals
        np.cumsum(log_incr, axis=1, out=paths[:, 1:])
        np.exp(paths[:, 1:], out=paths[:, 1:])
        paths[:, 1:] *= S0
    else:
        factor = 1.0 + mu * dt + sq * normals
        if scheme == "milstein":
            factor += 0.5 * sigma**2 * dt * (normals**2 - 1.0)
        np.cumprod(factor, axis=1, out=paths[:, 1:])
        paths[:, 1:] *= S0

    return paths
