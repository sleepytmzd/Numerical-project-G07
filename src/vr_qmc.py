"""
Randomized quasi-Monte Carlo: scrambled Sobol + Brownian bridge.

Stage 4 (Zaki Rehnoom Unmona, 2105016).

Each path consumes one point of a d = n_steps dimensional scrambled Sobol
sequence. Uniforms become normals by the inverse CDF (``ndtri``) — never
Box–Muller, which breaks the low-discrepancy structure (trap #3) — and are
drawn with ``random_base2`` only, so N is always a power of 2 (trap #2).

Two path constructions map Sobol dimension k to the per-step ``normals`` that
``paths.generate_paths`` consumes:

    incremental  dimension j drives time step j (the identity map)
    bridge       dimension 0 fixes W_T, dimension 1 the midpoint W_{T/2},
                 then successive midpoints breadth-first (Brownian bridge), so
                 the best-distributed leading dimensions carry the most variance

Error bars: a single RQMC estimate has NO valid within-run standard error (the
points are not independent), so ``std_error`` and the CI are logged as NaN and
the naive i.i.d. formula is kept in ``extra["naive_iid_se"]`` for reference
only. Variance must be measured across the R independent scrambles.
"""

import math
import time
from collections import deque
from functools import lru_cache

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from src.benchmark import register
from src.estimators import DEFAULT_CHUNK, EstimateResult
from src.paths import generate_paths
from src.payoffs import payoff_for

CONSTRUCTIONS = ("bridge", "incremental")


def sobol_uniforms(n_paths: int, dim: int, rng) -> np.ndarray:
    """``(n_paths, dim)`` scrambled Sobol points strictly inside (0, 1)^dim.

    ``n_paths`` must be a power of 2. scipy's points are integer multiples of
    2^-bits, and after the random digital shift an exact 0 (``ndtri(0) = -inf``)
    has probability 2^-bits per coordinate — likely somewhere in a full sweep —
    so every point is moved to the centre of its dyadic cell.
    """
    m = int(n_paths).bit_length() - 1
    if n_paths < 1 or (1 << m) != n_paths:
        raise ValueError(f"RQMC needs a power-of-2 sample size, got n_paths={n_paths}")
    sampler = qmc.Sobol(d=dim, scramble=True, seed=rng)
    U = sampler.random_base2(m)
    U += 0.5 ** (sampler.bits + 1)
    return U


@lru_cache(maxsize=None)
def bridge_matrix(n_steps: int) -> np.ndarray:
    """Orthogonal ``(n_steps, n_steps)`` Brownian-bridge map: ``normals = z @ M``.

    Row k of M says how Sobol dimension k spreads across the per-step normals.
    Time is measured in steps, so W_j = normals[:, :j].sum() has unit variance
    per step and M does not depend on T. With i.i.d. N(0,1) input z the output
    is again i.i.d. N(0,1) (M is orthogonal), so the path law is unchanged —
    only which input dimension drives which feature of the path.
    """
    n = int(n_steps)
    W = np.zeros((n + 1, n))            # W[j] = coefficients of W_j in terms of z
    W[n, 0] = math.sqrt(n)              # dimension 0 -> terminal value
    k = 1
    queue = deque([(0, n)])
    while queue:                        # breadth-first: coarse levels first
        left, right = queue.popleft()
        if right - left < 2:
            continue
        mid = (left + right) // 2
        W[mid] = ((right - mid) * W[left] + (mid - left) * W[right]) / (right - left)
        W[mid, k] = math.sqrt((mid - left) * (right - mid) / (right - left))
        k += 1
        queue.extend([(left, mid), (mid, right)])
    M = np.ascontiguousarray(np.diff(W, axis=0).T)
    M.flags.writeable = False           # shared through the cache
    return M


def rqmc_mc(scenario, n_paths, n_steps, seed_seq, *, option=None, scheme="exact",
            construction="bridge", chunk=DEFAULT_CHUNK, **_):
    """One randomized-QMC estimate from a single independent Sobol scramble.

    ``seed_seq`` seeds the scramble, so R replicates = R independent scrambles,
    which is the only valid source of RQMC error bars.
    """
    if construction not in CONSTRUCTIONS:
        raise ValueError(f"Unknown construction {construction!r}; expected one of {CONSTRUCTIONS}")
    t0 = time.perf_counter()
    U = sobol_uniforms(n_paths, n_steps, np.random.default_rng(seed_seq))
    M = bridge_matrix(n_steps) if construction == "bridge" else None
    payoff = payoff_for(scenario, option)
    disc = math.exp(-scenario.r * scenario.T)

    Y = np.empty(n_paths)
    for pos in range(0, n_paths, chunk):
        Z = ndtri(U[pos:pos + chunk])
        if M is not None:
            Z = Z @ M
        Y[pos:pos + len(Z)] = payoff(generate_paths(scenario, len(Z), n_steps, Z, scheme))
    Y *= disc

    price = float(Y.mean())
    naive_se = float(Y.std(ddof=1) / math.sqrt(n_paths)) if n_paths > 1 else float("nan")
    extra = {"construction": construction, "sobol_dim": n_steps,
             "naive_iid_se": naive_se, "se_valid": False}
    nan = float("nan")
    return EstimateResult(price, nan, nan, nan, time.perf_counter() - t0, n_paths, extra)


@register("rqmc")
def rqmc_bridge_mc(scenario, n_paths, n_steps, seed_seq, **kwargs):
    """RQMC with Brownian-bridge construction (the project's headline RQMC)."""
    kwargs.pop("construction", None)
    return rqmc_mc(scenario, n_paths, n_steps, seed_seq, construction="bridge", **kwargs)


@register("rqmc_incremental")
def rqmc_incremental_mc(scenario, n_paths, n_steps, seed_seq, **kwargs):
    """RQMC with incremental construction (Sobol dimension j -> time step j).

    Registered under its own name because ``benchmark.summarize`` groups by
    method, not ``method_params`` — a params switch would pool the two.
    """
    kwargs.pop("construction", None)
    return rqmc_mc(scenario, n_paths, n_steps, seed_seq, construction="incremental", **kwargs)
