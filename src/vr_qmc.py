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
    m = int(n_paths).bit_length() - 1
    if n_paths < 1 or (1 << m) != n_paths:
        raise ValueError(f"RQMC needs a power-of-2 sample size, got n_paths={n_paths}")
    sampler = qmc.Sobol(d=dim, scramble=True, seed=rng)
    U = sampler.random_base2(m)
    U += 0.5 ** (sampler.bits + 1)
    return U


@lru_cache(maxsize=None)
def bridge_matrix(n_steps: int) -> np.ndarray:
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
    kwargs.pop("construction", None)
    return rqmc_mc(scenario, n_paths, n_steps, seed_seq, construction="bridge", **kwargs)


@register("rqmc_incremental")
def rqmc_incremental_mc(scenario, n_paths, n_steps, seed_seq, **kwargs):
    kwargs.pop("construction", None)
    return rqmc_mc(scenario, n_paths, n_steps, seed_seq, construction="incremental", **kwargs)
