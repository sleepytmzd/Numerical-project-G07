"""
Monte Carlo estimators: plain MC and antithetic variates.

Stage 2 (Nafis Nahian, 2105007).

Every estimator in the project (including Stages 3–5) shares one signature so
``benchmark.run_sweep`` can treat them uniformly:

    estimator(scenario, n_paths, n_steps, seed_seq, *, option=None,
              scheme="exact", **params) -> EstimateResult

``seed_seq`` is a ``np.random.SeedSequence`` from ``make_seed_seq`` — each
estimator builds its own ``normals`` array from it (pseudo-random, Sobol, ...).
"""

import math
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from src.config import BASE_SEED
from src.paths import generate_paths
from src.payoffs import payoff_for

Z95 = 1.959963984540054
DEFAULT_CHUNK = 8192   # paths per block — keeps memory at ~16 MB per block at 252 steps


@dataclass
class EstimateResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    runtime_sec: float
    n_paths: int
    extra: dict = field(default_factory=dict)   # rho, b_hat, ESS, max_weight, ...


# ---------------------------------------------------------------------------
# Seeding (§1.5): SeedSequence(402) keyed by experiment and cell, never np.random.seed
# ---------------------------------------------------------------------------

def make_seed_seq(experiment_id: str, *keys: int) -> np.random.SeedSequence:
    """Deterministic SeedSequence keyed by ``(experiment_id, *keys)``.

    Equivalent to walking ``SeedSequence(402).spawn()`` down a fixed tree:
    the same key always yields the same stream, and distinct keys are independent.
    """
    key = (zlib.crc32(experiment_id.encode()), *(int(k) for k in keys))
    return np.random.SeedSequence(BASE_SEED, spawn_key=key)


def seed_int(seed_seq: np.random.SeedSequence) -> int:
    """A single integer fingerprint of a SeedSequence, for the CSV ``seed`` column."""
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def summarize_samples(samples: np.ndarray, runtime_sec: float, n_paths: int,
                      extra: dict | None = None) -> EstimateResult:
    """Build an EstimateResult from iid (already discounted) samples."""
    m = len(samples)
    price = float(samples.mean())
    se = float(samples.std(ddof=1) / math.sqrt(m)) if m > 1 else float("nan")
    return EstimateResult(price, se, price - Z95 * se, price + Z95 * se,
                          runtime_sec, n_paths, extra or {})


def _chunks(n, chunk):
    done = 0
    while done < n:
        k = min(chunk, n - done)
        yield k
        done += k


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def plain_mc(scenario, n_paths, n_steps, seed_seq, *, option=None, scheme="exact",
             chunk=DEFAULT_CHUNK, **_):
    """Crude Monte Carlo: mean of discounted payoffs over n_paths iid paths."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed_seq)
    payoff = payoff_for(scenario, option)
    disc = math.exp(-scenario.r * scenario.T)

    Y = np.empty(n_paths)
    pos = 0
    for k in _chunks(n_paths, chunk):
        Z = rng.standard_normal((k, n_steps))
        Y[pos:pos + k] = payoff(generate_paths(scenario, k, n_steps, Z, scheme))
        pos += k
    Y *= disc

    return summarize_samples(Y, time.perf_counter() - t0, n_paths)


def antithetic_mc(scenario, n_paths, n_steps, seed_seq, *, option=None, scheme="exact",
                  chunk=DEFAULT_CHUNK, **_):
    """Antithetic variates: n_paths TOTAL paths = n_paths/2 pairs driven by [Z, -Z].

    The pairs are the iid units, so the standard error is computed from the
    n_paths/2 pair averages (Y + Y')/2 — NOT from the n_paths correlated payoffs.
    """
    if n_paths % 2:
        raise ValueError("antithetic_mc needs an even n_paths")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed_seq)
    payoff = payoff_for(scenario, option)
    disc = math.exp(-scenario.r * scenario.T)

    n_pairs = n_paths // 2
    Y1 = np.empty(n_pairs)
    Y2 = np.empty(n_pairs)
    pos = 0
    for k in _chunks(n_pairs, max(1, chunk // 2)):
        Z = rng.standard_normal((k, n_steps))
        P = payoff(generate_paths(scenario, 2 * k, n_steps, np.vstack([Z, -Z]), scheme))
        Y1[pos:pos + k] = P[:k]
        Y2[pos:pos + k] = P[k:]
        pos += k
    Y1 *= disc
    Y2 *= disc

    pair_avg = 0.5 * (Y1 + Y2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = float(np.corrcoef(Y1, Y2)[0, 1]) if n_pairs > 1 else float("nan")
    extra = {"pairs": n_pairs, "rho_pair": rho}
    return summarize_samples(pair_avg, time.perf_counter() - t0, n_paths, extra)
