"""
Live demo: price two options with every method, in a few seconds.

    python demo.py                 # N = 16,384 paths per estimate
    python demo.py --n 65536       # bigger N, still under ~15 s

For each method it prints the price, its 95% error bar, the actual error against
the reference, and the run time.  Plain MC's error bar is the one to beat.

Reference prices are the 252-date (daily-monitoring) values used throughout the
report: 7.094133 for the paper barrier (64 RQMC scrambles x 65,536 paths) and
3.162981 for the arithmetic Asian, which has no closed form.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import math
import sys
import time

import numpy as np

import src.vr_control      # noqa: F401  (registers "control_variate")
import src.vr_importance   # noqa: F401  (registers "importance")
import src.vr_qmc          # noqa: F401  (registers "rqmc")
from src.benchmark import ESTIMATORS
from src.config import N_STEPS, SCENARIOS
from src.estimators import make_seed_seq

CASES = [
    ("paper_barrier", "Up-and-in barrier call, B = 110.68", 7.094133,
     ["plain", "antithetic", "control_variate", "rqmc", "importance"]),
    ("asian_arith", "Arithmetic Asian call (no closed form)", 3.162981,
     ["plain", "antithetic", "control_variate", "rqmc"]),
]
LABEL = {"plain": "Plain MC", "antithetic": "Antithetic", "control_variate": "Control variate",
         "rqmc": "RQMC + bridge", "importance": "Importance sampling"}
RQMC_SCRAMBLES = 8     # RQMC has no within-run SE: its error bar comes from independent scrambles


def estimate(name, method, n, seed):
    """(price, 95% half-width, seconds) for one method."""
    fn, sc = ESTIMATORS[method], SCENARIOS[name]
    t0 = time.perf_counter()
    if method == "rqmc":
        prices = [fn(sc, n, N_STEPS, make_seed_seq("demo", seed, k), option="call").price
                  for k in range(RQMC_SCRAMBLES)]
        # report ONE n-point scramble (same work as the other rows); the extra scrambles
        # only supply its error bar, and are left out of the time
        price = prices[0]
        half = 1.96 * float(np.std(prices, ddof=1))
        secs = (time.perf_counter() - t0) / RQMC_SCRAMBLES
    else:
        r = fn(sc, n, N_STEPS, make_seed_seq("demo", seed), option="call")
        price, half, secs = r.price, 1.96 * r.std_error, time.perf_counter() - t0
    return price, half, secs


def main():
    ap = argparse.ArgumentParser(description="Live pricing demo.")
    ap.add_argument("--n", type=int, default=16384, help="paths per estimate (power of 2)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.n & (args.n - 1):
        sys.exit("--n must be a power of 2 (Sobol needs it)")

    print(f"\n{args.n:,} paths per estimate, {N_STEPS} daily monitoring dates, "
          f"S0=100, K=105, r=3%, sigma=20%, T=1\n")
    for name, title, ref, methods in CASES:
        print(f"{title}   reference = {ref:.6f}")
        print(f"  {'method':<20}{'price':>10}{'± 95%':>11}{'error':>11}{'time':>9}   error bar vs plain")
        base = None
        for m in methods:
            price, half, secs = estimate(name, m, args.n, args.seed)
            base = base or half
            print(f"  {LABEL[m]:<20}{price:>10.5f}{half:>11.5f}{price - ref:>+11.5f}"
                  f"{secs:>8.2f}s   {base / half:>7.1f}× tighter")
        print()
    print("Error bars shrink like 1/sqrt(N) for plain MC: 100x more paths for one more digit.\n"
          "'× tighter' = how much narrower the 95% error bar is than plain MC's at the same N.")


if __name__ == "__main__":
    main()
