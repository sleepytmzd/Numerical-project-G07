"""
Regenerate every figure and table in the report with one command.

    python run_all.py                  # full re-simulation (~25 min, single-threaded)
    python run_all.py --analyze-only   # rebuild figures/tables from results/raw/*.csv (~2 min)
    python run_all.py --tests          # run pytest first, stop if it fails
    python run_all.py --only qmc master

Scripts run in dependency order: exp_importance and exp_master read Stage 4's
252-date barrier references from ``results/raw/zaki_qmc.csv``, so exp_qmc must
come before them on a full re-simulation.  Each script runs in its own process
with ``OMP_NUM_THREADS=1`` (WORKPLAN §1.5: honest, comparable timings).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (short name, script, what it produces)
EXPERIMENTS = [
    ("baseline", "experiments/exp_baseline.py", "Stage 2: plain/antithetic baseline, paper claims"),
    ("control", "experiments/exp_control.py", "Stage 3: control variates"),
    ("barrier_bias", "experiments/exp_barrier_bias.py", "Stage 3: barrier monitoring bias vs BGK"),
    ("qmc", "experiments/exp_qmc.py", "Stage 4: RQMC vs MC, bridge vs incremental"),
    ("scheme_order", "experiments/exp_scheme_order.py", "Stage 4: strong/weak scheme order"),
    ("importance", "experiments/exp_importance.py", "Stage 5: importance sampling"),
    ("master", "experiments/exp_master.py", "Stage 5: master five-method comparison"),
]


def main():
    ap = argparse.ArgumentParser(description="Regenerate all results.")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip simulation; rebuild figures and tables from the existing CSVs.")
    ap.add_argument("--tests", action="store_true", help="Run pytest first.")
    ap.add_argument("--only", nargs="+", choices=[e[0] for e in EXPERIMENTS], metavar="NAME",
                    help="Run only these experiments: " + ", ".join(e[0] for e in EXPERIMENTS))
    args = ap.parse_args()

    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8",
           "MPLBACKEND": "Agg"}

    if args.tests:
        print("=== pytest ===", flush=True)
        if subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=ROOT, env=env).returncode:
            sys.exit("pytest failed — not regenerating results.")

    selected = [e for e in EXPERIMENTS if not args.only or e[0] in args.only]
    timings = []
    t_all = time.perf_counter()
    for name, script, what in selected:
        print(f"\n=== {name}: {script} — {what} ===", flush=True)
        cmd = [sys.executable, script] + (["--analyze-only"] if args.analyze_only else [])
        t0 = time.perf_counter()
        code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        timings.append((name, time.perf_counter() - t0, code))
        if code:
            print(f"\n{script} failed with exit code {code}.", file=sys.stderr)
            break

    print("\n=== summary ===")
    for name, sec, code in timings:
        print(f"  {name:<13} {sec:7.1f}s  {'ok' if code == 0 else f'FAILED ({code})'}")
    print(f"  {'total':<13} {time.perf_counter() - t_all:7.1f}s")
    sys.exit(max((c for _, _, c in timings), default=0))


if __name__ == "__main__":
    main()
