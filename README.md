# CSE 402 Group G07 — Monte Carlo Variance Reduction for Exotic Options

## Overview

This project extends [Gottimukkala (2024)](resources/Optimizing_Exotic_Option_Pricing_Monte_Carlo_Simul.pdf)
by implementing three additional variance-reduction techniques (control variates,
quasi-Monte Carlo, importance sampling) for pricing barrier and Asian options under
Geometric Brownian Motion. We benchmark all techniques against plain Monte Carlo and
antithetic variates on an honest efficiency metric.

**Members:** Nafis Nahian (2105007) · Aritra Debnath (2105010) · Tamzeed Mahfuz (2105012) · Arnob Biswas (2105015) · Zaki Rehnoom Unmona (2105016)

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd Numerical-project-G07
python -m venv .venv
.venv/Scripts/activate     # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Run tests (must pass before anything else)
pytest tests/ -v

# Regenerate every figure and table
python run_all.py --analyze-only   # from the committed CSVs, ~2 min
python run_all.py                  # full re-simulation, ~25 min single-threaded
python run_all.py --tests --only importance master   # pytest first, then a subset
```

Every experiment script also takes `--analyze-only` on its own, e.g.
`python experiments/exp_master.py --analyze-only`. `run_all.py` sets
`OMP_NUM_THREADS=1` for every script.

## Experiment Protocol

| Constant | Value | Rationale |
|---|---|---|
| `N_GRID` | `[256, 1024, 4096, 16384, 65536]` | Powers of 2 (Sobol requirement); 5 log-log points |
| `N_STEPS` | 252 | Daily monitoring, 1 year |
| `R` | 20 replications | Variance measured across replicates |
| `BASE_SEED` | 402 | `SeedSequence(402).spawn()` |
| Timing | `perf_counter`, `OMP_NUM_THREADS=1` | Honest, comparable timings |
| Authoritative timings | `experiments/exp_master.py` only | One machine, one interleaved run; earlier stages' timings are indicative |

### Scenarios (all use S0=100, r=0.03, σ=0.2, T=1.0)

| Name | Type | K | B | Used in |
|---|---|---|---|---|
| `paper_barrier` | Up-and-in call | 105 | 110.6772 | Stages 2–5 |
| `mid_barrier` | Up-and-in call | 105 | 130 | Stage 1 parity test |
| `deep_barrier` | Up-and-in call | 105 | 140 | Stages 3, 4, 5 |
| `paper_asian_geo` | Geometric Asian call/put | 105 | — | Stages 2, 4, 5 |
| `asian_30d` | Geometric Asian (last 30 days) | 105 | — | Stage 2 |
| `asian_arith` | Arithmetic Asian call | 105 | — | Stages 3, 5 |
| `rare_barrier` | Up-and-in call (extension, defined only inside `exp_importance.py`) | 105 | 160 | Stage 5 |

## Project Structure

```
Numerical-project-G07/
  README.md  requirements.txt  .gitignore  run_all.py
  src/
    __init__.py
    config.py      — Scenario dataclass and frozen constants           (Stage 1)
    analytic.py    — BS, barrier, BGK, geometric Asian closed forms    (Stage 1)
    results.py     — log_result() and load_all_results()               (Stage 1)
    paths.py       — GBM path generation (exact/Euler/Milstein)        (Stage 2)
    payoffs.py     — barrier, geometric Asian, arithmetic Asian payoffs (Stage 2+3)
    estimators.py  — plain MC, antithetic MC estimators                (Stage 2)
    benchmark.py   — generic sweep runner                              (Stage 2)
    plots.py       — shared figure style helpers                       (Stage 2)
    vr_control.py  — control-variate estimator                         (Stage 3)
    vr_qmc.py      — quasi-Monte Carlo (Sobol + Brownian bridge)       (Stage 4)
    vr_importance.py — importance sampling estimator                   (Stage 5)
  experiments/
    exp_baseline.py       — plain MC and antithetic baseline           (Stage 2)
    exp_control.py        — control-variate experiments                (Stage 3)
    exp_barrier_bias.py   — discretisation bias study                  (Stage 3)
    exp_qmc.py            — RQMC vs MC experiments                    (Stage 4)
    exp_scheme_order.py   — SDE scheme order study                    (Stage 4)
    exp_importance.py     — importance sampling experiments            (Stage 5)
    exp_master.py         — master comparison sweep                   (Stage 5)
    _stage5_common.py     — shared Stage-5 analysis helpers           (Stage 5)
  tests/
    test_analytic.py      — closed-form verification                   (Stage 1)
    test_paths.py         — engine sanity checks                       (Stage 2)
    test_estimators.py    — plain/antithetic estimators, sweep seeding (Stage 2)
    test_control.py       — control-variate coefficient and estimator  (Stage 3)
    test_qmc.py           — Sobol, Brownian bridge, RQMC estimator     (Stage 4)
    test_importance.py    — P(Z>4) smoke test, likelihood ratio, IS    (Stage 5)
  results/
    raw/        — one CSV per (person, experiment_id)
    figures/    — all generated plots
    tables/     — markdown findings and CSV summary tables
  report/
    proposal.md
    sections/
      01_introduction.md    (Stage 1)
      02_methodology.md     (Stage 2)
      03_baseline.md        (Stage 2)
      04_control_variates.md (Stage 3)
      05_convergence.md     (Stage 4)
      06_importance_sampling.md (Stage 5)
      07_comparison.md      (Stage 5)
      08_conclusion.md      (Stage 5)
    final_report.md         (Stage 6: all sections assembled, abstract, contribution matrix, references)
  resources/
    CSE 402 Course Outline.pdf
    Optimizing_Exotic_Option_Pricing_Monte_Carlo_Simul.pdf
```

## Figure → Script → CSV Map

| Figure | Generating Script | Input CSV(s) |
|---|---|---|
| `baseline_barrier_ci.png` | `experiments/exp_baseline.py` | `results/raw/nafis_baseline.csv` |
| `baseline_asian_call_ci.png` | `experiments/exp_baseline.py` | `results/raw/nafis_baseline.csv` |
| `baseline_asian_put_ci.png` | `experiments/exp_baseline.py` | `results/raw/nafis_baseline.csv` |
| `baseline_rmse_vs_n.png` | `experiments/exp_baseline.py` | `results/raw/nafis_baseline.csv` |
| `baseline_scheme_comparison.png` | `experiments/exp_baseline.py` | `results/raw/nafis_baseline.csv` |
| `coverage_table.png` | `experiments/exp_baseline.py` | `results/raw/nafis_baseline.csv` |
| `cv_asian_vrf.png` | `experiments/exp_control.py` | `results/raw/aritra_control.csv` |
| `cv_rho_vs_vrf.png` | `experiments/exp_control.py` | `results/raw/aritra_control.csv` |
| `cv_barrier_collapse.png` | `experiments/exp_control.py` | `results/raw/aritra_control.csv` |
| `barrier_bias_vs_steps.png` | `experiments/exp_barrier_bias.py` | `results/raw/aritra_barrier_bias.csv` |
| `qmc_vs_mc_asian.png` | `experiments/exp_qmc.py` | `results/raw/zaki_qmc.csv` |
| `qmc_vs_mc_barrier.png` | `experiments/exp_qmc.py` | `results/raw/zaki_qmc.csv` |
| `qmc_bridge_vs_incremental.png` | `experiments/exp_qmc.py` | `results/raw/zaki_qmc.csv` |
| `qmc_effective_dimension.png` | `experiments/exp_qmc.py` | — (analytic, no CSV) |
| `scheme_strong_order.png` | `experiments/exp_scheme_order.py` | `results/raw/zaki_scheme_order.csv` |
| `scheme_weak_order.png` | `experiments/exp_scheme_order.py` | `results/raw/zaki_scheme_order.csv` |
| `is_deep_barrier.png` | `experiments/exp_importance.py` | `results/raw/tamzeed_importance.csv`, `zaki_qmc.csv` (references) |
| `is_weight_distribution.png` | `experiments/exp_importance.py` | `results/raw/tamzeed_importance.csv` (+ a deterministic diagnostic draw) |
| `is_theta_scan.png` | `experiments/exp_importance.py` | `results/raw/tamzeed_importance.csv` |
| `master_efficiency_table.png` | `experiments/exp_master.py` | `results/raw/tamzeed_master.csv`, `tamzeed_master_reference.csv`, `zaki_qmc.csv` |
| `master_rmse_vs_n.png` | `experiments/exp_master.py` | same as above |
| `master_efficiency_bars.png` | `experiments/exp_master.py` | same as above |

Tables: each script also writes `results/tables/<topic>_summary.csv` and
`<topic>_findings.md` (Stage 5: `importance_*`, `master_summary.csv`,
`master_efficiency.csv`, `master_findings.md`). Report numbers are quoted from
the `*_findings.md` files.

## Verified Reference Values

| Quantity | Value |
|---|---|
| Vanilla European call, K=105 | **7.12806** |
| Up-and-in call, B=110.6772 (continuous) | **7.10553** |
| Up-and-out call (in-out parity) | **0.02254** |
| Geometric Asian call (continuous) | **2.98488** |
| BGK-corrected barrier (252 steps) | **≈7.093** |

252-date (discretely monitored) references used for bias/RMSE from Stage 4 on:
paper barrier **7.094133 ± 0.000096**, deep barrier **3.385730 ± 0.001014**
(64 RQMC scrambles × 65,536, `zaki_qmc.csv`), arithmetic Asian **≈3.16298**
(64 scrambles, `tamzeed_master_reference.csv`), geometric Asian **3.000200**
(discrete closed form). The initial plan's "≈7.076" for the 252-step barrier is
superseded (Stage 2 finding, report §3.5).

## Report and handoff

- The full report is `report/final_report.md`, assembled from
  `report/sections/01–08`. Edit the section files, then re-assemble.

## License

Course project — CSE 402, BUET.
