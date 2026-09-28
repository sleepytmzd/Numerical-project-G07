# Stage 3 Summary — Aritra Debnath (2105010)

## Status

Stage 3 is implemented and verified locally. The frozen experiment protocol was
used (`N_GRID`, 252 steps, 20 replications), both experiment scripts run
end-to-end, all four required figures and raw CSVs were generated, and the full
test suite passes **50/50**. Changes have not been committed or pushed, leaving
the completed diff available for team review.

## What was done

1. Added `arithmetic_asian_payoff`, using the same monitoring-window convention
   as the existing geometric payoff.
2. Added a chunked control-variate estimator with:
   - a geometric-Asian control for the arithmetic Asian;
   - a European-call control for up-and-in barriers;
   - analytic control means;
   - `b_hat` fitted on an independent pilot sample;
   - `rho`, `b_hat`, pilot size, raw-vs-adjusted SE, theoretical VRF, and
     observed VRF logged in `extra_json`;
   - pilot cost included in runtime.
3. Added tests for the arithmetic payoff/dispatcher, synthetic known-coefficient
   regression, arithmetic-Asian variance reduction, and barrier consistency.
4. Ran the main control experiment over `asian_arith`, `paper_barrier`, and
   `deep_barrier`, comparing plain MC with the control estimator.
5. Validated the no-closed-form arithmetic price against a separate
   2,000,000-path plain-MC run.
6. Ran the barrier-monitoring study at 12, 52, and 252 dates and compared it
   with the BGK continuity correction.
7. Wrote report Section 04 with methods, equations, results, limitations, and
   acceptance evidence.

## Key findings

At `N=65536`:

| Scenario | Price | rho | b_hat | observed within-run VRF |
|---|---:|---:|---:|---:|
| Arithmetic Asian call | 3.162776 | 0.999488 | 1.045681 | **975.7×** |
| Paper barrier, B=110.6772 | 7.094189 | 0.999675 | 1.000879 | **1542.4×** |
| Deep barrier, B=140 | 3.379102 | 0.830485 | 0.763175 | **3.2×** |

- Arithmetic validation: the 2M-path plain reference is
  `3.162517 ± 0.008511` (95% CI); the control result differs by only 0.06
  combined standard errors.
- The vanilla-call control is spectacular at the paper barrier because the
  up-and-in payoff is almost the vanilla payoff; it collapses at B=140 because
  many positive vanilla payoffs do not knock in.
- Daily monitoring gives `7.093583`, a bias of **−0.011945** from the continuous
  value. BGK gives `7.092995`, only 0.000588 lower.
- The daily bias is resolved at 54.1 SE of the replicated control mean, while a
  same-size plain run has SE ≈0.0489 and cannot reveal it.
- BGK is excellent at 252 dates but visibly less accurate at 12 and 52 dates;
  it should be described as an asymptotic correction.
- The workplan's provisional 252-step value (~7.076) is obsolete. Stage 2's
  corrected ~7.0941 target is independently confirmed here.

## Files changed or added

### Source and tests

- `src/payoffs.py` — appended `arithmetic_asian_payoff`.
- `src/vr_control.py` — new control-variate implementation and registry entry.
- `tests/test_control.py` — four Stage-3 tests, including the required synthetic
  correlated-normal smoke test.
- `tests/test_paths.py` — replaced the obsolete Stage-2 placeholder expecting
  `NotImplementedError` with a positive arithmetic dispatcher test.

### Experiments and report

- `experiments/exp_control.py`
- `experiments/exp_barrier_bias.py`
- `report/sections/04_control_variates.md`
- `README.md` — corrected the Stage-3 figure-to-CSV ownership entries.
- `HANDOFF-S1.md` — filled the Stage-3/Aritra handoff entry.
- `readme-AD.md` — this summary.

### Generated outputs

- Raw: `results/raw/aritra_control.csv` (601 rows),
  `results/raw/aritra_barrier_bias.csv` (60 rows).
- Figures: `cv_asian_vrf.png`, `cv_rho_vs_vrf.png`,
  `cv_barrier_collapse.png`, `barrier_bias_vs_steps.png`.
- Tables/notes: `control_summary.csv`, `control_findings.md`,
  `barrier_bias_summary.csv`, `barrier_bias_findings.md`.

## Reproduce

From the repository root, with one BLAS thread:

```powershell
$env:OMP_NUM_THREADS = "1"
python -m pytest tests/ -q
python experiments/exp_control.py
python experiments/exp_barrier_bias.py
```

To rebuild figures and tables without rerunning simulation:

```powershell
python experiments/exp_control.py --analyze-only
python experiments/exp_barrier_bias.py --analyze-only
```

## Notes for the next stage

- Importing `src.vr_control` registers `"control_variate"` in the generic
  benchmark registry; no edit to `benchmark.py` was necessary.
- `n_paths` is the number of production paths. Read `pilot_paths` or
  `total_paths_simulated` from `extra_json` when auditing cost.
- The raw `std_error` for this estimator is valid because it is computed from
  i.i.d. adjusted production samples and `b_hat` comes from an independent
  pilot.
- Use the precise within-run VRF for technical comparisons. R=20 makes
  across-replicate variance and bootstrap efficiency ratios noisy.
- Stage-3 runtimes are indicative only. The workplan makes Stage 5's
  single-machine master sweep authoritative for final efficiency rankings.

