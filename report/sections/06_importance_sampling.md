## 6. Importance Sampling for Barrier Options

*(Stage 5 — Tamzeed Mahfuz, 2105012)*

Control variates and RQMC reduce variance by exploiting structure that the
payoff shares with something simpler: a correlated closed-form option, or a
low effective dimension. Importance sampling (IS) instead changes *where the
paths go*. An up-and-in call pays only on paths that first reach the barrier and
then finish above the strike. If that event is rare, most plain-MC paths
contribute exactly zero, and the estimator's variance is dominated by the few
that do not. IS simulates under a drift that sends paths towards the barrier
and corrects the bias with a likelihood ratio.

This chapter covers:

- the change of measure and why its likelihood ratio needs only `W_T` (§6.1);
- how the drift is chosen, including a units correction to the WORKPLAN's
  heuristic (§6.2);
- the results on three barriers of increasing rarity (§6.3–6.5);
- the weight diagnostics that decide whether an IS result can be trusted (§6.6).

Files:

- Estimator: `src/vr_importance.py`, registered as `"importance"`.
- Experiment: `experiments/exp_importance.py`.
- Tests: `tests/test_importance.py` (10 tests, including the task-0 smoke test).
- Raw output: `results/raw/tamzeed_importance.csv` (975 rows: 900 sweep rows and
  75 θ-scan rows).
- Derived tables: `results/tables/importance_summary.csv` and
  `importance_findings.md`.

Timings in this chapter are indicative. The authoritative comparison is the
master sweep in Chapter 7.

### 6.1 Change of measure

The engine simulates `dS = μ S dt + σ S dW~`, where `W~` is the Brownian motion
built from the normals we draw. Under the risk-neutral measure `Q` the drift is
`μ = r`. Importance sampling uses instead

    μ = r + σθ,        passed to generate_paths as drift_override,

which is the same as saying the risk-neutral Brownian motion is `W = W~ + θt`.
By Girsanov's theorem the density of `Q` relative to the shifted measure `P_θ`
is

    L = dQ/dP_θ = exp(−θ W~_T − ½ θ² T).

The estimator is the sample mean of `L · Y`, where `Y` is the discounted payoff
of the shifted path. It is unbiased for any fixed `θ`, because
`E_θ[L·Y] = E_Q[Y]`.

**The likelihood ratio telescopes (trap #8).** The per-step density ratios
multiply to a function of the *sum* of the increments, which is `W~_T`. So the
barrier monitoring, the path maximum and the 252-step structure never enter the
weight. The code computes `W~_T = √Δt · Σ_j Z_j` from the **same** normal array
it passes to the engine, and never draws fresh normals for the weight. A test
checks this path by path (`test_likelihood_ratio_uses_the_same_normals`).

**Task-0 smoke test.** Before any option was involved, the same
`likelihood_ratio` function estimated `P(Z > 4) = 3.167 × 10⁻⁵` with a proposal
shifted to mean 4. With 20,000 draws per run, plain MC would see about 0.6 hits.
Across 50 independent runs the pooled IS estimate is within 3 SE of
`norm.sf(4)`, and the per-run z-scores have mean below 0.5 and SD in [0.7, 1.3].
The mean relative SE is under 3%.

The test checks calibration across many seeds rather than one. That was a
deliberate choice: at seed 402, a single 100,000-draw run landed 3.5 SE away. We
checked 200 further seeds: z-mean −0.14, z-SD 1.05, and no |z| > 3. That
confirmed the estimator is calibrated and seed 402 was a roughly 1-in-2000
draw. A test that passes or fails on one seed is testing the seed.

### 6.2 Choosing the drift

**Heuristic.** Choose `θ₀` so that the expected terminal log-price sits on the
barrier:

    log S0 + (r + σθ₀ − σ²/2) T = log B
    ⇒  θ₀ = [ log(B/S0)/T − (r − σ²/2) ] / σ.

The WORKPLAN writes this without the final `/σ`. That expression is a
*log-price drift*, not the Brownian shift that `drift_override = r + σθ`
expects. With σ = 0.2 the uncorrected value would shift by only 20% of the
intended amount. A unit test pins `E[log S_T] = log B` at `θ₀`.

**Pilot grid search.** Each estimate tunes its own `θ` on an independent pilot
of 10,000 paths, drawn from a separate `SeedSequence` child. The 7 candidates
are `θ₀ · {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5}`, and the one with the smallest
empirical second moment `E_θ[(L_θ Y)²]` wins. Including `θ = 0` lets the search
fall back to plain MC if shifting does not help.

All seven candidates are scored from **one** pilot, because `L` depends only on
the terminal value. Draw the pilot under `θ₀` and let `W_T` be each pilot path's
risk-neutral terminal value. Then for any other `θ`:

    E_θ[(L_θ Y)²] = E_Q[L_θ Y²] = E_θ₀[L_θ₀ · L_θ · Y²],
    with L_θ = exp(−θ W_T + ½θ²T).

The pilot's runtime is included in every IS timing, and its path count is logged
(`pilot_paths`, `total_paths_simulated`). There is no cross-entropy or adaptive
machinery, as the WORKPLAN specifies.

| Scenario | B | θ₀ | θ chosen by the pilot | Optimum of the full scan (§6.5) |
|---|---:|---:|---|---|
| `paper_barrier` | 110.68 | 0.457 | 0.686 = 1.5·θ₀ in **every** replicate (the grid edge) | ≈2.9·θ₀ |
| `deep_barrier` | 140 | 1.632 | 1.632 = θ₀ in every replicate | θ₀ |
| `rare_barrier` | 160 | 2.300 | 2.300 = θ₀ in every replicate | θ₀ |

The pilot choice is stable: across all 100 estimates per scenario it always
picked the same grid point.

### 6.3 Headline: the deep barrier (B = 140)

The WORKPLAN calls `deep_barrier` the rare-event case. Measuring it (diagnostic
draw, 65,536 paths) shows it is only moderately rare:

| Scenario | P(knock in) under Q | P(payoff > 0) under Q | P(payoff > 0) under the IS proposal |
|---|---:|---:|---:|
| `paper_barrier` | 60.4% | 40.5% | 67.7% |
| `deep_barrier` | 9.3% | 9.2% | 59.8% |
| `rare_barrier` (B=160) | 1.9% | 1.9% | 56.6% |

Plain MC does get a non-zero payoff on about one path in eleven. IS raises that
to six in ten. At N = 65,536 (from `importance_findings.md`):

| Method | Mean price | RMSE | Within-run VRF | Efficiency vs plain [bootstrap 95% CI] | Efficiency vs antithetic |
|---|---:|---:|---:|---|---|
| Plain MC | 3.4107 | 0.0498 | 1 | 1 | 0.93 [0.30, 3.67] |
| Antithetic | 3.3713 | 0.0476 | 1.11 | 1.07 [0.28, 3.20] | 1 |
| **Importance** | **3.3881** | **0.0132** | **9.88** | **10.7 [2.8, 27.0]** | **10.0 [4.1, 20.7]** |

The 252-date reference is 3.385730 ± 0.001014 (Stage 4). IS lands 2.4 × 10⁻³
from it, well inside its own SE. The within-run VRF of 9.9 is stable across N
(9.6–10.6), and it matches the θ scan's minimum variance of 0.101 × plain.

The across-replicate VRF (11.1, with 95% F-interval [4.4, 27.9]) agrees with it,
but it is more than six times wider. That is the R = 20 limitation that Stage 2
already flagged. For IS the within-run SE is valid, because the weighted samples
are i.i.d., so the within-run VRF is the precise number.

Antithetic variates, the base paper's only technique, do **nothing** here: VRF
1.11. The payoff is non-monotone in the Brownian path, since paths must go up to
the barrier *and* finish in the money, so `Z` and `−Z` are barely
anti-correlated.

### 6.4 The pilot cost, and when IS pays

The pilot is a fixed 10,000 paths regardless of N. At small N it dominates:

| N | IS time ÷ plain time (deep) | IS efficiency vs plain (deep) | (paper) | (rare) |
|---:|---:|---|---|---|
| 256 | 51.8 | 0.27 [0.09, 0.68] | 0.05 [0.02, 0.13] | 0.71 [0.30, 2.17] |
| 1,024 | 10.1 | 0.71 [0.26, 2.22] | 0.29 [0.10, 0.90] | 0.94 [0.35, 2.28] |
| 4,096 | 3.2 | 3.9 [1.5, 12.6] | 0.64 [0.33, 1.24] | 10.4 [4.5, 31.4] |
| 16,384 | 1.6 | 8.2 [3.7, 18.1] | 6.3 [3.0, 13.7] | 12.8 [6.5, 26.1] |
| 65,536 | 1.03 | 10.7 [2.8, 27.0] | 5.9 [3.0, 11.2] | 46.6 [19.8, 107.8] |

At N = 256, IS is significantly *less* efficient than plain MC on the deep and
paper barriers. The pilot is 39 times the production run. The break-even point
is around N ≈ 2,000–4,000. From N = 16,384 up, IS beats plain MC on all three
barriers, with confidence intervals excluding 1.

Two remedies exist, and we did not use either in the benchmark, to keep it
honest:

- **Tune θ once offline and fix it.** The pilot chose the same θ in all 100
  estimates per scenario, so a fixed θ would lose nothing and remove the
  overhead entirely.
- **Scale the pilot with N.**

Either way, a fixed-size pilot is the wrong design for small-N use.

### 6.5 How good is the heuristic? The θ scan

To see the whole variance curve the pilot samples, `exp_importance.py` also runs
fixed-θ IS at N = 65,536:

- 25 values `θ = θ₀ · {0, 0.125, …, 3}`;
- common random numbers across θ;
- per-path variance `N · SE²` recorded for each value.

The result is `results/figures/is_theta_scan.png`.

- **Deep and rare barriers.** The minimum is exactly at `θ₀`: 0.101 × plain
  (deep) and 0.042 × plain (rare). The heuristic is optimal when the barrier is
  the binding constraint, since a path that reaches `B = 140` or `160` has
  almost certainly finished above `K = 105`.
- **Paper barrier.** Here `B = 110.68` is barely above `K = 105`, and 60% of
  plain paths already knock in, so **the strike, not the barrier, is the
  binding constraint**. A heuristic aimed at the barrier therefore undershoots.
  - The variance keeps falling past the grid's 1.5·θ₀ and flattens only at
    ≈2.9·θ₀, at 0.095 × plain.
  - At the pilot's choice it is 0.217 × plain. The 7-point grid therefore leaves
    a factor of **≈2.3** of variance unused.
  - This is a limit of the spec'd grid, not of IS. The grid is fixed relative to
    `θ₀`, so it can never reach an optimum that sits far from `θ₀`.
- **Overshooting is dangerous.** For the rare barrier, the variance is 1 × plain
  at 2·θ₀ and 22 × plain at 2.4·θ₀. Beyond that the *estimated* variance becomes
  erratic: 577 at 2.5·θ₀, then 17 at 3·θ₀. That is the classic IS failure. The
  weights are so heavy-tailed that even the sample variance is unreliable, and
  one run can look fine by chance. The curve is steep on the right and gentle
  on the left, so a pilot search should err towards smaller θ.

### 6.6 Weight diagnostics: ESS and the maximum weight

Every IS estimate logs the Kish effective sample size `ESS = (Σw)² / Σw²`, the
maximum weight, and its share of `Σw`. At the tuned θ (N = 65,536,
`is_weight_distribution.png`):

| Scenario | ESS / N | max w | max w / Σw | Top 1% of paths' share of the price | Within-run VRF |
|---|---:|---:|---:|---:|---:|
| `paper_barrier` | 62.5% | 18.7 | 2.9 × 10⁻⁴ | 1% | 4.6 |
| `deep_barrier` | 7.6% | 227 | 3.5 × 10⁻³ | 2% | 9.9 |
| `rare_barrier` | 1.1% | 1,126 | 1.7 × 10⁻² | 6% | 22.2 |

At first sight this table is a paradox: the lower the ESS, the *better* IS
works. The ESS here measures degeneracy of the weights over **all** paths. The
histograms show where the large weights sit:

- Paths that knock in and pay have weights `w < 1`. The shift over-samples them,
  so they are down-weighted.
- The large weights (up to 1,126) sit almost entirely on paths with **zero
  payoff**. These paths moved against the shifted drift and never reached the
  barrier, so they contribute nothing to `L · Y` and nothing to its variance.

So for an indicator-type payoff, ESS is a *conservative* diagnostic. The one
that matters is the concentration of the weighted *payoff*. Even for the rare
barrier, the top 1% of paths carry only 6% of the price, and no single path
carries more than 1.7% of `Σw`. The estimate is therefore not resting on a
handful of paths.

We report both, as trap #8 asks. An IS result that quoted only its VRF, without
these two columns, could not be distinguished from the overshoot failure in
§6.5.

### 6.7 The rare-barrier extension (B = 160)

`deep_barrier` turned out to be only moderately rare (9.3%). To test the
rare-event premise itself, `exp_importance.py` adds `rare_barrier`:

- B = 160, with a knock-in probability of 1.9%;
- defined at runtime with `dataclasses.replace`, so `src/config.py`'s frozen
  scenarios are unchanged;
- no independent 252-date reference, so only variance-based metrics are
  reported for it.

Its IS mean is 1.0649, next to the BGK approximation of 1.0617.

At N = 65,536, IS reaches:

- within-run VRF **22.2**;
- efficiency vs plain **46.6× [19.8, 107.8]**;
- efficiency vs antithetic **20.3× [9.2, 46.8]**.

This is the regime the WORKPLAN had in mind: plain MC wastes 98% of its paths,
and IS is the right tool.

### 6.8 Acceptance and summary

- **Correctness (§1.7.1–2).** Every IS cell on the two referenced barriers is
  within 3 SE of the 252-date reference (10/10), and every t-interval covers it
  (10/10). Per-replicate 95% CI coverage is 0.97 (deep) and 0.96 (paper).
- **Sanity check (WORKPLAN §6.6), "IS wins big on `deep_barrier`, marginally on
  `paper_barrier`".** It holds in direction:
  - deep ≈ 10× (within-run VRF 9.9);
  - paper ≈ 5–6× (within-run VRF 4.6).

  "Big" is only true where the event is actually rare: 47× at B = 160. On the
  paper barrier the gain is modest for a second reason, the heuristic
  undershooting (§6.5), and not only because the event is common.
- **Place in the comparison.** In the master sweep (Chapter 7), IS is the
  second-best method on the deep barrier, at 8.6× [3.0, 37.8]. RQMC is first
  at 21.3× [7.9, 51.4], and the confidence intervals overlap.
  - On the paper barrier IS gives only 2.8× [1.3, 5.7], while the control
    variate and RQMC give over 1,000×.
  - IS attacks rarity, and only the deep and rare barriers are rare. The paper
    barrier is nearly a vanilla call, which the other two techniques exploit
    directly.
