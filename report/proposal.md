# Project Proposal

**Course:** CSE 402 — Numerical Analysis, Simulation and Modeling Sessional
**Group:** G07
**Members:** Nafis Nahian (2105007) · Aritra Debnath (2105010) · Tamzeed Mahfuz (2105012) · Arnob Biswas (2105015) · Zaki Rehnoom Unmona (2105016)
**Date:** [Submission date]

## 1. Base Paper

**Title:** Optimizing Exotic Option Pricing: Monte Carlo Simulation and Variance Reduction Techniques
**Author:** Srinivas R. Gottimukkala (2024), Preprints.org, doi: 10.20944/preprints202409.2256.v1

The paper prices two path-dependent exotic options — an up-and-in Barrier call and a
geometric-average-rate Asian option — using Monte Carlo (MC) simulation, validated
against Black-Scholes-style closed-form solutions. Underlying asset paths are generated
under three GBM discretization schemes (Euler, Euler-Maruyama, Milstein), and a single
variance-reduction technique, antithetic variates, is applied to both option types.

**Key findings of the base paper:**
- The choice of discretization scheme (Euler vs. Euler-Maruyama vs. Milstein) has
  negligible effect on pricing accuracy.
- Antithetic variates reduce variance and accelerate convergence by roughly 1.5x for
  the Barrier option and 1.3x for the Asian option.
- The authors explicitly identify **more advanced variance-reduction techniques** as
  the priority direction for future work, since the discretization scheme was shown not
  to matter.

## 2. Motivation and Gap

The base paper tests only one variance-reduction method (antithetic variates) and only
two option configurations, leaving its own central question unanswered: *how much
further can variance be reduced, and at what computational cost, using more advanced
techniques?* It also never prices an option without a closed-form benchmark (both of
its examples — geometric Asian and a vanilla Barrier — happen to have exact solutions),
so it never demonstrates MC simulation's real advantage: pricing something that
*cannot* be priced in closed form.

## 3. Proposed Extension

We extend the base paper along the exact direction it identifies as future work:
implementing and rigorously benchmarking **three additional variance-reduction
techniques** against the paper's antithetic-variates baseline, and applying them to a
genuinely path-dependent option with no closed-form solution.

**Objectives:**
1. Replicate the base paper's baseline (closed-form pricers, 3 discretization schemes,
   antithetic variates) as a correctness-verified foundation, reproducing its reported
   exact values and speedup factors.
2. Implement **control variates**, using the closed-form geometric-average Asian price
   as a control for the **arithmetic-average Asian option** (which has no closed-form
   solution — unlike the paper's geometric-average example).
3. Implement **quasi-Monte Carlo (QMC)** simulation using Sobol low-discrepancy
   sequences combined with a **Brownian bridge path construction**, so the
   low-discrepancy advantage is not lost to the high dimensionality of path simulation.
4. Implement **importance sampling** for the Barrier option, shifting the simulated
   drift (Girsanov change of measure) so barrier-crossing paths occur more frequently,
   correcting with a likelihood-ratio reweighting.
5. Build a shared benchmarking framework that reports, for every technique: variance
   reduction factor, standard error, 95% confidence interval, and an **efficiency
   metric** (1 / (variance × CPU time)) that accounts for the extra per-path cost some
   techniques introduce — something the base paper never quantifies.

## 4. Methodology

| Stage | Description |
|---|---|
| Baseline replication | Closed-form BS/Barrier/geometric-Asian pricers; Euler, Euler-Maruyama, Milstein path generators; plain MC and antithetic-variate estimators. Validated against the paper's reported values (e.g. exact Barrier price = 7.1055). |
| New technique 1 | Control variates for arithmetic-average Asian options, using the geometric-average Asian as control. |
| New technique 2 | Sobol-sequence QMC with Brownian-bridge path construction, applied to both Barrier and Asian options. |
| New technique 3 | Importance sampling (drift-shifted GBM + likelihood-ratio correction) for the Barrier option. |
| Benchmarking | Convergence plots (price and CI width vs. number of simulations), variance reduction factors, and efficiency comparison across all techniques and both option types. |

**Tools:** Python (NumPy, SciPy — `scipy.stats.qmc` for Sobol sequences — and Matplotlib
for plotting), following the same technology choice as the base paper.

## 5. Expected Outcomes

- A validated, reusable Monte Carlo pricing framework covering Barrier and Asian
  (both geometric- and arithmetic-average) exotic options.
- A head-to-head efficiency comparison of five variance-reduction techniques (plain MC,
  antithetic, control variate, QMC + Brownian bridge, importance sampling), directly
  extending the base paper's 1.3–1.5x antithetic-only result with a quantified answer
  to which technique performs best, for which option type, and why.
- A demonstration of MC simulation's core value proposition — pricing an option
  (arithmetic-average Asian) that has no closed-form solution — which the base paper's
  examples never required.

## 6. Timeline

| Week | Milestone |
|---|---|
| 8 | Project proposal (this document) |
| 9–10 | Baseline replication and validation against base paper's reported values |
| 11–12 | Implementation of control variates, QMC + Brownian bridge, and importance sampling |
| 13 | Benchmarking framework, convergence plots, efficiency comparison table |
| 14 | Final project update and report submission |

## 7. References

1. Gottimukkala, S. R. (2024). *Optimizing Exotic Option Pricing: Monte Carlo Simulation
   and Variance Reduction Techniques.* Preprints.org. doi: 10.20944/preprints202409.2256.v1
