# 1. Introduction

## 1.1 Motivation

Monte Carlo (MC) simulation is the dominant computational tool for pricing path-dependent
exotic options, precisely because it requires no assumptions beyond the ability to simulate
the underlying stochastic process. Unlike closed-form solutions — which exist only for
special payoff structures — MC can price any contingent claim whose payoff is a function of
the simulated path. The cost, however, is statistical: a naive MC estimate converges at the
slow rate O(N^{−1/2}), so reducing its variance without introducing bias is one of the most
practically consequential problems in computational finance.

## 1.2 Theoretical Background

### 1.2.1 Options and Exotic Options

A European option gives its holder the right (but not the obligation) to buy (call) or sell
(put) an underlying asset at a fixed strike price K at maturity T. The celebrated
**Black–Scholes (1973)** formula provides a closed-form solution for European options under
Geometric Brownian Motion (GBM):

  dS_t = r S_t dt + σ S_t dW_t

where S_t is the asset price, r the risk-free rate, σ the volatility, and W_t a standard
Brownian motion under the risk-neutral measure.

**Exotic options** are path-dependent derivatives whose payoff depends on the entire trajectory
of S_t, not just its terminal value:

- **Barrier options** become activated ("knock-in") or extinguished ("knock-out") if the
  underlying breaches a barrier level B during the option's life. An **up-and-in call** has
  payoff (S_T − K)⁺ · **1**{max_{0≤t≤T} S_t ≥ B}. Reiner & Rubinstein (1991) provide
  closed-form prices under continuous monitoring; discrete monitoring introduces a downward
  bias that Broadie, Glasserman & Kou (1997) quantified via their celebrated continuity
  correction.

- **Asian options** have payoff depending on the average asset price. The **geometric-average**
  Asian has a closed-form solution (Kemna & Vorst, 1990) because the geometric mean of
  lognormals is lognormal. The **arithmetic-average** Asian has no closed form, making it the
  canonical demonstration of Monte Carlo's indispensability.

### 1.2.2 Monte Carlo Simulation Under GBM

Under the risk-neutral measure, the exact solution of GBM at discrete times is:

  S_{t+Δt} = S_t · exp((r − σ²/2)Δt + σ√Δt · Z),   Z ~ N(0,1)

This "exact" (log-GBM) scheme is bias-free at any step size. Two alternative SDE
discretisation schemes are commonly used:

- **Euler–Maruyama**: S_{t+Δt} = S_t + r·S_t·Δt + σ·S_t·√Δt·Z (strong order 0.5)
- **Milstein**: adds the Itô correction +0.5·σ²·S_t·(Z²−1)·Δt (strong order 1.0)

Both converge to the same expectation as Δt → 0 (weak order 1.0 for both), so the scheme
choice cannot change a European-style price — a fact the base paper reports as a finding but
which is, as we demonstrate, a mathematical tautology.

### 1.2.3 Variance Reduction Techniques

The practical limitation of MC is its O(N^{−1/2}) convergence rate. **Variance reduction**
techniques accelerate convergence by reducing the estimator's variance per sample:

1. **Antithetic variates**: exploit the symmetry of the standard normal by pairing each path
   generated with Z with a mirror path using −Z. This halves the variance of symmetric
   payoffs and costs essentially nothing extra.

2. **Control variates**: if a correlated quantity has a known expectation (e.g., the geometric
   Asian price for the arithmetic Asian), the optimal linear control subtracts a regression-
   adjusted residual. The variance reduction factor is 1/(1−ρ²), where ρ is the correlation
   between the target and control payoffs.

3. **Quasi-Monte Carlo (QMC)**: replaces pseudo-random sampling with low-discrepancy sequences
   (e.g., Sobol) that fill the unit hypercube more uniformly. Combined with the Brownian
   bridge construction (which assigns the lowest-discrepancy Sobol dimensions to the highest-
   variance time components), QMC can achieve convergence rates of O(N^{−α}) with α ∈ (0.5, 1]
   for smooth integrands.

4. **Importance sampling (IS)**: changes the simulation measure so that rare but important events
   (e.g., barrier crossings for deep out-of-the-money barriers) occur more frequently, then
   corrects with a likelihood ratio. The payoff of IS is largest for rare events but can
   backfire if the importance distribution has heavy tails (diagnosed via effective sample size).

## 1.3 Related Work

Gottimukkala (2024) prices an up-and-in barrier call (K=105, B=110.6772) and a geometric-
average Asian option under GBM, applying three discretisation schemes and antithetic variates
as the sole variance-reduction method. The paper's key findings are:

- Discretisation scheme choice is negligible for pricing accuracy.
- Antithetic variates reduce variance by approximately 1.5× for the barrier option and 1.3×
  for the Asian option.
- "Advanced variance reduction techniques" are identified as the top future-work item.

The paper does not quantify computational cost alongside variance, so its speedup claims are
variance ratios, not efficiency ratios. It also never prices an option without a closed-form
benchmark, limiting its demonstration of Monte Carlo's practical value.

## 1.4 What the Base Paper Did vs. What We Add

| Aspect | Base Paper | Our Extension |
|---|---|---|
| **Options** | Up-and-in barrier call; geometric Asian | + Arithmetic Asian (no closed form) |
| **Variance reduction** | Antithetic variates only | + Control variates, QMC (Sobol + Brownian bridge), importance sampling |
| **Efficiency metric** | Variance reduction factor (VRF) | Efficiency = 1/(variance × time) with bootstrap CI |
| **Error metric** | Std error | RMSE = √(bias² + variance), resolving discretisation bias |
| **Scheme comparison** | Claims Euler ≠ Euler–Maruyama; "negligible" Milstein difference | Demonstrates this is a tautology (same weak order); shows strong vs weak order separation |
| **Barrier bias** | Not discussed | Quantified ≈−0.03 at 252 steps; BGK correction; bias vs n_steps sweep |
| **Discrete monitoring** | Continuous formulas only | Discrete Kemna–Vorst and BGK-corrected barrier |
| **Rare events** | B=110.6772 (near the money) | B=140 (deep barrier) — demonstrates IS advantage |

Our project addresses the exact gap the base paper identifies: "more advanced variance
reduction techniques." We implement all three techniques named in the standard computational
finance curriculum (control variates, QMC, IS) and compare them on a metric that accounts
for the computational cost each technique introduces.
