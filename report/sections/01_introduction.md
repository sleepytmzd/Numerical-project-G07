## 1. Introduction

*(Stage 1 — Arnob Biswas, 2105015)*

### 1.1 Motivation

Monte Carlo (MC) simulation is the dominant computational tool for pricing path-dependent
exotic options, precisely because it requires no assumptions beyond the ability to simulate
the underlying stochastic process. Unlike closed-form solutions — which exist only for
special payoff structures — MC can price any contingent claim whose payoff is a function of
the simulated path. The cost, however, is statistical: a naive MC estimate converges at the
slow rate O(N^{−1/2}), so reducing its variance without introducing bias is one of the most
practically consequential problems in computational finance.

### 1.2 Theoretical Background

#### 1.2.1 Options and Exotic Options

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

#### 1.2.2 Monte Carlo Simulation Under GBM

Under the risk-neutral measure, the exact solution of GBM at discrete times is:

  S_{t+Δt} = S_t · exp((r − σ²/2)Δt + σ√Δt · Z),   Z ~ N(0,1)

This "exact" (log-GBM) scheme is bias-free at any step size. Two alternative SDE
discretisation schemes are commonly used:

- **Euler–Maruyama**: S_{t+Δt} = S_t + r·S_t·Δt + σ·S_t·√Δt·Z (strong order 0.5)
- **Milstein**: adds the Itô correction +0.5·σ²·S_t·(Z²−1)·Δt (strong order 1.0)

Both have weak order 1.0, so at daily steps the choice of scheme changes a price only at
O(Δt), about 0.001 here (§3.4, §5.7). That is far below Monte Carlo error at any practical N.
The base paper reports the scheme choice as "negligible". That is true, but it is expected
from theory rather than a discovery. Milstein's advantage is pathwise (strong order), and
an expectation does not benefit from it.

#### 1.2.3 Variance Reduction Techniques

The practical limitation of MC is its O(N^{−1/2}) convergence rate. **Variance reduction**
techniques accelerate convergence by reducing the estimator's variance per sample:

1. **Antithetic variates**: exploit the symmetry of the standard normal by pairing each path
   generated with Z with a mirror path using −Z. The pair average cancels the linear (odd)
   part of the payoff exactly, and does nothing for the even part. The gain therefore depends
   on how close to linear the payoff is: our measured variance reduction factors are 1.1–3.9
   (§3.3, §7.2).

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
   backfire if the importance distribution has heavy tails. We diagnose this with the effective
   sample size and with how concentrated the payoff-weighted likelihood ratios are (§6.6).

### 1.3 Related Work

Gottimukkala (2024) is a short concept paper (not peer reviewed). It uses MC under GBM with
three "random walk models" (Euler, Euler–Maruyama and Milstein) and antithetic variates as
the only variance-reduction method, and prices:

- an up-and-in barrier call with S0=100, K=105, B=110.6772, σ=0.2, r=0.03, maturity one year;
- a geometric-average Asian call and put, and the call with averaging over only the final
  30 days of the contract. The Asian parameters are not stated in the paper.

The paper contains no equations. It never writes down its schemes, closed forms or
estimators, and it does not state its number of time steps or random seed. The only price it
states is the barrier's exact value, 7.1055. That matches the continuous-monitoring
Reiner–Rubinstein formula at T = 1. Its conclusions are:

- The choice of "random walk model" has a negligible effect on accuracy.
- Antithetic variates reduce variance and accelerate convergence "by 1.5 times for Barrier
  options and 1.3 times for Asian options, despite increased computational time". The factor
  is never defined.
- Convergence "stabilizes" at about 5,000 simulations for the barrier and 750–1,000 for the
  Asian.
- Future work should "prioritize the development of variance reduction techniques over the
  creation of new random walk models" (the abstract says "advanced variance reduction
  techniques"). The paper also names models with changing volatility and interest rates.

The paper mentions computational time only qualitatively and never puts cost and variance
into one measure. It shows an arithmetic-average curve in a volatility-sensitivity plot
(its Fig. 7), but never prices, validates or reports an option without a closed form. Chapter 3
tests each of these claims against our own replication.

### 1.4 What the Base Paper Did vs. What We Add

| Aspect | Base Paper | Our Extension |
|---|---|---|
| **Options** | Up-and-in barrier call; geometric Asian call, put and 30-day call | + Arithmetic Asian (no closed form), priced and validated; deep (B=140) and rare (B=160) barriers |
| **Variance reduction** | Antithetic variates only | + Control variates, QMC (Sobol + Brownian bridge), importance sampling |
| **Efficiency metric** | An undefined "1.5×/1.3×" speedup; computational time mentioned only qualitatively | Efficiency = 1/(variance × time) with bootstrap CI; time to reach a target accuracy |
| **Error metric** | CI plots only; no standard errors reported as numbers | RMSE = √(bias² + variance) against 252-date reference prices |
| **Scheme comparison** | Three named but undefined schemes; "negligible" difference | Measured strong vs weak order (Euler–Maruyama 0.49 / Milstein 0.97 strong; both ≈1 weak); explains why prices cannot tell them apart |
| **Barrier bias** | Discrete monitoring and the 1997 continuity correction mentioned in the background, not applied | Quantified at −0.011 (252 dates, 7.0941 vs 7.1055) and −0.160 on the deep barrier; BGK correction; bias vs n_steps sweep |
| **Discrete monitoring** | Formulas not stated; the 7.1055 benchmark is the continuous value | Discrete Kemna–Vorst over the actual averaging window; high-precision 252-date barrier references |
| **Rare events** | B=110.6772 only (the up-and-out leg is worth just 0.02) | B=140 (9.3% knock-in): IS 8.6×, second to RQMC's 21.3×; B=160 (1.9% knock-in): IS ≈49× (§6.7) |

Our project takes up the base paper's own future-work direction: advanced variance-reduction
techniques. We implement the three standard ones from the computational-finance literature
(control variates, QMC, IS). We compare them on a metric that accounts for the extra cost each
one introduces.
