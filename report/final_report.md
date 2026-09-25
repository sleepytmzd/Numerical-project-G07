# Beyond Antithetic Variates: Benchmarking Variance Reduction for Monte Carlo Pricing of Barrier and Asian Options

**CSE 402 — Numerical Analysis, Simulation and Modeling Sessional · Group G07**

Nafis Nahian (2105007) · Aritra Debnath (2105010) · Tamzeed Mahfuz (2105012) ·
Arnob Biswas (2105015) · Zaki Rehnoom Unmona (2105016)

Base paper: S. R. Gottimukkala (2024), *Optimizing Exotic Option Pricing: Monte
Carlo Simulation and Variance Reduction Techniques*, Preprints.org,
doi:10.20944/preprints202409.2256.v1.

---

## Abstract

The base paper prices an up-and-in barrier call and geometric-average Asian
options by Monte Carlo under geometric Brownian motion, using antithetic
variates as its only variance-reduction technique. It names more advanced
variance reduction as future work. We rebuilt its experiment from scratch,
validated every estimator against closed forms, and tested its claims. Its
antithetic variance ratios (1.47 barrier, 1.35 Asian call) are confirmed. Its
"convergence stabilization" at 5,000 / 750–1,000 paths is not a convergence
property: the error keeps falling as N^-1/2. Its claim that the 30-day-average
Asian keeps the same closed form is false (6.708 vs 3.000). We then added
control variates, randomized quasi-Monte Carlo with a Brownian bridge, and
importance sampling. All five methods were benchmarked in one interleaved run
on an efficiency metric, 1/(variance × time), with bootstrap confidence
intervals and 252-date reference prices. At N = 65,536, control variates and
RQMC are 10²–10³ times more efficient than plain Monte Carlo on the near-strike
barrier and on both Asians. Importance sampling is the specialist for genuinely
rare knock-ins (≈49× at B = 160). Antithetic variates are the weakest technique,
and their gain is not resolvable at 20 replicates. Variance reduction also makes
the −0.011 discrete-monitoring bias visible, which plain Monte Carlo cannot see.
To reach a target accuracy, we recommend RQMC with a Brownian bridge for barrier
options and the geometric-Asian control variate for the arithmetic Asian.

## Contribution matrix

The project ran as a five-stage relay; each member owned one stage end to end
(code, tests, experiments, report chapter). Stage 6 (assembly) was shared.

| Stage | Member | Code and tests | Experiments and raw data | Report |
|---|---|---|---|---|
| 1. Foundations and analytic oracles | Arnob Biswas (2105015) | `src/config.py`, `src/analytic.py`, `src/results.py`; `tests/test_analytic.py` | — (reference values) | Ch. 1 |
| 2. Engine, estimators, benchmark infrastructure, baseline | Nafis Nahian (2105007) | `src/paths.py`, `src/payoffs.py`, `src/estimators.py`, `src/benchmark.py`, `src/plots.py`; `tests/test_paths.py`, `tests/test_estimators.py` | `exp_baseline.py` → `nafis_baseline.csv` | Ch. 2, 3 |
| 3. Control variates, arithmetic Asian, barrier bias | Aritra Debnath (2105010) | `src/vr_control.py`, `arithmetic_asian_payoff`; `tests/test_control.py` | `exp_control.py`, `exp_barrier_bias.py` → `aritra_*.csv` | Ch. 4 |
| 4. RQMC, Brownian bridge, discretization order | Zaki Rehnoom Unmona (2105016) | `src/vr_qmc.py`; `tests/test_qmc.py` | `exp_qmc.py`, `exp_scheme_order.py` → `zaki_*.csv` | Ch. 5 |
| 5. Importance sampling, master comparison, synthesis | Tamzeed Mahfuz (2105012) | `src/vr_importance.py`, `experiments/_stage5_common.py`, `run_all.py`; `tests/test_importance.py` | `exp_importance.py`, `exp_master.py` → `tamzeed_*.csv` | Ch. 6, 7, 8 |
| 6. Assembly | Everyone; final consistency pass and assembly by Nafis Nahian | — | — | This document, abstract, references |

## Contents

1. Introduction
2. Methodology
3. Baseline Replication
4. Control Variates, Arithmetic Asian Options, and Barrier Bias
5. Convergence Analysis: Quasi-Monte Carlo and Discretization Order
6. Importance Sampling for Barrier Options
7. Comparative Results and Efficiency Analysis
8. Discussion, Limitations and Conclusion

References · Appendix A: Reproducing every number

---

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

---

## 2. Methodology

*(Stage 2 — Nafis Nahian, 2105007)*

This section describes the shared simulation engine, the estimators built on top of
it, and the benchmarking protocol that every later stage's variance-reduction
technique plugs into.

### 2.1 Path simulation engine

Under the risk-neutral measure, the underlying follows geometric Brownian motion

```
dS_t = mu S_t dt + sigma S_t dW_t,        mu = r  (risk-neutral drift)
```

`src/paths.py::generate_paths` discretizes this on a uniform grid of `n_steps`
steps over `[0, T]` (`n_steps = 252`, i.e. daily monitoring for `T = 1`), and
implements three schemes with `dt = T/n_steps`:

| Scheme | Update |
|---|---|
| `exact` (log-GBM) | `S_{j+1} = S_j * exp((mu - sigma^2/2) dt + sigma*sqrt(dt)*Z_j)` |
| `euler_maruyama` | `S_{j+1} = S_j * (1 + mu*dt + sigma*sqrt(dt)*Z_j)` |
| `milstein` | Euler–Maruyama plus `0.5*sigma^2*dt*(Z_j^2 - 1)` |

The `exact` scheme has zero discretization error at the monitoring dates
regardless of `dt`, so it is used as the ground truth for the barrier and Asian
payoffs (which only ever look at the simulated grid points, never the
continuous path). `euler_maruyama` and `milstein` are included because the base
paper uses them; Stage 4's `exp_scheme_order.py` quantifies their actual strong
and weak convergence order.

**Frozen interface.** Per the group's frozen interface agreement, `generate_paths` never draws its
own randomness — it takes a pre-built `normals` array of shape
`(n_paths, n_steps)` (column *j* drives the step *j → j+1* transition for every
path). This is what lets every later variance-reduction technique reuse the
exact same engine and payoff code:

- **Antithetic variates** (this stage) build `normals` as `[Z, -Z]` — the two
  halves are mirror images, so `generate_paths` never needs to know it is being
  used antithetically.
- **Quasi-Monte Carlo** (Stage 4) builds `normals` from a scrambled Sobol
  sequence via `ndtri`, then applies the Brownian bridge as an orthogonal linear
  map (`normals = z @ M`, §5.2), so the first Sobol dimension drives `W_T`.
- **Importance sampling** (Stage 5) shifts the drift via the `drift_override`
  parameter — this replaces `mu = r` with an arbitrary drift `mu`, so it can
  pass `mu = r + sigma*theta` to simulate under the tilted measure `W̃ = W + theta*t`
  while keeping the *same* `normals` and the *same* discounting; the
  Radon–Nikodym weight is applied afterward by the estimator, not by the
  engine.

All three schemes are fully vectorized (no per-step Python loop): the log
increments (or multiplicative factors) for every step are computed in one array
operation and then combined with `numpy.cumsum` / `numpy.cumprod` along the
time axis.

### 2.2 Payoffs

`src/payoffs.py` turns a path array `(n_paths, n_steps+1)` into an *undiscounted*
payoff vector `(n_paths,)`; every estimator applies the `exp(-rT)` discount
itself, so payoffs stay reusable across scenarios with different `r`.

- `barrier_payoff(paths, K, B, kind, option)` — **discrete monitoring**: the
  barrier condition is checked at every simulated grid point (`paths.max(1) >= B`
  for an up-barrier), not via a continuity correction. This is intentionally
  different from the paper's closed-form benchmark, which assumes continuous
  monitoring — the gap between the two is the discretization bias discussed in
  §3.5 below and revisited with variance reduction in Stage 3.
- `geometric_asian_payoff(paths, K, option, avg_start_idx)` — geometric average
  of `paths[:, avg_start_idx:]`. `avg_start_idx = 1` averages all 252 monitoring
  dates (excluding `S0`); the last-30-day scenario sets
  `avg_start_idx = n_steps - 29` so only the final 30 grid points are averaged.
- `payoff_for(scenario, option=None)` — a dispatcher keyed on
  `scenario.exotic_type` that estimators call to get the right payoff function
  for a given scenario without a big if/else at every call site. For
  `arithmetic_asian`, it looks up `arithmetic_asian_payoff` **by name at call
  time** rather than importing it directly — Stage 3 appends that one function
  to `payoffs.py` (the project's single sanctioned shared-file edit) and the
  dispatcher picks it up automatically, with no other change needed.

### 2.3 Estimators

`src/estimators.py` defines the common return type used by every method in the
project:

```python
@dataclass
class EstimateResult:
    price: float; std_error: float; ci_low: float; ci_high: float
    runtime_sec: float; n_paths: int; extra: dict
```

`plain_mc` is crude Monte Carlo: `n_paths` iid paths, mean discounted payoff,
`std_error = sample_std / sqrt(n_paths)`, 95% CI = `price ± 1.96*std_error`.

`antithetic_mc` treats `n_paths` as the *total* number of simulated paths, i.e.
`n_paths/2` antithetic **pairs** driven by `[Z, -Z]`. The estimator computed
naively from all `n_paths` correlated payoffs would understate its own
variance, so the standard error is instead computed from the `n_paths/2`
**pair averages** `(Y_i + Y_i')/2`, which *are* i.i.d. across pairs. `extra`
records the pair count and the empirical correlation `rho_pair = corr(Y, Y')`
between the two antithetic legs — a negative `rho_pair` is what produces
variance reduction. For payoffs that are monotone in the driving normals,
which covers every payoff in this project, `rho_pair ≤ 0` is guaranteed, so
antithetic can never *increase* variance at equal path count. How large the
gain is depends on how close to linear the payoff is (§3.3).

Both estimators generate paths in chunks (default 8192 paths per block, see
§2.5) and time only the estimation loop with `time.perf_counter()`.

**Seeding.** `make_seed_seq(experiment_id, *keys)` derives a deterministic
`np.random.SeedSequence` from the shared `BASE_SEED = 402`, keyed by a CRC32 of
the experiment id plus integer keys. `run_sweep` passes CRC32s of the scenario
name, option type and method name, then `n_paths` and the replicate id. The keys
are built from names, not list positions, so two separate `run_sweep` calls can
never collide. They use CRC32 rather than Python's `hash()`, which is salted per
process and would make runs irreproducible. This is never `np.random.seed`.
Every replicate gets its own independent, reproducible stream, and the same key
always reproduces the same run; `tests/test_estimators.py` checks this across
two processes with different hash salts. `seed_int()` folds a `SeedSequence` down to one integer for the
`seed` column in the results CSV.

### 2.4 Benchmark infrastructure

`src/benchmark.py` is deliberately generic:

- `ESTIMATORS` is a registry (`{name: callable}`), populated with `"plain"` and
  `"antithetic"` at import time. Stages 3–5 call
  `register("control_variate")(fn)` (etc.) from their own modules to add their
  estimator — `run_sweep` never hard-codes the method list, so nothing here
  needs to change when a new technique is added.
- `exact_price(scenario, option, n_steps)` is the analytic benchmark used for
  bias/RMSE/coverage. For barriers it calls Stage 1's continuous
  Reiner–Rubinstein formula. For geometric Asians it uses a **general discrete
  Kemna–Vorst-style formula over the scenario's actual averaging window**
  (`avg_start_idx .. n_steps`, not just full averaging): since
  `log S_{t_i} = log S_0 + (r - sigma^2/2) t_i + sigma W_{t_i}`, the log of the
  geometric average is itself normal with mean
  `log S_0 + (r - sigma^2/2)*mean(t_i)` and variance
  `sigma^2 * mean_ij[min(t_i, t_j)] `, which reduces to a Black-76-style
  lognormal option price. At full averaging this is checked (in
  `tests/test_paths.py`) to reproduce Stage 1's `geometric_asian_closed_form(m=252)`
  exactly; it is also the *only* correct benchmark for `asian_30d`, since
  Stage 1's formula only covers full-window averaging (see §3.4 — the base
  paper's claim that the 30-day price is "identical" to the full-average price
  is false and easily checked). For arithmetic Asians it returns `None` (no
  closed form — that is the whole point of Stage 3's control-variate headline
  experiment).
- `run_sweep(person, experiment_id, cases, methods, n_grid, R, n_steps, schemes,
  method_params)` loops `case × scheme × method × N × replicate`, builds a
  fresh `SeedSequence` per cell, calls the registered estimator, and logs one
  row per replicate via Stage 1's `log_result`.
  - The discretization **scheme is deliberately left out of the seed key**. All
    schemes therefore see the same normals, and scheme comparisons are paired
    (common random numbers).
  - Different methods, scenarios and N values get independent streams, so a
    comparison between two methods is a comparison of independent samples. The
    bootstrap below relies on that.
- `summarize(df)` aggregates raw per-replicate rows into one row per
  `(scenario, option, scheme, method, n_paths)` cell: mean price, **variance
  measured across the `R` replicates** (never taken from a single estimator's
  own reported `std_error`), bias against `exact_price`, `RMSE = sqrt(bias^2 +
  variance)`, mean runtime, `efficiency = 1/(variance * mean_runtime)`, and CI
  coverage (fraction of replicates whose 95% CI contains the exact price).
- `bootstrap_efficiency_ratio(df, keys_a, keys_b)` bootstraps the ratio of two
  methods' efficiencies by resampling replicates with replacement within each
  matched cell (default 2000 resamples). It returns a point estimate and a 95%
  percentile CI. `R = 20` replicates give roughly 32% relative standard error on
  a variance estimate (§2.7, criterion 3), so a bare point-estimate speedup means
  little.
  - Each side must select **exactly one** `(scenario, option, scheme, method,
    n_paths)` cell; otherwise the function raises. Pooling replicates across N
    mixes variance scales and produces a meaningless ratio. An early version of
    our own baseline analysis made this mistake before review caught it.
  - For estimators whose within-run SE is valid (plain and antithetic, both
    built on i.i.d. units), the ratio of mean squared SEs is a far more precise
    variance-reduction estimate than the 20-replicate bootstrap. §3.3 reports
    both.
  - RQMC is different: its within-run SE is not valid, and only the
    across-replicate variance can be used.
- `fit_loglog_slope(n, y)` is a small helper used throughout for RMSE-vs-N and
  bias-vs-step-size log-log fits.

### 2.5 Compute considerations

Path generation is done in chunks of at most 8192 paths (`estimators.py`'s
`DEFAULT_CHUNK`) so that the largest grid point in this project, `N = 65536`
paths at 252 steps, never requires allocating a single
`(65536, 253)` array (~132 MB of floats, times several intermediate arrays);
antithetic chunks in half-sized blocks of `Z` before mirroring. All timings in
this stage were produced with `OMP_NUM_THREADS=1` set before importing NumPy,
for comparability with later stages (by the group protocol, the *authoritative*
timings for the final efficiency table come from Stage 5's single-machine
master run in Chapter 7; this stage's numbers are indicative).

### 2.6 Shared plotting style

`src/plots.py` centralizes the look of every figure downstream: a fixed
color/label per method (`plain`, `antithetic`, `control_variate`, `rqmc`,
`importance`) so the same method always has the same color across every
stage's plots, plus two reusable plot builders:

- `ci_vs_n_plot` — the paper's Figure 2–4 style: estimate ± 95% CI vs. `N` on a
  log-*x* axis, one (slightly horizontally offset) series per method, with a
  dashed line for the closed-form value.
- `loglog_rmse_plot` — RMSE vs. `N` on log-log axes, with the fitted slope
  reported in each series' legend entry and an `N^-1/2` reference line for
  comparison.

### 2.7 Acceptance criteria

Every chapter validates its estimators against the same four criteria, fixed
before any experiment was run:

1. **Unbiasedness.** Wherever a reference price exists, every estimator lands
   within **3 standard errors** of it.
2. **CI coverage.** Across the R replicates, the 95% confidence interval
   contains the reference price about 95% of the time. With 20 replicates per
   cell a single cell's coverage is noisy, so coverage is judged over many
   intervals (pooled over N, or over cells).
3. **Efficiency with uncertainty.** Every speedup is reported as
   `efficiency = 1/(variance × time)` with a bootstrap CI. With R = 20, a
   variance estimate has roughly 32% relative standard error, so a bare
   variance-reduction number means little on its own.
4. **RMSE, not variance alone.** A biased low-variance estimator scores well
   on VRF and is still wrong, so we report `RMSE = sqrt(bias² + variance)`.
   This matters for the discretely monitored barrier, whose continuous-formula
   benchmark is off by the monitoring bias (§3.5, §5.3).

RQMC has no valid within-run standard error. For RQMC (and, from Chapter 5 on,
for every method), criterion 1 is also checked with a t-interval over the R
replicate estimates, and criterion 2 is reported only for methods with a valid
within-run SE.

---

## 3. Baseline Replication

*(Stage 2 — Nafis Nahian, 2105007)*

This section runs the engine and estimators from §2 on the base paper's own
configuration: the up-and-in barrier call and the geometric-average Asian
call/put. It then checks the paper's claims one by one.

- Experiment script: `experiments/exp_baseline.py`.
- Raw output: `results/raw/nafis_baseline.csv`, 1200 rows:
  - `paper_barrier` × 3 schemes × {plain, antithetic} × 5 N × 20 reps;
  - {`paper_asian_geo` call, `paper_asian_geo` put, `asian_30d`} × exact scheme × {plain, antithetic} × 5 N × 20 reps.
- Derived tables:
  - `results/tables/baseline_summary.csv` (per-cell statistics);
  - `results/tables/baseline_paper_claims.md` (every number quoted in §3.4);
  - `results/tables/coverage_table.csv`.

### 3.1 Setup

All scenarios use `S0=100, r=0.03, sigma=0.2, T=1.0`, `N_STEPS=252` (daily
monitoring), `N_GRID=[256, 1024, 4096, 16384, 65536]`, and `R=20` replications
per cell, seeded from `SeedSequence(402)` (§2.3).

- The barrier scenario (`K=105, B=110.6772`) is run under all three
  discretization schemes. The schemes share random numbers, so the comparison
  between them is paired.
- The Asian scenarios are run under the `exact` scheme: `paper_asian_geo` call
  and put with `K=105`, and `asian_30d`, which averages only the final 30
  trading days (monitoring dates).
- The paper states no Asian parameters. We use the barrier's `S0, r, sigma, T`
  and `K=105`, which matches the strike line in its Fig. 5. "Final 30 days" is
  our reading of the paper's "final 30 days of the contract" (see Claim 4).

Two high-precision reference runs back the sweep:

1. **The 252-step discretely monitored barrier price**, computed by in-out
   parity as `BS call − E[discrete up-and-out]` with 4M paths. The up-and-out
   payoff is small and has low variance, so the result is precise to ±0.0002.
2. **Scheme-minus-exact price differences** with common random numbers, 1M
   paths.

### 3.2 Convergence, coverage and validation

**Figures:** `baseline_barrier_ci.png`, `baseline_asian_call_ci.png`,
`baseline_asian_put_ci.png`, `baseline_rmse_vs_n.png`, `coverage_table.png`,
`baseline_scheme_comparison.png`.

**CI-vs-N.** Every CI-vs-N plot straddles the benchmark at every N. The CIs
shrink as N grows, and the antithetic CIs are visibly narrower.

**RMSE-vs-N.** The fitted log-log slopes lie between **−0.45 and −0.52** across
all eight scenario/method combinations. This matches the theoretical `N^-1/2`
Monte Carlo rate. The scatter around −0.5 comes from estimating each RMSE with
only R=20 replicates.

**CI coverage.** The pooled coverage over all five N values (100 intervals per
cell) is **91%–98%**. With 100 Bernoulli(0.95) trials, the sampling spread is
about ±4 points, so every cell is consistent with the nominal 95%.

**3-SE check.** Between 98% and 100% of replicates fall within 3 SE of the
closed form. **Acceptance criteria 1 and 2 (§2.7) are met.**

**Barrier benchmark.** For the barrier, "closed form" means the *continuous*
Reiner–Rubinstein value, 7.1055. That is the only price the paper states, and it
matches this formula at T = 1. Our payoff is monitored discretely, and its true
price is 7.0941 (§3.5). The 0.011 gap is far below
the per-run SE at every N in the grid (≥0.049), so coverage is unaffected.

### 3.3 Antithetic variates

Our antithetic estimator computes its SE from the N/2 i.i.d. pair averages.
Across replicates, the squared SE matches the observed variance of the price
estimates within the ±32% sampling error of a 20-replicate variance. At
N=65536:

| Option | Method | mean SE² | across-replicate variance |
|---|---|---|---|
| Asian put | antithetic | 0.00022 | 0.00019 |
| Asian call | antithetic | 0.00039 | 0.00036 |
| Barrier | antithetic | 0.00163 | 0.00223 |
| Barrier | plain | 0.00240 | 0.00157 |

The within-run variance reduction factor (VRF) is therefore a valid, precise
measure.

The table compares antithetic and plain at equal total path count:

| Option | corr(Y, Y′) | VRF (flat across all 5 N) | time ratio plain/anti | efficiency ratio |
|---|---|---|---|---|
| Barrier call | −0.32 | 1.47 | 1.21 | 1.77 |
| Geometric Asian call | −0.26 | 1.35 | 1.17 | 1.58 |
| Geometric Asian put | −0.74 | 3.91 | 1.16 | 4.54 |
| 30-day Asian call | −0.32 | 1.48 | 1.20 | 1.78 |

- **Why every VRF exceeds 1.** All four payoffs are monotone in the driving
  normals, which guarantees `corr(Y, Y′) ≤ 0` and hence VRF ≥ 1.
- **Why the put gains the most.** The put is in the money, so its payoff is
  close to linear in the path, and antithetic variates cancel linear
  components exactly.
- **Why the barrier and 30-day call behave alike.** Both are close to a vanilla
  call: the up-and-out leg is worth only 0.02 under continuous monitoring (0.034
  at 252 dates), and a 30-day average is nearly the terminal price.
- **Why the efficiency gain exceeds the VRF here.** Antithetic draws half as
  many normals per path, so on this machine it was ~1.2× cheaper per path.

**These timings are indicative.** In the authoritative, interleaved master
sweep (Chapter 7) antithetic costs about the same as plain MC on these options
(time ratio 0.94–0.99). Its efficiency gain there is therefore about its
variance reduction factor: 1.5 (barrier) and 1.4 (geometric Asian) from the
within-run SEs. The VRFs above are timing-free and are the robust result.

**The across-replicate bootstrap is too noisy to use here.** With R=20, the
bootstrap efficiency ratio required by criterion 3 (§2.7) is very noisy: at N=65536 it is
0.85 [0.38, 1.91] for the barrier. That run happened to draw a low plain
variance and a high antithetic variance, as the table above shows. A ratio of
two 20-sample variances has a sampling range of roughly [0.4, 2.5]×. That test cannot tell 1.3 from 1.5, so
we rely on the within-run VRF, where each replicate contributes up to 32,768
i.i.d. pairs.

### 3.4 Testing the paper's claims

Each verdict below is phrased to be no stronger than the evidence. Two facts
about the paper frame all four: it contains **no equations** (schemes,
closed forms and estimators are only named), and it reports **no Monte Carlo
estimates, standard errors or timings as numbers**. Its claims can only be
tested by rebuilding its setup, which is what this chapter does.

**Claim 1: "convergence stabilization at approximately 5000 simulations for
Barrier options and 750-1000 for Asian options". Verdict: not supported as a
convergence property.**

- **No plateau.** `RMSE·√N` stays roughly constant across the grid (barrier:
  9.5–14.3; Asian call: 3.9–6.8, with R=20 noise), and the fitted slopes are
  −0.51 and −0.50. The error keeps falling at the ordinary `N^-1/2` rate, with
  no threshold near the paper's values.
- **Still wide at the paper's thresholds.** At the paper's "stabilized" sample
  sizes, the 95% CI half-width is still **±0.35 (±4.9%)** for the barrier at
  N=5000 and **±0.36–0.42 (±12–14%)** for the Asian call at N=750–1000.
- **What ±1% would take.** About 120,000 paths for the barrier and 146,000 for
  the Asian call.
- **What the paper's statement describes.** It reads as a description of how
  its log-x CI plots look: a `1/√N` band appears to flatten on a log axis.

The paper gives no quantitative criterion, so we cannot call the claim
"false". What we can say is that it is not a property of the estimator.

**Claim 2: antithetic "accelerates" convergence 1.5× (barrier) and 1.3× (Asian).
Verdict: confirmed for the barrier and the Asian call; the put result does not
match.**

- **Barrier and Asian call.** The paper never defines its factor. Read as a
  variance ratio at equal path count, our precise VRFs of **1.47** and
  **1.35** agree with 1.5 and 1.3.
- **Asian put.** The paper gives one factor, 1.3, for "Asian options", and no
  separate put number. Our put VRF is **3.91**, far from 1.3. The paper's put
  figure (Fig. 8) appears identical to its call figure (Fig. 6): same points,
  and an exact-value line at ≈3.0, whereas the put's price is 6.708.[^put] Its
  text also cites Fig. 7 for the put's convergence, but Fig. 7 is a volatility
  sensitivity plot. The put result was probably never measured separately.
- **Computational time.** The paper reports "increased computational time" for
  antithetic but gives no numbers. In our implementation antithetic is not
  more expensive per path: in the master sweep it costs the same as plain MC
  (§3.3), so its efficiency gain equals its VRF.

[^put]: The discrete geometric put is 6.70776 and the 30-day-window call in
Claim 4 is 6.70784. Both round to 6.708 by coincidence; they are different
options with different formulas.

**Claim 3: the discretization scheme (Euler / Euler–Maruyama / Milstein) has a
negligible effect. Verdict: confirmed, and expected from theory.**

- **Measured effect.** With common random numbers and 1M paths, the price
  differences from the exact scheme are:
  - Euler–Maruyama: **−0.0006 ± 0.0002**
  - Milstein: **−0.0011 ± 0.00001**

  These are real O(dt) weak-discretization effects. They are about 0.015% of
  the price and ~50× smaller than the SE of a single 65,536-path run.
- **Why this is expected.** Both schemes have *weak* order 1. Milstein's
  advantage is in *strong* (pathwise) order, 0.5 → 1.0, which does not improve
  the accuracy of an expectation. Our tests confirm the strong-order behaviour
  directly: Euler's pathwise error shrinks at slope ≈0.5, and Milstein's
  pathwise error is under 5% of Euler's.
- **Two caveats about the paper's comparison.** The paper never writes down its
  schemes. It is therefore unclear how its "Euler" differs from
  "Euler–Maruyama". Under the standard definitions they are the same scheme;
  alternatively "Euler" may mean Euler on log S, which is the exact scheme for
  GBM (its path plots are titled "Euler discretisation random walks"). Also, its
  Figs. 2–4 show identical sample points. Either the same random numbers were
  reused across schemes or the same figure was used three times; we cannot tell
  which. With a weak effect of ~0.001 at daily steps, plots with shared random
  numbers would look identical anyway. The paper's step count is not stated,
  and with coarse steps the effect would be larger. Chapter 5's strong/weak
  order study (§5.7–5.8) is the correct way to compare schemes.

**Claim 4: for the 30-day averaging window, "the exact closed-form solution
remains identical to the previous examples". Verdict: false, if read as the
same value.**

- **The correct benchmark.** Our general discrete-window formula (§2.4) gives
  **6.708** for the 30-day-window call, against **3.000** for full-window
  averaging and 7.128 for the vanilla call.
- **The window is ambiguous, the verdict is not.** The paper says "the final
  30 days of the contract"; we read that as 30 trading days (monitoring dates).
  Its Fig. 9 shades about 30 calendar days, roughly 21 trading days, which gives
  6.838. Every reading gives more than twice the full-window price.
- **Monte Carlo agrees.** Antithetic MC at 65,536 paths gives 6.698 ± 0.007,
  and an independent ad-hoc 4M-path check (not part of the scripts) gives
  6.7067 ± 0.0049.
- **Why the price is so much higher.** A 30-day average removes far less
  variance than a 252-day average, so the option is worth more than twice as
  much.
- **The paper's own evidence.** The paper shows no number for this case (its
  Fig. 9 shows only paths).

### 3.5 A correction to our own reference values

Our initial project plan gave the 252-step discretely monitored up-and-in
price as ≈7.076, a bias of ≈ −0.03. Our in-out-parity reference, with SE 0.0002,
gives **7.0941**:

- the discretization bias is **−0.0114**, not −0.03;
- the BGK-corrected value, 7.0930, is accurate to 0.0011.

Two independent ad-hoc checks, not part of the scripts, agree: a 4M-path
antithetic run gives 7.0914 ± 0.0052, and a separately written 2M-path parity
run gives 7.0945 ± 0.0002. The later stages confirm it with their own
estimators: 7.093583 from the Stage-3 control variate (§4.4) and
7.094133 ± 0.000096 from 64 RQMC scrambles (§5.3).

The qualitative point of the plan still holds, and more strongly: the bias is
invisible to plain Monte Carlo at practical N.

- plain-MC SE at N=1024 is 0.39, **about 34× the bias**;
- plain MC needs roughly `(12.5 / 0.011)² ≈ 1.3M` paths before the bias
  equals one SE.

Chapter 4 resolves this bias with the control variate, and Chapter 5 uses the
64-scramble RQMC value as the reference for every barrier RMSE.

### 3.6 Summary

| Acceptance criterion (§2.7) | Result |
|---|---|
| Estimates within 3 SE of the closed form | ✅ 98–100% of replicates |
| CI coverage ≈95% | ✅ 91–98% per cell (n=100) |
| Efficiency reported with a bootstrap CI | ✅ reported; shown to be too noisy at R=20, so a precise within-run VRF is reported alongside it |
| RMSE rather than variance alone | ✅ slopes −0.45 to −0.52 |

| Paper claim | Our verdict |
|---|---|
| "Stabilizes" at 5000 / 750–1000 paths | Not a convergence property: error keeps falling as `N^-1/2`, and the CIs are still ±5% / ±12% wide at those N |
| Antithetic 1.5× barrier / 1.3× Asian | Confirmed as a variance ratio (1.47 / 1.35); the Asian put (3.9×) does not match, and its figure appears to duplicate the call's |
| Scheme choice negligible | Confirmed (effect ≈0.001 at daily steps), but it follows from theory (weak order 1), and the paper's three figures are identical |
| 30-day Asian has "identical" closed form | False as a value: 6.708 vs 3.000 (6.84 under the calendar-day reading) |

---

## 4. Control Variates, Arithmetic Asian Options, and Barrier Bias

*(Stage 3 — Aritra Debnath, 2105010)*

This stage adds a control-variate estimator and uses it for two purposes. First,
it prices an arithmetic-average Asian call, for which no closed-form price is
available, using the closely related geometric-average Asian call as the
control. Second, it shows why a control that is extremely effective for one
barrier level can be weak at another, then uses the variance reduction to make
the small discrete-monitoring bias statistically visible.

- Estimator: `src/vr_control.py`.
- Main experiment: `experiments/exp_control.py`.
- Barrier-bias experiment: `experiments/exp_barrier_bias.py`.
- Raw outputs: `results/raw/aritra_control.csv` (601 rows) and
  `results/raw/aritra_barrier_bias.csv` (60 rows).
- Derived tables: `results/tables/control_summary.csv` and
  `results/tables/barrier_bias_summary.csv`.

### 4.1 Estimator and implementation

For a discounted target payoff `Y`, a discounted control payoff `X`, and a
known control value `mu_X = E[X]`, the estimator is

```
V_CV = (1/N) * sum_i [ Y_i − b_hat * (X_i − mu_X) ],      b* = Cov(Y, X) / Var(X)
```

At the optimal coefficient, its theoretical variance-reduction factor is

```
VRF = Var(Y) / Var(Y − b*(X − mu_X)) = 1 / (1 − rho_YX²)
```

`control_variate_mc` fits `b_hat` on an **independent pilot sample** and then
applies it to the production sample. The default pilot size is
`max(256, ceil(0.1*N))`. Production paths still define the shared `n_paths`
axis; pilot paths are additional work, included in the measured runtime and
recorded as `pilot_paths` and `total_paths_simulated` in `extra_json`. Paths are
generated in 8192-path chunks, preserving Stage 2's bounded-memory convention.

The two control definitions are:

| Target | Control | Known control expectation |
|---|---|---|
| Arithmetic-average Asian call | Discrete geometric-average Asian call on the same path | Discrete Kemna–Vorst-style price from `benchmark.exact_price` |
| Up-and-in barrier call | European terminal call on the same path | Black–Scholes call price |

The standard error and confidence interval are computed from the i.i.d.
adjusted production samples. Each raw row also records `rho`, `b_hat`, the
uncontrolled SE implied by the same target samples, the observed within-run
VRF, and the theoretical `1/(1 − rho²)`.

Before using option payoffs, the coefficient calculation was tested on
synthetic correlated normals, `Y = 1.7 X + 0.55 ε`. The fitted coefficient
recovers 1.7, and both the measured and correlation-predicted VRFs recover the
known value `1 + 1.7²/0.55²`. This test is retained in `tests/test_control.py`.

Both control means are exact-GBM prices. The estimator is therefore exactly
unbiased only under the `exact` path scheme, which every experiment uses. Under
Euler–Maruyama or Milstein the control's own O(Δt) discretization bias would
leak into the estimate (§8.4).

### 4.2 Arithmetic Asian: the headline result

The arithmetic payoff was added as the mean of the 252 simulated monitoring
prices (excluding `S_0`), followed by the usual call payoff. Plain MC and the
control estimator were each run at `N = [256, 1024, 4096, 16384, 65536]`, with
`R = 20` independent replications per cell. Because the arithmetic option has
no analytic benchmark, an additional 2,000,000-path plain-MC run provides an
independent validation value.

At `N = 65,536`:

| Quantity | Result |
|---|---:|
| Control-variate price, mean of 20 replications | **3.162776 ± 0.000361** (95% CI for the replicated mean) |
| 2,000,000-path plain-MC reference | **3.162517 ± 0.008511** (95% CI) |
| Difference in combined standard errors | **0.06** |
| Correlation `rho(Y, X)` | **0.999488** |
| Pilot coefficient `b_hat` | **1.045681** |
| Observed within-run VRF | **975.7×** |
| Theoretical `1/(1 − rho²)` | **976.6×** |

Thus the geometric control removes about 99.90% of the target payoff variance.
The independently estimated arithmetic price agrees with the much larger plain
run, satisfying the validation requirement without pretending that the
arithmetic option has a closed form. The plain reference can only detect a
bias above about 0.01; the sharper check comes later. Against the 64-scramble
RQMC reference of Chapter 7 (3.162981 ± 0.000064), the control-variate estimate
passes the t-test at every N (§7.7). `cv_asian_vrf.png` shows that the gain is
stable at approximately 10³ across the complete N grid.

### 4.3 Why the barrier control succeeds—and then collapses

The same European-call control was applied to two up-and-in barriers. Results
at `N = 65,536` are:

| Scenario | CV price | rho | b_hat | observed VRF | theory | across-replicate VRF |
|---|---:|---:|---:|---:|---:|---:|
| Paper barrier, `B = 110.6772` | 7.094189 | 0.999675 | 1.000879 | **1542.4×** | 1542.4× | 1365.6× |
| Deep barrier, `B = 140` | 3.379102 | 0.830485 | 0.763175 | **3.2×** | 3.2× | 5.7× |

For the paper barrier, the continuously monitored up-and-out leg is worth only
about 0.023 (0.034 at 252 monitoring dates). The up-and-in payoff therefore
almost equals the terminal vanilla call path by path, giving `b_hat ≈ 1` and
near-perfect correlation. At `B = 140`, many positive vanilla-call payoffs do
not knock in; the two payoffs no longer track each other closely, so the VRF
collapses to about 3.2. This is the central pedagogical result in
`cv_barrier_collapse.png`: a control variate is not intrinsically good or
bad—its value depends on its correlation with the target under the specific
scenario.

`cv_rho_vs_vrf.png` compares every scenario/N cell with the theoretical curve.
The observed within-run factors closely follow `1/(1 − rho²)`, including the
sharp growth as `rho → 1`. The across-replicate ratios fluctuate more because
each cell contains only 20 price estimates.

**Efficiency: use Chapter 7.** At the largest N, this stage's bootstrap
efficiency ratios (control over plain, including pilot runtime) are 405.5
[169.6, 991.8] for the arithmetic Asian, 3506.8 [1691.5, 7047.1] for the paper
barrier, and 5.1 [1.8, 14.8] for the deep barrier. These Stage-3 timings are
not reliable. `run_sweep` runs each method as one uninterrupted block, so any
slowdown of the machine lands on one method. On the paper barrier the plain
runs' timings were bimodal (0.33–1.49 s at N = 65,536). That made the control
variate look 2.6× *cheaper* than plain MC, which is impossible, since it does
strictly more work. The authoritative, interleaved master sweep gives
**770×, 1,602× and 4.2×** for the same three cells (§7.2). The VRFs above are
timing-free and are unaffected.

### 4.4 Discrete barrier-monitoring bias

The continuous Reiner–Rubinstein paper value is 7.105528, whereas a barrier
observed only on a grid is less likely to be hit. The control estimator was run
with `N = 65,536`, `R = 20`, and 12, 52, or 252 monitoring dates. The BGK
continuity correction shifts an up-barrier outward to
`B * exp(0.5826 σ sqrt(Δt))` before applying the continuous formula.

| Monitoring dates | CV price | measured bias vs continuous | BGK price | MC − BGK | CV SE per run | implied plain SE per run |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 7.042920 | −0.062608 | 7.008629 | +0.034291 | 0.002086 | 0.049044 |
| 52 | 7.078373 | −0.027155 | 7.072275 | +0.006098 | 0.001556 | 0.049047 |
| 252 | **7.093583** | **−0.011945** | **7.092995** | **+0.000588** | **0.001260** | **0.048851** |

At daily monitoring the result agrees with Stage 2's independent
in-out-parity reference of approximately 7.0941 and with the 64-scramble RQMC
reference 7.094133 ± 0.000096 used from Chapter 5 on. The replicated mean
resolves the −0.01195 bias at 54.1 standard errors (the SE of the 20-replicate
mean, 0.000221), while even one 65,536-path plain run has an SE about four
times larger than the bias. This is precisely the feature plain MC obscures.

`barrier_bias_vs_steps.png` also shows that BGK is highly accurate at 252 dates
but less accurate on coarse grids; it is an asymptotic continuity correction,
not an exact discrete-barrier formula. At 252 dates the BGK error is +0.0006
against this stage's CV estimate and −0.0011 against the more precise RQMC
reference. Either way it is 40–80× smaller than a plain run's SE.

The three step counts share seeds replicate by replicate (the sweep's seed key
does not include `n_steps`). Each level's mean is unbiased, but the levels are
weakly correlated, so only per-level results are reported.

Our initial project plan quoted about 7.076 (bias −0.03) for the daily barrier.
Stage 2's high-precision result superseded that value, and it is not used here.
The Stage-3 estimate independently confirms the corrected daily-monitoring bias
of about −0.012.

### 4.5 Finite-sample coefficient note

Estimating the optimal coefficient from the same sample used in the adjusted
mean can introduce an O(1/N) finite-sample bias through the random ratio
`sample_covariance/sample_variance`. At the sample sizes used here it would be
negligible relative to Monte Carlo error. Our implementation goes further: it
fits `b_hat` on an independent pilot, so conditional on that coefficient
`E[X − mu_X] = 0` in the production sample and the adjusted estimator remains
unbiased. The tradeoff is extra pilot cost and a small variance penalty if the
pilot coefficient differs from `b*`; both are included in the logged data.

### 4.6 Acceptance summary

| Requirement | Evidence |
|---|---|
| Coefficient/VRF smoke test | Synthetic-normal test recovers known `b*` and VRF |
| Arithmetic option without closed form | 3.162776; agrees with 2M-path plain reference within 0.06 combined SE, and with the RQMC reference in every master-sweep cell |
| Near/deep barrier contrast | VRF 1542.4× versus 3.2× |
| Theory versus observed reduction | All cells follow `1/(1 − rho²)` |
| Resolve daily barrier bias | −0.011945, measured at 54.1 SE of the replicated mean |
| BGK application | 252-step BGK within 0.0006–0.0011 of the discrete price |
| Reproducibility | 20 raw rows per frozen grid cell; deterministic `SeedSequence(402)` hierarchy; pilot metadata retained |

---

## 5. Convergence Analysis: Quasi-Monte Carlo and Discretization Order

*(Stage 4 — Zaki Rehnoom Unmona, 2105016)*

A Monte Carlo price has two independent error sources:

- **sampling error**, which shrinks with the number of paths N;
- **discretization error**, which shrinks with the time step Δt.

This chapter measures the *rate* of each. For sampling error we replace
pseudo-random numbers with randomized quasi-Monte Carlo (RQMC) and ask how much
faster than `N^-1/2` the error falls. For discretization error we measure the
strong and weak order of the Euler–Maruyama and Milstein schemes. The base paper
compared those two schemes, but it compared them in a way that could not show a
difference (§5.8).

- Estimator: `src/vr_qmc.py`.
- Experiments:
  - `experiments/exp_qmc.py`;
  - `experiments/exp_scheme_order.py`.
- Tests: `tests/test_qmc.py` (30 tests, including a smoke test on integrals with known values).
- Raw outputs:
  - `results/raw/zaki_qmc.csv` (1028 rows: 900 sweep rows and 128 reference rows);
  - `results/raw/zaki_scheme_order.csv` (1600 rows).
- Derived tables:
  - `results/tables/qmc_summary.csv` and `qmc_findings.md`;
  - `results/tables/scheme_order_summary.csv` and `scheme_order_findings.md`.

### 5.1 Why quasi-Monte Carlo can beat `N^-1/2`

Plain MC places points at random, so its error falls as `σ/√N` in any
dimension. A **low-discrepancy** point set fills the unit cube more evenly than
random points do. The Koksma–Hlawka inequality bounds the integration error by
`V(f) · D*(P_N)`, where:

- `V(f)` is the variation of the integrand;
- `D*(P_N) = O((log N)^d / N)` is the star discrepancy of the point set.

That promises an error close to `N^-1`. Two caveats matter for option pricing:

1. **Dimension.** One path of 252 daily steps is one point in `d = 252`
   dimensions. There `(log N)^d` makes the bound useless. QMC still works well
   when the integrand has **low effective dimension**, meaning most of its
   variance depends on a few coordinates. The *path construction* decides which
   coordinates those are (§5.5–5.6).
2. **Smoothness.** The bound needs `V(f) < ∞`. The fast rates also need
   smoothness. A kink (the call payoff `max(·,0)`) slows convergence, and a jump
   whose discontinuity is not aligned with the coordinate axes slows it
   further. The knock-in indicator `1{max S ≥ B}` is such a jump.

**Randomization.** Owen scrambling (Owen, 1997) randomly permutes the digits of
a Sobol net. It keeps the net's balance, and it makes every point uniformly
distributed, so each scrambled estimate is **unbiased**. Independent scrambles
then give independent estimates. The **R = 20 independent scrambles are the
only valid source of error bars**. The points inside one scramble are not
independent, so the usual `s/√N` computed from a single run means nothing for
RQMC.

### 5.2 Implementation

**Point generation.** `sobol_uniforms(n_paths, dim, rng)` does four things:

1. It draws `qmc.Sobol(d=252, scramble=True, seed=rng).random_base2(m)` with
   `N = 2^m`.
2. It rejects any N that is not a power of 2: a Sobol net is balanced only at
   powers of 2, and other sizes silently degrade it towards the MC rate.
3. It maps uniforms to normals only through the inverse CDF `ndtri`. Box–Muller
   would mix coordinates and destroy the low-discrepancy structure.
4. It moves every point to the centre of its dyadic cell before `ndtri`.
   scipy's Sobol points are multiples of `2^-30`. After the random digital
   shift, an exact 0 has probability `2^-30` per coordinate, and `ndtri(0) = −∞`.
   A full sweep draws about 4×10⁸ coordinates per case, so the shift is needed.

**Path construction.** The Stage-2 engine takes an `(N, 252)` array of per-step
normals. It was left unchanged: a construction is just a map from Sobol
dimension k to those normals.

| Construction | Dimension 0 drives | Map |
|---|---|---|
| Incremental | step 1 | identity: dimension j → step j |
| **Brownian bridge** | the terminal value `W_T` | `normals = z @ M` |

Under the bridge, dimension 1 sets the midpoint `W_{T/2}` conditional on
`W_T`. The remaining dimensions fill successive midpoints breadth first, using
the conditional mean `((t_r − t)W_l + (t − t_l)W_r)/(t_r − t_l)` and standard
deviation `√((t − t_l)(t_r − t)/(t_r − t_l))`.

`bridge_matrix(252)` builds the bridge once as a 252×252 matrix and caches it.
It is **orthogonal**, so i.i.d. normal inputs give i.i.d. normal steps, and the
path law is exactly that of the incremental scheme. Only the assignment of
input dimensions to path features changes. The tests check three properties:

- orthogonality for n = 1, 2, 3, 7, 64 and 252;
- dimension 0 alone determines `W_T = √n · z₀`;
- the midpoint depends only on dimensions 0 and 1.

**Estimator.** `rqmc_mc(...)` returns one estimate from one scramble; the
scramble is seeded by the `SeedSequence` that `run_sweep` passes in. Paths are
built and priced in 8192-path chunks, as in Stage 2.

- **Registered names.** It is registered twice: `"rqmc"` (bridge) and
  `"rqmc_incremental"`. `benchmark.summarize` groups rows by method, not by
  `method_params`, so a parameter switch would pool the two constructions into
  one cell.
- **Error fields.** `std_error` and the CI are logged as **NaN on purpose**.
  The naive i.i.d. SE is kept in `extra_json["naive_iid_se"]` for reference
  only.

**Task-0 smoke test** (`tests/test_qmc.py`). Two integrals with known answers
were computed before any option was priced. Both use 32 scrambles at
`N = 2^6 … 2^14`:

| Integrand | Exact value | RQMC RMSE slope | Plain-MC slope | RMSE ratio at N=16,384 |
|---|---:|---:|---:|---:|
| `E[exp(a·Z)]`, d=8 (smooth) | 1.404571 | **−0.83** | −0.43 | 28.8× |
| Gaussian orthant `P(X₁>0, X₂>0)`, ρ=0.5 (discontinuous, diagonal boundary) | 1/3 | **−0.77** | −0.50 | 17.8× |

Both slopes are clearly steeper than `N^-1/2`, as the smoke test requires.

### 5.3 Measuring the error correctly: the barrier reference

Acceptance criterion 4 (§2.7) requires RMSE rather than variance. The target matters as much
as the metric. `benchmark.exact_price` returns the *continuous* Reiner–Rubinstein
value for barriers. Our payoff is monitored daily, and Stages 2–3 showed that the
two differ by about 0.011 on the paper barrier. Once RQMC's error drops below
that, an RMSE computed against the continuous value measures the monitoring bias
instead of the estimator.

We therefore measure barrier RMSE against a **252-date reference**. It was
estimated from 64 extra bridge scrambles × 65,536 paths per barrier, on a seed
stream separate from the sweep.

| Scenario | 252-date reference | Continuous formula | BGK (252) |
|---|---:|---:|---:|
| Paper barrier (B=110.68) | **7.094133 ± 0.000096** | 7.105528 | 7.092995 |
| Deep barrier (B=140) | **3.385730 ± 0.001014** | 3.545321 | 3.377810 |
| Geometric Asian call | 3.000200 (discrete closed form) | — | — |

Cross-checks for the paper barrier:

- Stage 2's plain-MC in-out-parity value is 7.0941 ± 0.0002 (1 SE). The
  difference is 0.15 combined SE.
- Stage 3's control-variate value is 7.093583 ± 0.00028 (1 SE, including the
  spread of the replicates). The difference is 1.86 combined SE.

**The effect is large.** Measured against the continuous formula, the paper
barrier's RQMC RMSE-vs-N slope is **−0.26**; measured against the 252-date
reference it is **−0.80** (`qmc_vs_mc_barrier.png`, dotted curve). The
continuous benchmark would have produced a false "QMC barely helps on barriers"
result.

**The deep barrier's monitoring bias is 14× larger.** It is **−0.160** (4.5% of
the price), against −0.011 on the paper barrier. Only paths that cross the
barrier and then finish *below* it can be affected, because a path that
finishes above B is caught at the final monitoring date whatever the grid.
The continuous price splits as follows:

| Barrier | Paths finishing above B | Paths that crossed, finished below B | Bias as % of the exposed part |
|---|---:|---:|---:|
| B=110.68 | 6.861 (97%) | 0.245 (3%) | 5% |
| B=140 | 2.375 (67%) | **1.171 (33%)** | **14%** |

A 11× larger *share* of the deep barrier's value is exposed to missed crossings
(33% against 3%; 4.8× in absolute terms, 1.171 against 0.245), and it loses a
larger fraction of that exposed value.

BGK over-corrects on both barriers. It moves the price by 0.1675 where 0.1596
is needed, landing 0.008 (about 8 SE) below our deep-barrier reference; on the
paper barrier it lands 0.0011 below. This extends Stage 3's observation that
BGK is an asymptotic correction.

### 5.4 Headline result: RQMC vs plain MC

**Figures:** `qmc_vs_mc_asian.png`, `qmc_vs_mc_barrier.png`.

The sweep covers plain MC, `rqmc` and `rqmc_incremental`, on each scenario ×
`N_GRID` × R=20, with 252 steps and the exact scheme. Slopes are least-squares
fits of log RMSE on log N. Their 95% CIs come from 2000 bootstrap resamples of
the replicates within each N.

| Scenario | Plain MC | RQMC (bridge) | RQMC (incremental) |
|---|---|---|---|
| Geometric Asian call | −0.57 [−0.63, −0.48] | **−0.72 [−0.78, −0.65]** | −0.61 [−0.66, −0.54] |
| Paper barrier | −0.52 [−0.59, −0.44] | **−0.80 [−0.88, −0.71]** | −0.76 [−0.84, −0.70] |
| Deep barrier | −0.44 [−0.50, −0.38] | **−0.60 [−0.67, −0.52]** | −0.58 [−0.64, −0.51] |

All plain-MC slopes are consistent with −0.5 within their CIs.

At N = 65,536:

| Scenario | Method | SD across 20 reps | RMSE | VRF vs plain [95% CI] | Time vs plain | Efficiency ratio [bootstrap 95% CI] |
|---|---|---:|---:|---|---:|---|
| Asian call | plain | 0.01768 | 0.01768 | 1 | 1.00 | 1 |
| Asian call | **rqmc** | 0.00054 | 0.00055 | **1052 [417, 2659]** | 1.69 | **621 [303, 1372]** |
| Asian call | rqmc_incremental | 0.00628 | 0.00629 | 7.9 [3.1, 20.0] | 1.25 | 6.4 [3.0, 13.9] |
| Paper barrier | plain | 0.04368 | 0.04513 | 1 | 1.00 | 1 |
| Paper barrier | **rqmc** | 0.00052 | 0.00061 | **6988 [2766, 17654]** | 1.85 | **3768 [1729, 8954]** |
| Paper barrier | rqmc_incremental | 0.00876 | 0.00896 | 24.9 [9.8, 62.9] | 1.31 | 19.0 [8.5, 43.5] |
| Deep barrier | plain | 0.05278 | 0.05389 | 1 | 1.00 | 1 |
| Deep barrier | **rqmc** | 0.00782 | 0.00791 | **45.5 [18.0, 115.0]** | 1.85 | **24.7 [8.6, 62.3]** |
| Deep barrier | rqmc_incremental | 0.02467 | 0.02597 | 4.6 [1.8, 11.6] | 1.28 | 3.6 [1.3, 9.0] |

The VRF CIs use the F-distribution for a ratio of two 20-sample variances. The
efficiency CIs use Stage 2's `bootstrap_efficiency_ratio` (§2.7, criterion 3).
Timings are indicative; Chapter 7's master sweep is authoritative.

How the results develop with N:

- **RQMC's advantage grows with N,** because its error falls faster than plain
  MC's. With the bridge, VRF rises from 224 to 1052 on the Asian call, from 254
  to 6988 on the paper barrier, and from 4.8 to 45.5 on the deep barrier over
  the grid.
- **RQMC has a fixed setup cost.** Scrambling a 252-dimensional Sobol generator
  takes about 35 ms per estimate, against 3–4 ms for a whole plain-MC run at
  N=256.
  - At N=256 the bridge's efficiency ratio on the deep barrier is therefore
    only **0.4**: RQMC is *less* efficient than plain MC there.
  - The incremental construction is below 1 at N=256 on all three options.
  - The bridge pays off from N ≈ 1024 upwards (deep barrier 3.5×). The
    incremental construction pays off from N ≈ 1024 on the Asian call and
    only from N ≈ 4096 on the barriers.

**Acceptance.**

- **Within 3 SE (§2.7, criterion 1).** An RQMC estimate has no per-run CI, so each cell
  is checked with one t-interval built from its 20 scrambles. Every cell of
  every method lies within 3 SE of the reference (45/45).
- **Coverage (§2.7, criterion 2).** The 95% t-interval covers the reference in 14/15
  (plain), 14/15 (rqmc) and 15/15 (rqmc_incremental) cells.
- **Plain-MC per-replicate CIs** cover the 252-date reference in 85–100% of
  replicates per cell. With 20 replicates per cell, that range is consistent
  with 95%.

### 5.5 Why the rates differ by option

Our initial plan predicted slopes of about −0.8 to −1.0 for the Asian and only
−0.5 to −0.7 for the barrier. Our results support the *mechanism* but not the
*grouping*.

- **Deep barrier: −0.60, VRF 45×. This is as predicted.** The knock-in
  indicator depends on the path *maximum*. That is a discontinuous function of
  many bridge coordinates at once, and its discontinuity surface is not aligned
  with the coordinate axes. RQMC still helps, since it makes a good constant
  cheaper, but its rate stays close to the MC rate. The smoke test's orthant
  probability showed the same effect in 2 dimensions. The shortfall is
  expected from theory and is not a failure.
- **Paper barrier: −0.80, VRF 6988×. The largest gain in the study.** At
  B=110.68 the up-and-out leg is worth only 0.023, so the up-and-in payoff is
  almost exactly the vanilla call `(S_T − K)⁺`. Under the bridge, `S_T` is a
  function of **Sobol dimension 0 alone** (100% of `Var(log S_T)`, table
  below). The integrand is therefore essentially one-dimensional with a single
  kink, which is QMC's best case. The same near-identity with the vanilla call
  gave Stage 3's control variate its 1542× VRF.
- **Asian call: −0.72 [−0.78, −0.65]. Below the predicted −0.9.** The bridge
  puts 75% of `Var(log G)` in dimension 0 and 98.4% in the first four
  dimensions. The remaining ~1.6% is spread thinly over hundreds of
  high-index dimensions. The Sobol points balance those dimensions poorly, so
  that part of the variance behaves almost like plain MC. The fitted slope over
  N = 256–65,536 blends a fast-decaying low-dimensional part with this
  `N^-1/2` remainder. The payoff kink `max(G − K, 0)` slows the fast part as
  well. The measured CI excludes −0.9; we report the measurement and the
  reason rather than the prior expectation.

**Our plan's pre-registered sanity check** ("RQMC should beat MC more clearly on
the Asian than on the barrier"; a very different outcome was to be treated as a
possible bug):

- It holds for the genuinely path-dependent deep barrier: 1052× against 45×.
- It fails for the paper barrier, for the reason above.
- We investigated before reporting it, as that check requires, and it is not a bug:
  - the RQMC mean agrees with two independent references (§5.3);
  - the bridge passes its orthogonality tests;
  - the behaviour is explained by the variance-share calculation;
  - it matches Stage 3's 1542× control-variate result.

### 5.6 Brownian bridge vs incremental construction

**Figures:** `qmc_bridge_vs_incremental.png`, `qmc_effective_dimension.png`.

The two constructions produce the same path law and use the same Sobol points.
The only difference is which dimension drives what.

Variance ratio, incremental / bridge, at each N (256 → 65,536):

| Scenario | Ratio by N |
|---|---|
| Asian call | 50, 27, 52, 81, 133 |
| Paper barrier | 89, 169, 536, 74, 281 |
| Deep barrier | 8.6, 6.9, 5.4, 6.3, 9.9 |

The effective-dimension explanation can be computed exactly for any functional
that is linear in the Brownian path. The share of the variance carried by the
first k Sobol dimensions is:

| Functional | Construction | k=1 | k=4 | k=16 |
|---|---|---:|---:|---:|
| `log S_T` | incremental | 0.4% | 1.6% | 6.3% |
| `log S_T` | **bridge** | **100%** | 100% | 100% |
| `log G` (252-date average) | incremental | 1.2% | 4.7% | 17.8% |
| `log G` (252-date average) | **bridge** | **75.1%** | 98.4% | 99.9% |

- **Incremental construction.** The variance is spread almost evenly over all
  252 dimensions. The high-index dimensions are exactly where a Sobol sequence
  is least uniform, so incremental RQMC keeps only a constant-factor gain
  (VRF ≈ 8 on the Asian call), and its slope (−0.61) is barely steeper than
  MC's.
- **Bridge.** It puts the variance where the Sobol sequence is best. That is
  worth a further 27–536× in variance on the Asian and the paper barrier.
- **Deep barrier, bridge vs incremental.** The bridge still wins, by a factor
  of 5–10 in variance, even though the path maximum depends on the fine
  structure of the path. The bridge places the path's coarse shape, which
  decides most barrier crossings, in the leading dimensions.

### 5.7 SDE discretization order

**Figures:** `scheme_strong_order.png`, `scheme_weak_order.png`.

**Design.**

- **Common random numbers.** Each sample path is driven by *one* 512-step
  Brownian path. Blocks of it are summed to get the normals for every
  `m ∈ {4, 8, …, 512}` (Δt = 1/m). Every step size and every scheme therefore
  sees the same Brownian motion.
- **Reference.** The error is taken relative to the **exact log-GBM scheme on
  the same grid**. That scheme has no discretization error at the grid points,
  so the difference is pure scheme error. For the barrier, it also holds the
  monitoring dates fixed, which separates the scheme effect from the
  monitoring bias of §5.3.
- **Sample size.** 20 replicates × 65,536 paths per step size.
- **Analytic checks.** Each step factor of both schemes has mean `1 + rΔt`,
  because Milstein's correction `½σ²Δt(Z²−1)` has mean zero. Hence
  `E[S_T^Δ] − E[S_T] = S₀(1 + rΔt)^m − S₀e^{rT}` exactly, for both schemes. The
  second moment is also available analytically.

Fitted orders (log-log slope against Δt). Weak slopes use only points resolved
from zero at more than 3 SE.

| Error | Euler–Maruyama | Milstein | Theory |
|---|---:|---:|---|
| Strong, `E\|S_T^Δ − S_T\|` | **0.49** | **0.97** | 0.5 / 1.0 |
| Weak, `E[S_T]` | 1.17 (3 resolved points) | 1.00 | 1 / 1 |
| Weak, European call K=105 | **1.01** | **0.99** | 1 / 1 |
| Weak, up-and-in barrier (paper's option) | **1.03** | **0.99** | 1 / 1 |

- **The MC agrees with the exact formula.** The simulated `E[S_T]` weak error
  matches `S₀(1 + rΔt)^m − S₀e^{rT}` within 3 SE for both schemes at all eight
  step sizes.
- **Why Euler–Maruyama's `E[S_T]` points are hollow.** Its weak error is
  resolved only at coarse Δt. The analytic line covers the rest. The reason is
  that Euler–Maruyama's pathwise error is large, so even the paired difference
  is noisy.

At Δt = 1/256, the grid point closest to the paper's daily grid:

| Quantity | Euler–Maruyama | Milstein |
|---|---:|---:|
| Strong error | 0.1450 | 0.00219 (**66× smaller**) |
| Weak error, call | −0.00071 ± 0.00016 | −0.00102 ± 0.000002 |
| Weak error, barrier | −0.00078 ± 0.00021 | −0.00104 ± 0.00001 |

These agree with Stage 2's common-random-number measurements at Δt = 1/252:
−0.0006 (Euler–Maruyama) and −0.0011 (Milstein).

### 5.8 What the base paper's scheme comparison should have measured

The base paper concludes that the choice between "Euler", "Euler–Maruyama" and
Milstein "has a negligible impact on simulation accuracy". Our measurements
support the conclusion but not its status as a *finding*:

1. **Euler and Euler–Maruyama.** The paper never writes its schemes down. Under
   the standard definitions, "Euler" for an SDE *is* Euler–Maruyama, so two of
   its three schemes are presumably one scheme under two names; alternatively
   "Euler" means Euler on log S, which is the exact scheme for GBM. Either way
   this is an inference (§3.4).
2. **Milstein.** Milstein's advantage is **strong** (pathwise) order: 0.97
   against 0.49 here, with a 66× smaller pathwise error at daily steps. A price
   is an expectation, so it depends on **weak** order, which is 1 for both
   schemes (0.99–1.03 measured). The strong-order gain does not carry over to
   the price.
   - At daily steps, Milstein's barrier price bias (−0.00104) is in fact
     slightly *larger* than Euler–Maruyama's (−0.00078).
   - Both are O(Δt) and about 0.01% of the price.
   - The per-path SD of the barrier payoff is 12.55. So these biases are
     **47×** smaller than the plain-MC SE at N=65,536 and **171×** smaller at
     the paper's N=5,000.

   Stage 2 also found that the paper's Figs. 2–4 show identical sample points
   (shared random numbers or a duplicated figure). Its scheme plots could not
   have come out differently, and "negligible" was the result theory predicts.
   It is not a discovery. (Our initial plan's wording, that Milstein "cannot
   change" the expectation, is too strong: it does change it, at O(Δt), as
   measured.)
3. **What should have been measured.** Strong and weak error against Δt, with
   common random numbers and fitted orders, as above. That is the only design
   that separates the schemes, and it shows what each is good for: Milstein
   when pathwise accuracy matters, neither when only a price is needed.
4. **The exact scheme makes both unnecessary.** Under GBM the exact log-GBM
   scheme costs about the same (0.64 s against 0.54 s for Euler–Maruyama and
   0.77 s for Milstein per 65,536 paths at m=512) and has **zero**
   discretization error. The only discretization error left in this project is
   the barrier's *monitoring* bias (§5.3, §4.4). That is a property of the
   contract's observation dates, not of the SDE scheme.

### 5.9 Summary

| Requirement (Stage 4 plan) | Evidence |
|---|---|
| Smoke test: known integral, slope steeper than `N^-1/2` | Smooth exponential −0.83; Gaussian orthant −0.77; plain MC −0.43 / −0.50 |
| Scrambled Sobol, `random_base2` only, `ndtri`, R independent scrambles | `sobol_uniforms`; non-power-of-2 N raises; one scramble per replicate; within-run SE logged as NaN |
| Brownian bridge | Orthogonal bridge matrix; tests for dimension 0 → `W_T` and dimensions 0–1 → midpoint |
| Headline RQMC vs MC with fitted slopes | Asian −0.72, paper barrier −0.80, deep barrier −0.60 (plain ≈ −0.5); VRF at 65,536 = 1052 / 6988 / 45 |
| Explain the barrier rate | Deep barrier matches the prediction; paper barrier ≈ a 1-D vanilla call under the bridge; variance-share table |
| Bridge vs incremental | Bridge 27–536× lower variance (Asian, paper barrier), 5–10× (deep barrier) |
| Strong / weak order | Strong 0.49 / 0.97; weak 0.99–1.03 for both schemes; `E[S_T]` MC matches the analytic formula |
| Critique of the paper's scheme comparison | §5.8 |
| Validation (§2.7) | 45/45 cells within 3 SE; t-interval coverage 43/45 |

**Recommendation carried into Chapters 6–7.**

- RQMC with a Brownian bridge is the most efficient technique measured so far
  on both the geometric Asian (efficiency 621×) and the near-the-money barrier
  (3768×).
- On a genuinely path-dependent barrier, its gain drops to about 25× in
  efficiency, and at N=256 it is not worth its setup cost. That is the regime
  where importance sampling should be compared against it.

**References**

- Caflisch, R. E., Morokoff, W., & Owen, A. B. (1997). Valuation of
  mortgage-backed securities using Brownian bridges to reduce effective
  dimension. *Journal of Computational Finance*, 1(1).
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*,
  ch. 3 (Brownian bridge), ch. 5 (quasi-Monte Carlo), ch. 6 (discretization).
  Springer.
- Owen, A. B. (1997). Scrambled net variance for integrals of smooth functions.
  *Annals of Statistics*, 25(4).

---

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
- how the drift is chosen, including a units correction to our plan's
  heuristic (§6.2);
- the results on three barriers of increasing rarity (§6.3–6.5);
- the weight diagnostics that decide whether an IS result can be trusted (§6.6).

Files:

- Estimator: `src/vr_importance.py`, registered as `"importance"`.
- Experiment: `experiments/exp_importance.py`.
- Tests: `tests/test_importance.py` (10 tests, including a smoke test on a known tail probability).
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

**The likelihood ratio telescopes.** The per-step density ratios
multiply to a function of the *sum* of the increments, which is `W~_T`. So the
barrier monitoring, the path maximum and the 252-step structure never enter the
weight. The code computes `W~_T = √Δt · Σ_j Z_j` from the **same** normal array
it passes to the engine, and never draws fresh normals for the weight. A test
checks this path by path (`test_likelihood_ratio_uses_the_same_normals`).

This identity is exact for the `exact` scheme, which every experiment uses,
and for Euler–Maruyama. It does not hold under Milstein: there the drift shift
also changes the scheme's correction term, so the weight no longer undoes the
shift exactly. The estimate is then biased at O(Δt): about 0.02 on `E[S_T]` at
daily steps with the deep barrier's θ, twenty times Milstein's own weak error
(§8.4).

**Smoke test.** Before any option was involved, the same
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

Our initial plan wrote this without the final `/σ`. That expression is a
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
machinery, as the project plan specified.

| Scenario | B | θ₀ | θ chosen by the pilot | Optimum of the full scan (§6.5) |
|---|---:|---:|---|---|
| `paper_barrier` | 110.68 | 0.457 | 0.686 = 1.5·θ₀ in **every** replicate (the grid edge) | ≈2.9·θ₀ |
| `deep_barrier` | 140 | 1.632 | 1.632 = θ₀ in every replicate | θ₀ |
| `rare_barrier` | 160 | 2.300 | 2.300 = θ₀ in every replicate | θ₀ |

The pilot choice is stable: across all 100 estimates per scenario it always
picked the same grid point.

### 6.3 Headline: the deep barrier (B = 140)

Our plan called `deep_barrier` the rare-event case. Measuring it (diagnostic
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
| Plain MC | 3.4107 | 0.0498 | 1 | 1 | 1.00 [0.32, 3.88] |
| Antithetic | 3.3713 | 0.0476 | 1.11 | 1.00 [0.26, 3.00] | 1 |
| **Importance** | **3.3881** | **0.0132** | **9.88** | **8.8 [2.3, 22.8]** | **8.8 [3.5, 18.7]** |

The 252-date reference is 3.385730 ± 0.001014 (Stage 4). IS lands 2.4 × 10⁻³
from it, well inside its own SE. The within-run VRF of 9.9 is stable across N
(9.6–10.6), and it matches the θ scan's minimum variance of 0.101 × plain.

The across-replicate VRF (11.1, with 95% F-interval [4.4, 27.9]) agrees with it,
but its interval is much wider. That is the R = 20 limitation that Chapter 3
already flagged. For IS the within-run SE is valid, because the weighted samples
are i.i.d., so the within-run VRF is the precise number.

Antithetic variates, the base paper's only technique, do almost **nothing**
here: VRF 1.11. The payoff is still monotone in every driving normal (raising
any `Z_j` can only raise the path), so antithetic cannot hurt. But it is
strongly non-linear: it is zero on about 91% of paths and pays only through an
indicator times a kink. `Z` and `−Z` therefore give almost uncorrelated payoffs,
and antithetic only cancels the linear part of a payoff.

### 6.4 The pilot cost, and when IS pays

The pilot is a fixed 10,000 paths regardless of N. At small N it dominates:

| N | IS time ÷ plain time (deep) | IS efficiency vs plain (deep) | (paper) | (rare) |
|---:|---:|---|---|---|
| 256 | 50.3 | 0.28 [0.09, 0.70] | 0.07 [0.02, 0.20] | 0.77 [0.33, 2.31] |
| 1,024 | 9.5 | 0.76 [0.27, 2.37] | 0.50 [0.18, 1.45] | 0.95 [0.35, 2.24] |
| 4,096 | 2.9 | 4.2 [1.6, 13.7] | 1.4 [0.7, 2.7] | 10.1 [4.4, 30.2] |
| 16,384 | 1.7 | 7.9 [3.5, 17.2] | 8.3 [4.0, 17.8] | 14.4 [7.1, 29.8] |
| 65,536 | 1.26 | 8.8 [2.3, 22.8] | 6.3 [3.3, 12.0] | 48.6 [20.9, 112.1] |

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

We report both, as our plan required for any IS result. An IS result that quoted only its VRF, without
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
- efficiency vs plain **48.6× [20.9, 112.1]**;
- efficiency vs antithetic **21.0× [9.5, 48.3]**.

This is the regime the plan had in mind: plain MC wastes 98% of its paths,
and IS is the right tool.

### 6.8 Acceptance and summary

- **Correctness (§2.7, criteria 1–2).** Every IS cell on the two referenced barriers is
  within 3 SE of the 252-date reference (10/10), and every t-interval covers it
  (10/10). Per-replicate 95% CI coverage is 0.97 (deep) and 0.96 (paper).
- **Pre-registered sanity check, "IS wins big on `deep_barrier`, marginally on
  `paper_barrier`".** It holds in direction:
  - deep ≈ 9× (within-run VRF 9.9);
  - paper ≈ 6× (within-run VRF 4.6).

  "Big" is only true where the event is actually rare: ≈49× at B = 160. On the
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

---

## 7. Comparative Results and Efficiency Analysis

*(Stage 5 — Tamzeed Mahfuz, 2105012)*

Chapters 3–6 each tested one technique, on its own machine and its own
scenarios. This chapter puts all five on one footing:

- one run, one machine, one thread;
- one efficiency metric, `1/(variance × time)`;
- one set of 252-date reference prices.

It is the only chapter whose timings are authoritative (by the group protocol,
§2.5); every other chapter's timings are indicative.

Files:

- Experiment: `experiments/exp_master.py`, with helpers in
  `experiments/_stage5_common.py`.
- Raw outputs:
  - `results/raw/tamzeed_master.csv` (1,700 rows);
  - `results/raw/tamzeed_master_reference.csv` (64 rows).
- Derived tables: `results/tables/master_summary.csv`,
  `master_efficiency.csv` and `master_findings.md`.
- Figures: `master_efficiency_table.png`, `master_efficiency_bars.png` and
  `master_rmse_vs_n.png`.

### 7.1 Protocol

**Grid.** The sweep covers 17 (scenario, method) cells × `N_GRID` (256 …
65,536) × R = 20 replicates, at 252 steps with the exact scheme:

| Family | Scenario | Methods |
|---|---|---|
| Barrier | `paper_barrier` (B = 110.68), `deep_barrier` (B = 140) | plain, antithetic, control variate, RQMC, importance sampling |
| Asian | `asian_arith` (no closed form) | plain, antithetic, control variate, RQMC |
| Asian | `paper_asian_geo` call (the base paper's option) | plain, antithetic, RQMC |

Two methods are not applied to the Asian options, and the gaps are by design:

- **Control variate on the geometric Asian.** That option has a closed form, so
  its natural control would be itself, and `vr_control` refuses it.
- **Importance sampling on either Asian.** Our IS is a barrier-crossing drift
  shift, so it has no rare event to target there.

**Machine.** `LAPTOP-G689CP2J_AMD64`, `OMP_NUM_THREADS=1`, `time.perf_counter`.
The sweep ran in 358 s. Every runtime includes all of a method's work: the
pilot runs for the control variate (10% of N, at least 256 paths) and for IS
(10,000 paths), and the 252-dimensional Sobol scramble for RQMC.

**Interleaving.** The first attempt used `benchmark.run_sweep`, which runs each
method as one contiguous block. Its first two minutes were measurably slower,
probably from CPU frequency scaling on the laptop. Every `paper_barrier` cell
ran in that window, so its timings were inflated relative to `deep_barrier`
cells doing identical work:

- antithetic 0.96 s vs 0.48 s;
- RQMC 1.49 s vs 1.06 s.

In a blocked design any drift lands entirely on whichever method happens to be
running, and silently becomes part of that method's "efficiency".

The authoritative sweep therefore **interleaves**:

1. An untimed warm-up call per cell. This also builds RQMC's cached
   Brownian-bridge matrix, which would otherwise be charged to one replicate.
2. For each (replicate, N) block, all 17 cells run once, in a freshly shuffled
   order (`default_rng(402)`).

The seed keys are exactly `run_sweep`'s. We verified that all 1,700 prices are
**bit-identical** to the blocked run, so only the timings changed. After
interleaving, the paper and deep barrier timings agree (antithetic 0.48 s and
0.45 s).

**References and error bars.**

- **Reference prices.** Bias and RMSE use the discretely monitored (252-date)
  price, never the continuous formula (§5.3):
  - 7.094133 ± 0.000096 (paper barrier) and 3.385730 ± 0.001014 (deep barrier),
    from Stage 4's 64-scramble RQMC reference rows;
  - 3.000200 for the geometric Asian (discrete closed form);
  - **3.162981 ± 0.000064** for the arithmetic Asian. This is a new 64-scramble
    × 65,536-path RQMC reference, simulated *after* the sweep so it cannot
    disturb the timings.
- **Error bars.**
  - Efficiency ratios are against plain MC in the same (scenario, N) cell,
    with 2,000-resample bootstrap 95% CIs over replicates.
  - VRF carries an F-distribution 95% CI.
  - Where the within-run SE is valid (every method except RQMC), we also report
    the within-run VRF, the ratio of mean SE². It is far more precise at
    R = 20.

### 7.2 Headline table (N = 65,536)

Efficiency relative to plain MC, [bootstrap 95% CI], with RMSE against the
252-date reference (`master_efficiency_table.png`):

| Method | Paper barrier | Deep barrier | Arithmetic Asian | Geometric Asian |
|---|---|---|---|---|
| Plain MC | 1 · RMSE 4.7e-2 | 1 · RMSE 4.6e-2 | 1 · RMSE 2.2e-2 | 1 · RMSE 2.4e-2 |
| Antithetic | 1.2 [0.5, 2.8] | 1.1 [0.5, 2.9] | 0.97 [0.35, 2.6] | 1.1 [0.5, 2.9] |
| Control variate | 1,602 [656, 4,276] · 1.1e-3 | 4.2 [1.6, 10.0] · 2.2e-2 | 770 [266, 1,970] · 6.9e-4 | n/a |
| **RQMC + bridge** | **2,260 [1,065, 5,050]** · 6.2e-4 | **21.3 [7.9, 51.4]** · 7.3e-3 | **794 [269, 2,120]** · 5.2e-4 | **791 [358, 1,638]** · 5.7e-4 |
| Importance sampling | 2.8 [1.3, 5.7] · 2.4e-2 | 8.6 [3.0, 37.8] · 1.5e-2 | n/a | n/a |

The underlying VRFs and per-estimate costs:

| Method | Within-run VRF (paper / deep / arith.) | Time ratio vs plain (paper / deep / arith. / geo.) |
|---|---|---|
| Antithetic | 1.5 / 1.1 / 1.4 | 0.99 / 0.73 / 0.94 / 0.95 |
| Control variate | 1,535 / 3.2 / 976 | 1.23 / 0.94 / 1.33 / — |
| RQMC (across-replicate VRF) | 5,723 / 40 / 1,790 | 2.53 / 1.89 / 2.25 / 2.24 |
| Importance sampling | 4.6 / 9.8 / — | 1.38 / 1.05 / — / — |

Four things stand out.

1. **Two orders of magnitude separate the techniques.** Control variates and
   RQMC gain 10²–10³ in efficiency on three of the four options. Antithetic
   variates, the base paper's only technique, gain nothing measurable at this
   R: every antithetic CI contains 1. That does not contradict the paper's
   1.3–1.5×. The within-run VRFs here (1.4–1.5; 1.1 on the deep barrier) agree with it and
   with Chapter 3.
   An efficiency gain of 1.2–1.5× simply cannot be resolved by 20 replicates
   whose variance estimates carry about ±32% relative error.
2. **RQMC has the best point estimate everywhere,** but on the paper barrier and
   the arithmetic Asian its CI overlaps the control variate's. Those two are
   statistically tied at N = 65,536.
3. **The deep barrier is hard for everyone.** The best method gains only ~21×.
4. **Importance sampling is never first.** It is a clear second on the deep
   barrier (8.6×, CI overlapping RQMC's) and a distant last among the new
   methods on the paper barrier.

### 7.3 Efficiency depends on N: who wins where

Efficiency at a single N hides the most useful structure. The best method at
each N (point estimate, from `master_findings.md`):

| Scenario | N = 256 | 1,024 | 4,096 | 16,384 | 65,536 |
|---|---|---|---|---|---|
| Paper barrier | CV (1,151) | CV (1,715) | CV (772) ≈ RQMC | CV (2,477) ≈ RQMC | RQMC (2,260) ≈ CV |
| Deep barrier | plain ≈ CV | RQMC (10.6) ≈ antithetic | RQMC (2.4) ≈ plain | RQMC (24) ≈ CV | RQMC (21) ≈ IS |
| Arithmetic Asian | CV (567) | CV (803) ≈ RQMC | CV (628) ≈ RQMC | CV (654) ≈ RQMC | RQMC (794) ≈ CV |
| Geometric Asian | RQMC (32) | RQMC (81) | RQMC (214) | RQMC (741) | RQMC (791) |

"≈ X" means the bootstrap CIs of the top two overlap.

Two mechanisms drive the crossovers.

- **Rate.** The control variate reduces the *constant* but keeps the `N^-1/2`
  rate. RQMC improves the *rate*. The RMSE-vs-N slopes, with bootstrap 95% CIs
  (`master_rmse_vs_n.png`):

  | Scenario | Plain | Antithetic | Control variate | RQMC | Importance |
  |---|---|---|---|---|---|
  | Paper barrier | −0.50 | −0.49 | −0.50 | **−0.79 [−0.85, −0.73]** | −0.47 |
  | Deep barrier | −0.42 | −0.46 | −0.51 | **−0.60 [−0.66, −0.53]** | −0.50 |
  | Arithmetic Asian | −0.56 | −0.51 | −0.49 | **−0.74 [−0.81, −0.68]** | — |
  | Geometric Asian | −0.51 | −0.48 | — | **−0.74 [−0.79, −0.68]** | — |

  So RQMC's efficiency ratio grows with N (paper barrier: 33 → 2,260), while the
  control variate's stays flat (772–2,477 with no trend). At small N the
  control variate wins, and at large N RQMC catches up and eventually passes it.
  These slopes reproduce Stage 4's on a different run: paper barrier −0.80,
  deep barrier −0.60, geometric Asian −0.72.
- **Fixed costs.** Every new method carries overhead that does not shrink
  with N:
  - RQMC's scramble;
  - the control variate's 256-path minimum pilot;
  - IS's 10,000-path pilot.

  On the deep barrier at N = 256, where the variance gains are small, these
  costs swamp them. IS (0.07×) and RQMC (0.45×) are significantly *worse* than
  plain MC there.

### 7.4 Time to reach a target accuracy

Efficiency ratios compare methods at equal N. A practitioner asks something
different: *how long until my price is accurate to ε?*

For each (scenario, method), `exp_master.py` fits `RMSE(N) = a·N^s` and
`runtime(N) = c + d·N` over the grid, then solves for the N and wall-clock
time that give RMSE = ε. In the table below, `*` marks an extrapolation beyond
N = 65,536, and "≈" in the last column means the runner-up is within about 1.2×.

| Scenario | ε | Plain | Antithetic | Control variate | RQMC | IS | Fastest |
|---|---:|---:|---:|---:|---:|---:|---|
| Paper barrier | 0.01 | 10.5 s* | 8.5 s* | **0.011 s** | 0.048 s | 3.8 s* | CV |
| Paper barrier | 0.001 | 1,020 s* | 920 s* | 0.65 s* | **0.59 s** | 471 s* | RQMC ≈ CV |
| Deep barrier | 0.01 | 27.9 s* | 14.7 s* | 2.6 s* | **0.66 s** | 1.5 s* | RQMC |
| Deep barrier | 0.001 | 6,460 s* | 2,170 s* | 227 s* | **31 s*** | 148 s* | RQMC |
| Arithmetic Asian | 0.01 | 2.3 s* | 1.7 s* | **0.005 s** | 0.031 s | — | CV |
| Arithmetic Asian | 0.001 | 147 s* | 152 s* | **0.44 s** | 0.51 s | — | CV ≈ RQMC |
| Geometric Asian | 0.01 | 3.1 s* | 2.9 s* | — | **0.029 s** | — | RQMC |
| Geometric Asian | 0.001 | 293 s* | 343 s* | — | **0.57 s** | — | RQMC |

The ratios are the point. For one-cent accuracy on the arithmetic Asian, plain
MC needs about 270,000 paths and 2.3 s. The control variate needs 340 paths and
5 ms, about **500× faster**.

At ε = 0.001 on the deep barrier, plain MC would need an extrapolated 6.9 × 10⁸
paths (almost two hours). RQMC needs about 30 s.

The RQMC extrapolations assume its fitted slope continues past N = 65,536, so
they are indicative only.

### 7.5 Why each technique wins where it wins

The rankings are not arbitrary. Each follows from how the payoff is related to
what the technique exploits.

| Scenario | Payoff structure | Consequence |
|---|---|---|
| Paper barrier (B = 110.68) | Almost a vanilla call: C_uo = 0.023 is 0.3% of the price | **CV:** the vanilla call is a nearly perfect control (ρ = 0.9997). **RQMC:** under the bridge, `S_T` depends on Sobol dimension 0 alone (§5.5), so the payoff is effectively one-dimensional. **IS:** the knock-in is not rare (60%), so there is little to gain. |
| Deep barrier (B = 140) | Knock-in depends on the path *maximum*: a discontinuity across many dimensions; 33% of the value is exposed to missed crossings (§5.3) | **CV** collapses (ρ = 0.83, VRF 3): many vanilla-in-the-money paths never knock in. **RQMC** slows to `N^-0.60` (§5.5). **IS** attacks the moderate rarity (9% knock-in) directly and lands between them. |
| Arithmetic Asian | Smooth average, extremely close to the geometric average | **CV:** the geometric Asian is an almost exact control (ρ = 0.9995, VRF 976). **RQMC:** 75% of `Var(log G)` sits in Sobol dimension 0 (§5.6), a low effective dimension. The two tie. |
| Geometric Asian | As above, but it *is* the closed form | Only RQMC applies. It gains ~800× at N = 65,536. |

A useful way to see this: **the control variate and RQMC both exploit the same
fact, that the payoff is nearly a function of one Gaussian.** They win together
on the paper barrier and the Asians, and they both weaken on the deep barrier,
where that fact fails. IS exploits a *different* fact, rarity. So it is the
only technique whose advantage *grows* as the barrier moves away: 2.8× → 8.6×
here, and ≈49× at B = 160 in §6.7 (an indicative Chapter-6 timing).

### 7.6 Agreement with earlier stages

The master run is an independent re-simulation with different seeds and
timings. It reproduces every earlier variance result:

| Quantity | Master sweep | Earlier stage |
|---|---:|---:|
| CV on arithmetic Asian: mean ρ | 0.999486 | 0.999488 (Stage 3) |
| CV on arithmetic Asian: within-run VRF | 976 | 975.7 (Stage 3) |
| RQMC VRF, deep barrier | 40 | 45 (Stage 4) |
| RQMC VRF, paper barrier | 5,723 | 6,988 (Stage 4) |
| IS within-run VRF, deep / paper | 9.8 / 4.6 | 9.9 / 4.6 (Chapter 6) |
| Antithetic within-run VRF, paper barrier | 1.5 | 1.47 (Stage 2) |

The RQMC VRFs differ by 11–18%. That is well inside their F-intervals, which
span a factor of about 6.

IS on the paper barrier gives 2.8× here and 6.3× in Chapter 6. The difference
is not in the within-run VRF, which is 4.6 in both. It comes from the noisier
across-replicate variance (VRF 3.9 vs 7.3, both inside each other's CI) and a
higher time ratio (1.38 vs 1.14). **This is exactly why a single
authoritative run is needed**, and why efficiency ratios at R = 20 should be
read with their intervals.

### 7.7 Acceptance (§2.7)

- **Within 3 SE (criterion 1).** 84 of 85 method cells are within 3 SE of the
  252-date reference. The exception is plain MC on the paper barrier at
  N = 4,096 (z = 3.35). With 85 cells, about 0.25 such events are expected, so
  this one is unremarkable. No variance-reduced cell fails.
- **CI coverage (criterion 2).** Per-replicate 95% CI coverage is between 0.93 and
  0.98 for every method with a valid SE. RQMC has no within-run CI by design;
  its error bars come from independent scrambles.
- **t-intervals.** 77 of 85 cells' t-intervals (from their 20 replicates) cover
  the reference, against 80.75 expected. P(≥ 8 misses | 95%) = 0.062. The misses
  are spread across methods and N (plain 4, antithetic 3, RQMC 1; none for the
  control variate or IS, and none on the deep barrier), with no pattern that
  would indicate a bias.
- **RMSE, not variance (criterion 4).** Every method is unbiased at 252 dates.
  - At N = 65,536 the largest |bias|/SD of any cell is 0.51, and the largest
    bias² share of RMSE² is 21%.
  - With R = 20 the replicated mean itself fluctuates by SD/√20 ≈ 0.22·SD, so a
    0.51 ratio is a ≤ 2.3-SE fluctuation, not a detectable bias.
  - RMSE and standard deviation therefore agree to within that noise, and the
    RMSE ranking equals the variance ranking.

  This held **only because we measured against the 252-date reference**.
  Against the continuous formula, the −0.011 (paper) and −0.160 (deep)
  monitoring biases would floor every RQMC and CV RMSE. On the paper barrier,
  for example, RQMC's RMSE slope would read −0.26 instead of −0.80 (§5.3), and
  the deep barrier's 0.160 bias would exceed every variance-reduced method's
  standard deviation, hiding the differences between them.

### 7.8 Caveats

- **R = 20 makes efficiency ratios wide.** A typical 95% CI spans a factor of
  3–5. We can rank techniques that differ by 10× or more. We cannot rank those
  within about 2× (RQMC vs CV on the paper barrier and the arithmetic Asian;
  RQMC vs IS on the deep barrier). The within-run VRFs are much tighter, but
  they are not available for RQMC.
- **One machine.** The timing ratios are specific to this laptop, NumPy 2.1.3
  and single-threaded code. RQMC's cost ratio (1.9–2.5×) in particular depends
  on SciPy's Sobol implementation.
- **Fixed-size pilots.** The IS pilot (10,000 paths) is a design choice that
  penalizes IS at small N (§6.4). Tuning θ once offline would remove it.
- **Techniques were not combined.** RQMC + control variate, or RQMC + IS, are
  standard and would likely dominate. They are future work (Chapter 8).

---

## 8. Discussion, Limitations and Conclusion

*(Stage 5 — Tamzeed Mahfuz, 2105012)*

### 8.1 What the base paper asked, and what we can now answer

The base paper priced an up-and-in barrier call and a geometric Asian option by
Monte Carlo. It tried one variance-reduction technique, antithetic variates,
and pointed to "advanced variance reduction techniques" as the direction for
future work (its conclusion: prioritize variance reduction over new random-walk
models). Its implicit question was *how much further can variance be
reduced, and at what cost?*

At N = 65,536, on one machine and at equal wall-clock time (Chapter 7), the
answer comes in three tiers:

| Tier | Techniques and gain |
|---|---|
| **Structural techniques** | **Control variates** and **RQMC with a Brownian bridge**: 10²–10³× the efficiency of plain Monte Carlo on the paper barrier (1,602× and 2,260×), the arithmetic Asian (770× and 794×) and the geometric Asian (RQMC 791×) |
| **Rare-event technique** | **Importance sampling**: 3–9× on the frozen barriers, rising to ≈49× (Chapter 6, indicative timing) where knock-in is genuinely rare (B = 160) |
| **The paper's technique** | **Antithetic variates**: a real variance ratio of 1.1–1.5×, but an efficiency gain that cannot be told apart from 1 at R = 20 |

The spread between the best and the worst is three orders of magnitude. The
base paper's 1.3–1.5× was not wrong. It was the smallest effect in the whole
comparison.

### 8.2 Recommendation

> **To reach accuracy ε on the barrier option, use randomized quasi-Monte Carlo
> with Brownian-bridge path construction; on the arithmetic Asian, use the
> geometric-Asian control variate.**

The evidence is in §7.3–7.4:

- **Barrier.** RQMC is the fastest method to ε = 0.001 on both barriers. On the
  deep barrier it wins at every accuracy we tested: 0.66 s to one cent, against
  1.5 s for IS, 2.6 s for the control variate and 28 s for plain MC. It is the
  only technique whose error falls faster than `N^-1/2` (slope −0.79 on the
  paper barrier, −0.60 on the deep barrier). Its lead therefore *grows* as ε
  shrinks.
- **Arithmetic Asian.** The control variate is the fastest to both ε = 0.01
  (5 ms, about 500× faster than plain MC) and ε = 0.001 (0.44 s). It needs no
  power-of-two sample sizes and no scrambling. Unlike RQMC, it provides a
  valid standard error from a single run.

Two qualifications make the recommendation precise. Both follow from
*mechanism*, not from the particular numbers.

1. **Barrier close to the strike, loose ε.** When the barrier is near the strike
   (as in the paper's B = 110.68, where the up-and-out leg is 0.3% of the
   price), the vanilla-call control variate is as good as RQMC, and cheaper for
   loose ε. It reaches one cent in 11 ms, against RQMC's 48 ms.
2. **Genuinely rare knock-in.** When the knock-in probability is a few percent
   or lower, importance sampling is the specialist tool. Its gain grows as the
   event gets rarer (2.8× → 8.6× → ≈49× across B = 110.68 → 140 → 160), while the
   control variate collapses (VRF 3 at B = 140).

   IS and RQMC exploit different structure, so they combine naturally. That
   combination is the obvious next step (§8.5).

### 8.3 Discussion

**Variance reduction is what makes the discretization bias visible.**
A daily-monitored barrier is worth less than its continuous-formula value,
because crossings between monitoring dates are missed. The monitoring biases
are:

- −0.011 on the paper barrier (7.0941 vs 7.1055);
- −0.160 on the deep barrier (3.3857 vs 3.5453).

At the paper's sample sizes, plain MC cannot see the paper barrier's bias at
all: its SE at N = 1,024 is about 0.39, 34 times the bias. Even the deep
barrier's much larger bias is only about 0.4 of a plain SE at that N (SE 0.36).
With the control variate or RQMC, the RMSE of a single estimate at N = 65,536 on
the paper barrier is 1 × 10⁻³ (CV) and 6 × 10⁻⁴ (RQMC). The replicated mean then
resolves the bias at more than 50 standard errors (§4.4, §5.3).

The same fact nearly corrupted our own comparison. Measured against the
continuous formula, every good method's RMSE hits a floor at the bias, and its
fitted slope flattens (RQMC on the paper barrier: −0.26 instead of −0.80). **An
efficiency comparison must use a reference at the same monitoring frequency as
the simulation.** The base paper benchmarked a 252-step simulation against a
continuous formula. At its precision that was harmless. At ours it would have
been wrong.

**The techniques that win share a reason.** The control variate and RQMC both
succeed because these payoffs are nearly functions of *one* Gaussian:

- the terminal value `W_T`, for the paper barrier (which is almost a vanilla
  call);
- the average log-price, for the Asians.

A control variate captures that one direction by regression. The Brownian
bridge puts it into the best-distributed Sobol coordinate. When the payoff
depends on the path maximum, as the deep barrier's does, both weaken together
(CV VRF 3, RQMC slope −0.60). Importance sampling depends on a different
property, rarity, and is the only method that improves as the barrier moves
away. This is why "which technique is best?" has no answer independent of the
option. The right question is *what structure does the payoff have?*

**Efficiency, not variance, is the right metric, and it depends on N.** Every
improved method carries a fixed overhead: a pilot run, or a Sobol scramble. At
N = 256 on the deep barrier, IS (0.07×) and RQMC (0.45×) are significantly
*worse* than plain MC. A variance reduction factor alone would have hidden that
completely. The time-to-accuracy view (§7.4) is the most useful single summary
for a practitioner. It folds rate, constant and overhead into one number.

**Checking the base paper's claims.** Across Chapters 3–7:

| Paper's claim | Our verdict |
|---|---|
| Antithetic gives 1.5× (barrier) and 1.3× (Asian); the factor is never defined | Confirmed as variance ratios (1.47 / 1.35, Chapter 3). As *efficiency* it is not resolvable at R = 20 (at N = 65,536 every master-sweep CI contains 1). The Asian put's VRF is 3.9×, and the paper's put figure appears to duplicate its call figure (§3.4). |
| Discretization scheme choice is negligible | True (≈0.001 at daily steps), but expected from theory: both schemes have weak order 1, and the paper's three scheme figures show identical points (§3.4, §5.8). |
| Convergence "stabilizes" at about 5,000 / 750–1,000 paths | Not supported as a convergence property: error keeps falling as `N^-1/2` with no plateau, and the CI is still ±5% / ±12% wide at those N (§3.4). |
| The 30-day-averaging closed form "remains identical" | False as a value: 6.708 against 3.000 for full averaging (§3.4). |

**Corrections we made to our own plan.** A relay project inherits its plan's
errors. We report ours because each one would have changed a number in this
report.

1. **The 252-step barrier reference.** It is 7.0941, not our plan's ≈7.076.
   The monitoring bias is therefore −0.011, not −0.03 (Stage 2, §3.5).
2. **The heuristic IS drift.** It needs a division by σ, since the plan's
   formula is a log-drift (§6.2).
3. **`deep_barrier` is not a rare event.** It has a 9.3% knock-in rate, so the
   rare-event demonstration needed B = 160 (§6.7).
4. **A contiguous-block benchmark design lets machine drift masquerade as a
   method effect.** The master sweep had to be interleaved (§7.1).

### 8.4 Limitations

- **Statistical resolution.** With R = 20 replicates per cell (frozen protocol),
  across-replicate variances carry about ±32% relative error. Efficiency CIs
  span a factor of 3–5. Techniques that differ by less than about 2× cannot be
  ranked:
  - CV vs RQMC on the paper barrier and the arithmetic Asian;
  - RQMC vs IS on the deep barrier.

  The within-run VRF is precise but unavailable for RQMC. A master run with
  R ≥ 100 would settle these ties.
- **One machine, one implementation.** The authoritative timings come from one
  laptop, running single-threaded NumPy 2.1.3 / SciPy 1.14.1. Cost *ratios*
  should transfer, but RQMC's (≈2.2×) depends on SciPy's Sobol generator, and
  the fixed overheads depend on Python call costs.
- **One IS family.** We used a constant drift shift tuned on a 7-point grid tied
  to the heuristic θ₀. On the paper barrier that grid could not reach the
  optimum (≈2.9·θ₀), leaving a factor of about 2.3 unused (§6.5).
  Cross-entropy or large-deviations tuning, time-varying drifts and
  barrier-conditional sampling were outside scope.
- **Fixed-size pilots.** The 10,000-path IS pilot and the control variate's
  256-path minimum penalize both methods at small N. Tuning once offline would
  remove the cost, since the IS pilot chose the same θ in every replicate.
- **Model scope.** Everything is under constant-volatility GBM, with one
  maturity, one strike and daily monitoring. Under stochastic volatility the
  geometric-Asian control and the telescoping likelihood ratio both lose their
  closed forms.
- **The arithmetic-Asian reference is itself an RQMC estimate** (64
  independent scrambles, SE 6.4 × 10⁻⁵). It is independent of the sweep's
  scrambles, and it agrees with the control-variate mean (3.16287, 0.7 SE) and
  with Stage 3's 2M-path plain MC (3.16252 ± 0.0043, 1 SE). A bias shared by all RQMC
  runs cannot be excluded by construction, but nothing suggests one.
- **Estimators validated only for the exact scheme.** Every experiment
  simulates with the exact log-GBM scheme. Two estimators would be biased under
  the other schemes, and we did not add guards for this:
  - importance sampling under Milstein, where the drift shift also changes the
    correction term, so the likelihood ratio no longer undoes it (about 0.02 on
    `E[S_T]` at daily steps with the deep barrier's θ; §6.1);
  - the control variate under Euler–Maruyama or Milstein, whose control means
    are exact-GBM prices, so the control's own O(Δt) bias leaks in (§4.1).
- **Minor analysis caveats.** The barrier-bias sweep (§4.4) reuses seeds across
  its three step counts, so the levels are weakly correlated.
  `benchmark.summarize` reports 0% (not NaN) coverage for RQMC rows; Chapters 5
  and 7 override it, but other callers must too.
- **Reconstructing the paper.** The base paper states no equations, step count,
  seed or Asian parameters. Our replication fixes these at the most plausible
  values (daily steps, K = 105, the barrier's market parameters). Every verdict
  in Chapter 3 is phrased to hold under the alternatives we could identify.

### 8.5 Future work

1. **Combine techniques.** RQMC with a control variate, and RQMC with IS, are
   standard and complementary. Stratification plus RQMC is another option.
   Our results suggest RQMC + IS would dominate on the deep barrier, where each
   alone gains only about 9–21×.
2. **Better IS tuning.** Try cross-entropy or large-deviations drifts, a pilot
   that scales with N, and a θ fixed offline.
3. **Multilevel Monte Carlo** for the discretization error that Chapter 5
   measured. It trades a Δt hierarchy for cost, which is the sampling-side dual
   of what Chapter 7 did.
4. **Richer models and products.** Candidates are Heston or local volatility,
   down-and-out and double barriers, and Greeks, where variance reduction
   matters even more.
5. **More replicates for the master sweep,** so the near-ties in §7.3 become
   rankings.

### 8.6 Conclusion

We rebuilt the base paper's experiment from scratch and validated it against
closed forms. In the master sweep, 84 of 85 method cells land within 3 SE of
their reference, and per-replicate CI coverage is 93–98%. We then extended it with the three techniques the paper
named as future work, and benchmarked all five on an efficiency metric that
charges each technique for its own costs, from a single authoritative,
interleaved run on one machine.

The picture that emerges is simple:

- The base paper's antithetic variates are the weakest technique we measured.
- Control variates and RQMC are 10²–10³ times better wherever the payoff has
  the low-dimensional structure they exploit.
- Importance sampling is the specialist for genuinely rare events.
- The only honest way to compare them is at equal wall-clock time, against a
  reference that matches the simulation's monitoring.

To reach a given accuracy on the barrier option, use RQMC with a Brownian
bridge. On the arithmetic Asian, use the geometric-Asian control variate.

---


---

## References

1. Gottimukkala, S. R. (2024). Optimizing exotic option pricing: Monte Carlo
   simulation and variance reduction techniques. *Preprints.org*.
   doi:10.20944/preprints202409.2256.v1.
2. Black, F., & Scholes, M. (1973). The pricing of options and corporate
   liabilities. *Journal of Political Economy*, 81(3), 637–654.
3. Reiner, E., & Rubinstein, M. (1991). Breaking down the barriers. *Risk*,
   4(8), 28–35.
4. Kemna, A. G. Z., & Vorst, A. C. F. (1990). A pricing method for options
   based on average asset values. *Journal of Banking & Finance*, 14(1),
   113–129.
5. Broadie, M., Glasserman, P., & Kou, S. (1997). A continuity correction for
   discrete barrier options. *Mathematical Finance*, 7(4), 325–349.
6. Boyle, P., Broadie, M., & Glasserman, P. (1997). Monte Carlo methods for
   security pricing. *Journal of Economic Dynamics and Control*, 21(8–9),
   1267–1321.
7. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
   Springer.
8. Kloeden, P. E., & Platen, E. (1992). *Numerical Solution of Stochastic
   Differential Equations*. Springer.
9. Sobol', I. M. (1967). On the distribution of points in a cube and the
   approximate evaluation of integrals. *USSR Computational Mathematics and
   Mathematical Physics*, 7(4), 86–112.
10. Joe, S., & Kuo, F. Y. (2008). Constructing Sobol sequences with better
    two-dimensional projections. *SIAM Journal on Scientific Computing*, 30(5),
    2635–2654.
11. Owen, A. B. (1997). Scrambled net variance for integrals of smooth
    functions. *Annals of Statistics*, 25(4), 1541–1562.
12. Moskowitz, B., & Caflisch, R. E. (1996). Smoothness and dimension
    reduction in quasi-Monte Carlo methods. *Mathematical and Computer
    Modelling*, 23(8–9), 37–54.
13. Caflisch, R. E., Morokoff, W., & Owen, A. B. (1997). Valuation of
    mortgage-backed securities using Brownian bridges to reduce effective
    dimension. *Journal of Computational Finance*, 1(1), 27–46.
14. Kish, L. (1965). *Survey Sampling*. Wiley.
15. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
    Chapman & Hall.
16. Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for
    scientific computing in Python. *Nature Methods*, 17, 261–272.

## Appendix A: Reproducing every number

```bash
pip install -r requirements.txt
pytest tests/ -q                   # 90 tests
python run_all.py --analyze-only   # every figure and table from the committed CSVs (~2 min)
python run_all.py                  # full re-simulation (~25 min, single-threaded)
python run_all.py --analyze-only --only master   # the headline table alone
```

- Every number in this report is quoted from a generated table in
  `results/tables/` (`*_findings.md`, `baseline_paper_claims.md`, the
  `*_summary.csv` files), except the few ad-hoc cross-checks that are labelled
  as such in §3.4–3.5.
- The figure → script → CSV map is in `README.md`.
- Seeds are deterministic (`SeedSequence(402)` keyed by CRC32 of the scenario,
  option and method names, N and the replicate), so a re-simulation reproduces
  every price bit for bit on the same library versions. Timings, and
  therefore efficiencies, depend on the machine. The authoritative timings are
  those of the master sweep (Chapter 7).
