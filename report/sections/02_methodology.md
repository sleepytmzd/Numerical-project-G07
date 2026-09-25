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
