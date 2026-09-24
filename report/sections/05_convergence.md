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
difference (§5.7).

- Estimator: `src/vr_qmc.py`.
- Experiments:
  - `experiments/exp_qmc.py`;
  - `experiments/exp_scheme_order.py`.
- Tests: `tests/test_qmc.py` (30 tests, including the task-0 smoke test).
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
   coordinates those are (§5.5).
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
2. It rejects any N that is not a power of 2 (trap #2).
3. It maps uniforms to normals only through the inverse CDF `ndtri` (trap #3).
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

Both slopes are clearly steeper than `N^-1/2`, as task 0 requires.

### 5.3 Measuring the error correctly: the barrier reference

WORKPLAN §1.7.4 requires RMSE rather than variance. The target matters as much
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

- Stage 2's plain-MC in-out-parity value is 7.0941 ± 0.0002. The difference is
  0.15 combined SE.
- Stage 3's control-variate value is 7.093583 ± 0.00028. The difference is
  1.86 combined SE.

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

The deep barrier has 11× more value exposed to missed crossings, and it loses a
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
efficiency CIs use Stage 2's `bootstrap_efficiency_ratio` (§1.7.3). Timings are
indicative; Stage 5's master sweep is authoritative.

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
    incremental construction pays off only from N ≈ 4096.

**Acceptance.**

- **Within 3 SE (§1.7.1).** An RQMC estimate has no per-run CI, so each cell
  is checked with one t-interval built from its 20 scrambles. Every cell of
  every method lies within 3 SE of the reference (45/45).
- **Coverage (§1.7.2).** The 95% t-interval covers the reference in 14/15
  (plain), 14/15 (rqmc) and 15/15 (rqmc_incremental) cells.
- **Plain-MC per-replicate CIs** cover the 252-date reference in 85–100% of
  replicates per cell. With 20 replicates per cell, that range is consistent
  with 95%.

### 5.5 Why the rates differ by option (trap #4)

WORKPLAN trap #4 predicted slopes of about −0.8 to −1.0 for the Asian and only
−0.5 to −0.7 for the barrier. Our results support the *mechanism* but not the
*grouping*.

- **Deep barrier: −0.60, VRF 45×. This is trap #4 as predicted.** The knock-in
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

**WORKPLAN §6.6 sanity check** ("RQMC should beat MC more clearly on the Asian
than on the barrier"):

- It holds for the genuinely path-dependent deep barrier: 1052× against 45×.
- It fails for the paper barrier, for the reason above.
- We investigated before reporting it, as §6.6 asks, and it is not a bug:
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

### 5.8 What the base paper's scheme comparison should have measured (trap #7)

The base paper concludes that the choice between "Euler", "Euler–Maruyama" and
Milstein "has a negligible impact on simulation accuracy". Our measurements
support the conclusion but not its status as a *finding*:

1. **Euler and Euler–Maruyama.** The paper never writes its schemes down. Under
   the standard definitions, "Euler" for an SDE *is* Euler–Maruyama, so two of
   its three schemes are presumably one scheme under two names. The caveat,
   already noted in §3.4, is that this is an inference.
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

   Stage 2 also found that the paper's Figs. 2–4 reuse the same random numbers.
   Its scheme plots could not have come out differently, and "negligible" was
   the result theory predicts. It is not a discovery. (WORKPLAN's wording,
   that Milstein "cannot change" the expectation, is too strong: it does change
   it, at O(Δt), as measured.)
3. **What should have been measured.** Strong and weak error against Δt, with
   common random numbers and fitted orders, as above. That is the only design
   that separates the schemes, and it shows what each is good for: Milstein
   when pathwise accuracy matters, neither when only a price is needed.
4. **The exact scheme makes both unnecessary.** Under GBM the exact log-GBM
   scheme costs about the same (0.64 s against 0.54 s for Euler–Maruyama and
   0.77 s for Milstein per 65,536 paths at m=512) and has **zero**
   discretization error. The only discretization error left in this project is
   the barrier's *monitoring* bias (§5.3; Stage 3). That is a property of the
   contract's observation dates, not of the SDE scheme.

### 5.9 Summary

| Requirement (WORKPLAN Stage 4) | Evidence |
|---|---|
| Task 0: known integral, slope steeper than `N^-1/2` | Smooth exponential −0.83; Gaussian orthant −0.77; plain MC −0.43 / −0.50 |
| Scrambled Sobol, `random_base2` only, `ndtri`, R independent scrambles | `sobol_uniforms`; non-power-of-2 N raises; one scramble per replicate; within-run SE logged as NaN |
| Brownian bridge | Orthogonal bridge matrix; tests for dimension 0 → `W_T` and dimensions 0–1 → midpoint |
| Headline RQMC vs MC with fitted slopes | Asian −0.72, paper barrier −0.80, deep barrier −0.60 (plain ≈ −0.5); VRF at 65,536 = 1052 / 6988 / 45 |
| Explain the barrier rate (trap #4) | Deep barrier matches trap #4; paper barrier ≈ a 1-D vanilla call under the bridge; variance-share table |
| Bridge vs incremental | Bridge 27–536× lower variance (Asian, paper barrier), 5–10× (deep barrier) |
| Strong / weak order | Strong 0.49 / 0.97; weak 0.99–1.03 for both schemes; `E[S_T]` MC matches the analytic formula |
| Trap #7 write-up | §5.8 |
| Validation (§1.7) | 45/45 cells within 3 SE; t-interval coverage 43/45 |

**Recommendation for Stage 5.**

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
