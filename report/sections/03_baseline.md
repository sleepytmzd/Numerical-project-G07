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
