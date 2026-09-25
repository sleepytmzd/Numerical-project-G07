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
