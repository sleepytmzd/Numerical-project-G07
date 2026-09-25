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
