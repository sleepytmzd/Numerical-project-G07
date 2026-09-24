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

For a discounted target payoff (Y), discounted control payoff (X), and
known control value (mu_X=E[X]), the estimator is

\[
    \widehat V_{CV} = \frac{1}{N}\sum_{i=1}^{N}
    \left[Y_i-\widehat b(X_i-\mu_X)\right],
    \qquad
    b^*=\frac{\operatorname{Cov}(Y,X)}{\operatorname{Var}(X)}.
\]

At the optimal coefficient, its theoretical variance-reduction factor is

\[
    \mathrm{VRF}
    =\frac{\operatorname{Var}(Y)}{\operatorname{Var}(Y-b^*(X-\mu_X))}
    =\frac{1}{1-\rho_{YX}^{,2}}.
\]

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
VRF, and (1/(1-\rho^2)).

Before using option payoffs, the coefficient calculation was tested on
synthetic correlated normals (Y=1.7X+0.55\epsilon). The fitted coefficient
recovers 1.7, and both the measured and correlation-predicted VRFs recover the
known value (1+1.7^2/0.55^2\). This test is retained in
`tests/test_control.py`.

### 4.2 Arithmetic Asian: the headline result

The arithmetic payoff was added as the mean of the 252 simulated monitoring
prices (excluding (S_0)), followed by the usual call payoff. Plain MC and the
control estimator were each run at
`N=[256, 1024, 4096, 16384, 65536]`, with `R=20` independent replications per
cell. Because the arithmetic option has no analytic benchmark, an additional
2,000,000-path plain-MC run provides an independent validation value.

At (N=65{,}536):

| Quantity | Result |
|---|---:|
| Control-variate price, mean of 20 replications | **3.162776 ± 0.000361** (95% CI for the replicated mean) |
| 2,000,000-path plain-MC reference | **3.162517 ± 0.008511** (95% CI) |
| Difference in combined standard errors | **0.06** |
| Correlation (ho(Y,X)) | **0.999488** |
| Pilot coefficient `b_hat` | **1.045681** |
| Observed within-run VRF | **975.7×** |
| Theoretical (1/(1-\rho^2)) | **976.6×** |

Thus the geometric control removes about 99.90% of the target payoff variance.
The independently estimated arithmetic price agrees with the much larger plain
run, satisfying the validation requirement without pretending that the
arithmetic option has a closed form. `cv_asian_vrf.png` shows that the gain is
stable at approximately (10^3) across the complete N grid.

### 4.3 Why the barrier control succeeds—and then collapses

The same European-call control was applied to two up-and-in barriers. Results
at (N=65{,}536) are:

| Scenario | CV price | rho | b_hat | observed VRF | theory | across-replicate VRF |
|---|---:|---:|---:|---:|---:|---:|
| Paper barrier, (B=110.6772) | 7.094189 | 0.999675 | 1.000879 | **1542.4×** | 1542.4× | 1365.6× |
| Deep barrier, (B=140) | 3.379102 | 0.830485 | 0.763175 | **3.2×** | 3.2× | 5.7× |

For the paper barrier, the continuously monitored up-and-out leg is worth only
about 0.023. The up-and-in payoff therefore almost equals the terminal vanilla
call path by path, giving `b_hat ≈ 1` and near-perfect correlation. At
(B=140), many positive vanilla-call payoffs do not knock in; the two payoffs
no longer track each other closely, so the VRF collapses to about 3.2. This is
the central pedagogical result in `cv_barrier_collapse.png`: a control variate
is not intrinsically good or bad—its value depends on its correlation with the
target under the specific scenario.

`cv_rho_vs_vrf.png` compares every scenario/N cell with the theoretical curve.
The observed within-run factors closely follow (1/(1-\rho^2)), including the
sharp growth as (ho\to1). The across-replicate ratios fluctuate more because
each cell contains only 20 price estimates. At the largest N, the bootstrap
efficiency ratios (control over plain, including pilot runtime) are 405.5
[169.6, 991.8] for the arithmetic Asian, 3506.8 [1691.5, 7047.1] for the paper
barrier, and 5.1 [1.8, 14.8] for the deep barrier. These Stage-3 timings are
indicative; per the frozen protocol, Stage 5's single-machine master run is the
authoritative efficiency comparison.

### 4.4 Discrete barrier-monitoring bias

The continuous Reiner–Rubinstein paper value is 7.105528, whereas a barrier
observed only on a grid is less likely to be hit. The control estimator was run
with (N=65{,}536), (R=20), and 12, 52, or 252 monitoring dates. The BGK
continuity correction shifts an up-barrier outward by
(B\exp(0.5826\sigma\sqrt{\Delta t})) before applying the continuous formula.

| Monitoring dates | CV price | measured bias vs continuous | BGK price | MC − BGK | CV SE per run | implied plain SE per run |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 7.042920 | −0.062608 | 7.008629 | +0.034291 | 0.002086 | 0.049044 |
| 52 | 7.078373 | −0.027155 | 7.072275 | +0.006098 | 0.001556 | 0.049047 |
| 252 | **7.093583** | **−0.011945** | **7.092995** | **+0.000588** | **0.001260** | **0.048851** |

At daily monitoring the result agrees with Stage 2's independent
in-out-parity reference of approximately 7.0941. The replicated mean resolves
the −0.01195 bias at 54.1 standard errors, while even one 65,536-path plain run
has an SE about four times larger than the bias. This is precisely the feature
plain MC obscures. `barrier_bias_vs_steps.png` also shows that BGK is highly
accurate at 252 dates but less accurate on coarse grids; it is an asymptotic
continuity correction, not an exact discrete-barrier formula.

The workplan's preliminary value of about 7.076 (bias −0.03) was superseded by
Stage 2's high-precision result and is not used here. The Stage-3 estimate
independently confirms the corrected daily-monitoring bias of about −0.012.

### 4.5 Finite-sample coefficient note

Estimating the optimal coefficient from the same sample used in the adjusted
mean can introduce an (O(1/N)) finite-sample bias through the random ratio
`sample_covariance/sample_variance`. At the sample sizes used here it would be
negligible relative to Monte Carlo error. Our implementation goes further: it
fits `b_hat` on an independent pilot, so conditional on that coefficient
(E[X-\mu_X]=0) in the production sample and the adjusted estimator remains
unbiased. The tradeoff is extra pilot cost and a small variance penalty if the
pilot coefficient differs from (b^*); both are included in the logged data.

### 4.6 Acceptance summary

| Requirement | Evidence |
|---|---|
| Coefficient/VRF smoke test | Synthetic-normal test recovers known `b*` and VRF |
| Arithmetic option without closed form | 3.162776; agrees with 2M-path plain reference within 0.06 combined SE |
| Near/deep barrier contrast | VRF 1542.4× versus 3.2× |
| Theory versus observed reduction | All cells follow (1/(1-\rho^2)) |
| Resolve daily barrier bias | −0.011945, measured at 54.1 SE of the replicated mean |
| BGK application | 252-step MC differs from BGK by only +0.000588 |
| Reproducibility | 20 raw rows per frozen grid cell; deterministic `SeedSequence(402)` hierarchy; pilot metadata retained |
