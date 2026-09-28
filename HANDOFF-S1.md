# Handoff Log

Append an entry at the end of your stage, before you push. Keep it short: what you added, anything
surprising, anything the next person must know. See `WORKPLAN.md` §1.3 for the Definition of Done.

---

## Stage 0 — Kickoff (everyone)

- Repo skeleton created: `src/`, `experiments/`, `tests/`, `results/{raw,figures,tables}/`,
  `report/sections/`.
- `WORKPLAN.md` committed — this is the authoritative plan. Read your stage before starting, and
  read §3 (Traps) regardless of which stage you own.
- `requirements.txt` pinned. Install with `pip install -r requirements.txt`.
- Base paper and course outline are in `resources/`. Proposal is in `report/proposal.md`.

**Next:** Stage 1 (Arnob) — `src/config.py` and `src/results.py` first, since everything downstream
imports them.

---

## Stage 1 — Arnob Biswas (2105015)

**Date:** 2026-09-24

### What was added

1. **`src/config.py`** — `Scenario` dataclass (frozen) and all six scenarios from §1.5:
   `paper_barrier`, `mid_barrier`, `deep_barrier`, `paper_asian_geo`, `asian_30d`,
   `asian_arith`. Constants: `N_GRID`, `N_STEPS=252`, `R=20`, `BASE_SEED=402`.

2. **`src/results.py`** — `log_result(**fields)` writes to
   `results/raw/<person>_<experiment_id>.csv` with the §1.6 schema (auto-creates header).
   `load_all_results()` globs everything into one DataFrame. Uses `uuid` for run_id and
   `platform` for machine_id.

3. **`src/analytic.py`** — All closed-form pricers:
   - `bs_call` / `bs_put` — standard Black–Scholes.
   - `barrier_closed_form(S0, K, B, r, sigma, T, kind, option, q)` — Reiner–Rubinstein
     for up-and-in and up-and-out, call and put. Implementation follows the **exact QuantLib
     AnalyticBarrierEngine** component decomposition (A, B, C, D with phi/eta flags).
   - `barrier_closed_form_bgk(...)` — same but with BGK shift: `B_adj = B * exp(+0.5826 * sigma * sqrt(dt))` for up-barriers. Stage 3 needs this for the discretisation bias study.
   - `geometric_asian_closed_form(..., m=None)` — Kemna–Vorst with **both** continuous
     (`m=None`) and discrete (`m=252`) variants. This is critical — trap #1 in the WORKPLAN.

4. **`tests/test_analytic.py`** — 14 tests:
   - Pins all §2 reference values (vanilla call, UIC, UOC, geometric Asian).
   - Put–call parity for BS and geometric Asian (both continuous and discrete).
   - **In-out parity at THREE barrier levels**: `paper_barrier` (B=110.6772), `mid_barrier`
     (B=130), and `deep_barrier` (B=140). The `paper_barrier` test is weak (trap #6) because
     `C_uo ≈ 0.023` is only 0.3% of the vanilla price.
   - BGK correction: checks it's < continuous, ≈7.093, and converges as n_steps → ∞.
   - Discrete vs continuous Asian: checks they differ but converge.

5. **`README.md`** — install instructions, experiment protocol summary, full project
   structure, and figure→script→CSV map (stub entries for later stages).

6. **`report/sections/01_introduction.md`** — Introduction, theoretical background,
   related work, and the "what the base paper did vs what we add" comparison table.

### Gotchas / surprises

- **The barrier formula is subtle.** The standard "A - B + D" decomposition from many
  textbooks does NOT work for up-and-in calls when K < B. The correct QuantLib decomposition
  for UIC (K < B) is `B(1) - C(-1, 1) + D(-1, 1)`. I verified this against the actual
  QuantLib C++ source code. If you ever need to debug, the key insight: `eta = -1` for up
  barriers (not +1 as some sources claim).

- **BGK sign**: the correction is `+beta*sigma*sqrt(dt)` for UP barriers (shifts the barrier
  outward = upward). This **reduces** the UIC price because a higher barrier is harder to hit.

- **Discrete Kemna–Vorst formula**: `sigma_G^2 = sigma^2 * (m+1)*(2m+1) / (6*m^2)` and
  `b = 0.5 * sigma_G^2 + (r - sigma^2/2) * (m+1) / (2*m)`. The continuous formula uses
  `sigma_G = sigma/sqrt(3)` and `b = 0.5*(r - sigma^2/6)`. At m=252, they differ by about
  0.01 — small but visible if you're chasing biases (trap #1).

### For Stage 2 (Nafis)

- **Import paths**: `from src.config import SCENARIOS, N_GRID, N_STEPS, R, BASE_SEED` and
  `from src.analytic import bs_call, barrier_closed_form, geometric_asian_closed_form`.
- **Results logging**: use `from src.results import log_result` with keyword arguments only.
  The `person` field should be your name.
- **The `Scenario` dataclass** has `avg_start_idx` for the last-30-day Asian. When computing
  geometric averages in `payoffs.py`, use `paths[:, scenario.avg_start_idx:]`.
- **`geometric_asian_closed_form` with `m=252`** is the correct benchmark for MC with 252
  daily monitoring points. Do NOT compare against the continuous formula (trap #1).
- **Verify `E[S_T] ≈ S0*exp(r*T)` = 103.045** and `Var[log S_T] ≈ sigma^2*T = 0.04`.

## Stage 2 — Nafis Nahian (2105007)

**Date:** 2026-09-24

### What was added

1. **`src/paths.py`** — `generate_paths(scenario, n_paths, n_steps, normals, scheme,
   drift_override)`. Fully vectorized `exact` (log-GBM), `euler_maruyama`, and
   `milstein` schemes. Frozen interface exactly per WORKPLAN §1.4: randomness is
   always a pre-built `(n_paths, n_steps)` array, never drawn internally.
   `drift_override` replaces `r` as the simulation drift `mu` — Stage 5 will pass
   `r + sigma*theta` for importance sampling.

2. **`src/payoffs.py`** — `barrier_payoff` (discrete monitoring: barrier checked
   at every simulated grid point, not continuity-corrected), `geometric_asian_payoff`
   (with `avg_start_idx`, so `asian_30d` works), and `payoff_for(scenario, option)`,
   a dispatcher that looks up `arithmetic_asian_payoff` **by name at call time** —
   Stage 3 only needs to append that one function to this file; nothing else
   changes.

3. **`src/estimators.py`** — `EstimateResult`; `plain_mc`; `antithetic_mc` (builds
   `normals = [Z, -Z]`, `n_paths` is the *total* path count = 2×pairs). Both
   generate paths in 8192-path chunks so nothing allocates a full `(65536, 253)`
   array. `make_seed_seq(experiment_id, *keys)` derives a deterministic
   `SeedSequence` from `BASE_SEED=402` via a CRC32-keyed spawn, never `np.random.seed`.
   `run_sweep` keys each cell by CRC32(scenario name, option, method), N and the
   replicate. The scheme is deliberately **not** in the key, so all schemes share
   normals and scheme comparisons are paired.

4. **`src/benchmark.py`** — `ESTIMATORS` registry + `register(name)` decorator
   (Stages 3–5 register their own estimator here; `run_sweep` never hard-codes
   the method list). `exact_price(scenario, option, n_steps)` — barrier uses
   Stage 1's continuous Reiner–Rubinstein formula; geometric Asian uses a
   **general discrete Kemna–Vorst-style formula over the scenario's actual
   averaging window** (not just full-window m=252), which is the only correct
   benchmark for `asian_30d`. `run_sweep(...)` is the generic
   case×scheme×method×N×replicate loop that logs one row per replicate.
   `summarize(df)` computes variance/bias/RMSE/coverage/efficiency **across
   replicates** per cell. `bootstrap_efficiency_ratio(df, keys_a, keys_b)` gives a
   95% bootstrap CI for an efficiency ratio (§1.7.3). It now **raises unless each
   side selects exactly one cell**, so you must include `n_paths` in the keys (see
   the review fixes below). `fit_loglog_slope(n, y)`.

5. **`src/plots.py`** — shared style: fixed `METHOD_COLORS`/`METHOD_LABELS` for
   all five methods (plain, antithetic, control_variate, rqmc, importance) so
   every stage's figures match; `ci_vs_n_plot` (paper's Fig 2–4 style) and
   `loglog_rmse_plot` (with fitted slope + N^-1/2 reference line).

6. **`tests/test_paths.py`, `tests/test_estimators.py`** — 32 new tests:
   - engine: shape and moment checks, `drift_override`;
   - schemes: paired scheme-vs-exact agreement, Milstein strong error < 5% of
     Euler's, Euler strong-order slope ≈ 0.5;
   - payoffs and formulas: barrier in/out complementarity; general discrete
     geometric formula == Stage 1's KV at m=252; 30-day window MC vs formula;
   - estimators: plain/antithetic within 3 SE; antithetic SE < plain SE;
     plain and antithetic chunking invariance;
   - sweep: seeds stable across processes with different hash salts; no seed
     shared across cases or methods; schemes share normals; `run_sweep` smoke
     test; bootstrap refuses to pool across N.

   All 46 tests (14 Stage-1 + 32 Stage-2) pass in ~13s.

7. **`experiments/exp_baseline.py`** — sweeps `paper_barrier` (all 3 schemes) and
   `paper_asian_geo` call/put + `asian_30d` (exact scheme) × {plain, antithetic}
   × N_GRID × R=20 (~180s total). Writes `results/raw/nafis_baseline.csv` (1200
   rows), the 4 required figures plus `baseline_rmse_vs_n.png` and
   `baseline_scheme_comparison.png`, `results/tables/baseline_summary.csv`, and
   `results/tables/baseline_paper_claims.md` (the task-8 paper-claim
   verification, including two high-precision reference runs). It deletes its
   own CSV before simulating, since `log_result` appends and a re-run would
   otherwise duplicate rows. Supports `--analyze-only` to replot without
   re-simulating.

8. **Report sections `02_methodology.md`, `03_baseline.md`** written with the
   actual numbers from this run.

### Review and verification pass (bugs found and fixed)

Before handing off, the Stage-2 results were re-verified in three ways:
- an independent code review;
- from-scratch recomputation in code that does not import `src/`;
- a check of the paper's actual figures.

This found real bugs. All are fixed, and the experiment was re-run.

1. **Seeds were not reproducible.** The seed key used Python's `hash()`, which
   is salted per process, so every run drew different numbers. It now uses
   CRC32, and a test checks stability across two hash salts.
2. **Scenarios shared random numbers.** Seeds were keyed on the list position
   `case_idx`, so two separate `run_sweep` calls reused streams: 200 of 1000
   seeds were shared between the barrier and the Asian call. Seeds are now
   keyed by name.
3. **The antithetic speedup numbers were wrong.** The bootstrap pooled all five
   N values into one "cell", which produced a meaningless ratio (e.g. "0.75×"
   for the Asian call). Two changes:
   - it is now computed per N;
   - `bootstrap_efficiency_ratio` raises if asked to pool.
4. **Overclaims in the write-up.** The scheme comparison used a pooled-over-N
   mean and a mislabelled "typical SE". Several statements were too strong
   ("provably cannot change", "the claim doesn't hold up"). They are replaced
   with paired (common-random-number) measurements and carefully worded
   verdicts.
5. **Weak tests.** A sign-flipped Milstein correction passed the old scheme
   test. Strong-error tests now catch it; this was checked by mutation.

### Final, verified findings (see `report/sections/03_baseline.md`)

**Antithetic variates.** Measured at equal total path count, the variance
reduction factor (VRF) is flat across all five N and matches a 4M-path
independent run to 2 decimals:

| Option | VRF | corr(Y, Y′) | efficiency gain |
|---|---|---|---|
| Barrier | 1.47 | −0.32 | 1.77× |
| Asian call | 1.35 | −0.26 | 1.58× |
| Asian put | 3.91 | −0.74 | 4.54× |
| 30-day Asian | 1.48 | −0.32 | 1.78× |

Antithetic is ~1.2× cheaper per path, which is why the efficiency gain exceeds
the VRF.

- **The paper's 1.5× / 1.3× is confirmed** for the barrier and Asian call, read
  as a variance ratio.
- **The Asian put (3.9×) does not match.** The paper's put figure (Fig. 8)
  appears to be a copy of its call figure (Fig. 6), with an exact line at ≈3.0
  where the true put price is 6.708.
- **Use within-run SEs for antithetic VRF.** With R=20, the across-replicate
  bootstrap ratio cannot resolve 1.3 vs 1.5 (barrier @65536: 0.85
  [0.38, 1.91]). Use the within-run VRF, the ratio of mean SE², wherever the
  within-run SE is valid.

**"Stabilization" at 5000 / 750–1000 paths.** Not supported as a convergence
property:
- `RMSE·√N` is flat and the slopes are ≈ −0.5;
- at those N the 95% CI is still ±4.9% (barrier) and ±12–14% (Asian);
- ±1% needs ~120k–146k paths.

We describe it as a visual impression, not a "wrong" claim, because the paper
gives no criterion.

**Scheme choice.** The paired differences from exact at dt=1/252 are
Euler–Maruyama −0.0006 ± 0.0002 and Milstein −0.0011 ± 0.00001. That makes the
choice negligible, and it is expected, since both schemes have weak order 1.

- Caveat on "Euler = Euler–Maruyama": the paper never writes its schemes down,
  so this is an inference.
- Its Figs. 2–4 show identical sample points (the same random numbers were
  reused), so identical plots were guaranteed.

**30-day Asian.** The price is 6.708, against 3.000 for full averaging. MC
agrees (6.698 ± 0.007; independent run 6.7067 ± 0.0049). The paper's statement
that the closed form "remains identical" is false if read as the same value.

**⚠ WORKPLAN §2 / trap #5 has a wrong reference value.** The 252-step discrete
up-and-in price is **7.0941 ± 0.0002**, computed by in-out parity as
`BS − E[discrete UOC]` with 4M paths and confirmed independently twice. It is
**not ≈7.076**.

- The discretization bias is **−0.011**, not −0.03.
- BGK (7.0930) is accurate to 0.001.
- Plain-MC SE at N=1024 is 0.39, about 34× the bias, so the trap-#5 narrative
  holds even more strongly.

- **README's figure→script→CSV map had wrong CSV names for my rows**
  (`aritra_baseline.csv` instead of `nafis_baseline.csv`) — fixed those 4 rows
  and added 2 missing rows for the extra figures. **The Stage 3/4 rows in that
  same map look swapped too** (`tamzeed_control.csv` for what's Aritra/Stage-3's
  control-variate figures; `nafis_qmc.csv`/`nafis_scheme_order.csv` for what's
  Tamzeed/Stage-4's QMC and scheme-order figures) — I left those alone since
  they're not my file rows to fix; Aritra and Tamzeed should double check and
  correct them when they push.
- **Environment note:** ran on system Python 3.12.3 with numpy 1.26.4 / scipy
  1.11.4 (not the exact versions pinned in `requirements.txt`, which weren't
  installed in this environment) — no issues encountered, but worth confirming
  on your own machine with the pinned versions.

### For Stage 3 (Aritra)

- **Append-only edit**: add `arithmetic_asian_payoff(paths, K, option="call",
  avg_start_idx=1)` to `src/payoffs.py` — the `payoff_for` dispatcher already
  looks it up by name (`globals().get("arithmetic_asian_payoff")`), so nothing
  else needs to change once it exists.
- **Register your estimator**: `from src.benchmark import register` then
  `register("control_variate")(your_fn)`. Your function's signature must match
  `estimator(scenario, n_paths, n_steps, seed_seq, *, option=None, scheme="exact",
  **params) -> EstimateResult` — reuse `src.estimators.summarize_samples` to build
  the `EstimateResult` if convenient, and put `rho`/`b_hat` in `extra`.
- **`exact_price(scenario, option, n_steps)`** from `src.benchmark` gives you the
  closed-form barrier/geometric-Asian price (`None` for arithmetic Asian, as
  expected — that's your headline experiment). Use `barrier_closed_form_bgk`
  from `src.analytic` directly for the BGK study (task 6).
- **Barrier-bias target (task 6): use 7.0941, not the WORKPLAN's 7.076.** The
  bias you should be resolving is ≈ −0.011. For your n_steps ∈ {12, 52, 252}
  sweep, compute references the same way (`exp_baseline.discrete_barrier_reference`
  does in-out parity via the vanilla BS price, which is very precise). Note that
  `exact_price` for barriers returns the *continuous* value.
- **`run_sweep`** is generic — call it with `methods=["control_variate"]` (or
  mixed with `["plain", "control_variate"]` for a direct comparison) and your own
  `experiment_id`; it logs to `results/raw/aritra_<experiment_id>.csv`
  automatically from the `person` argument you pass.
- **Use `src.plots.apply_style()` and the shared `ci_vs_n_plot`/`loglog_rmse_plot`**
  so your figures match the rest of the report; `METHOD_COLORS["control_variate"]`
  is already reserved.
- **`benchmark.summarize` and `benchmark.bootstrap_efficiency_ratio`** are ready
  to use for your VRF/efficiency tables — don't recompute variance/RMSE by hand.
  Pass keys that pin down one cell each, **including `n_paths`**. With R=20 the
  bootstrap CI is wide, so also report a within-run VRF (mean SE² ratio at the
  same N), which is valid for your CV estimator if its SE comes from i.i.d.
  samples.
- Deep-barrier scenario (`deep_barrier`, B=140) is already in `config.SCENARIOS`.

### For Stage 4 (Tamzeed)

- Same registry pattern: `register("rqmc")(your_fn)`. `generate_paths` accepts
  `normals` from any source, including a scrambled-Sobol + `ndtri` array — you
  don't need to touch `paths.py`.
- **Brownian bridge**: since `normals` columns map 1:1 to time steps for the
  `exact` and `euler_maruyama`/`milstein` schemes as I implemented them, your
  bridge construction should build a `(n_paths, n_steps)` array of *effective*
  per-step normal increments from the Sobol dimensions (fill terminal, then
  midpoints, ...) before calling `generate_paths` — the engine itself has no
  bridge-awareness, by design, so this logic belongs entirely in `vr_qmc.py`.
- Reuse `fit_loglog_slope` from `benchmark.py` for your RMSE-vs-N slope fits
  (expect ≈−0.9 Asian, ≈−0.6 barrier per WORKPLAN trap #4).
- The scheme-order study (task 5) can reuse `generate_paths` directly with
  varying `n_steps` and a fixed `T` — no changes needed there either.

### For Stage 5 (Zaki)

- `drift_override` in `generate_paths` is exactly the lever you need:
  `generate_paths(scenario, n, n_steps, Z, scheme, drift_override=scenario.r +
  scenario.sigma*theta)` simulates under the tilted measure. Compute the
  likelihood ratio yourself in `vr_importance.py` from the terminal `W_T`
  implied by the *same* `normals` array you passed in (don't regenerate `Z`).
- `benchmark.summarize` and `bootstrap_efficiency_ratio` are what your master
  comparison (task 4) should build on — they already compute RMSE (not just
  variance) and bootstrap CIs on efficiency ratios. Compare methods **per N**
  (the function refuses pooled keys). If a bootstrap CI is too wide at R=20,
  either raise R for the master run or also report within-run SE² ratios for
  the estimators where the within-run SE is valid (all except RQMC).
- `deep_barrier` (B=140) is already in `config.SCENARIOS`, ready for your
  headline rare-event experiment.
- Chunked path generation (8192-path blocks, `estimators.py`) is already in
  place in `plain_mc`/`antithetic_mc` — mirror that pattern in your own
  estimator so the master sweep at N=65536 doesn't blow up memory.

### General note for everyone

All 46 tests pass and `experiments/exp_baseline.py` runs end-to-end in ~4
minutes on this machine, including ~1 min of high-precision reference runs
(`OMP_NUM_THREADS=1`). **Nothing from this
stage has been committed or pushed** — Nafis wanted to review the diff first,
so `git status` on `Numerical-project-G07/` will show these files as new/
untracked until that review happens. Suggested commit message when ready:
`Stage 2 (Nafis Nahian): paths, payoffs, estimators, benchmark, plots, baseline experiment`.

## Stage 3 — Aritra Debnath

**Date:** 2026-09-24

### What was added

1. **`src/payoffs.py`** — appended `arithmetic_asian_payoff`, averaging the same
   monitoring slice used by the geometric payoff. The existing dynamic
   `payoff_for` dispatcher now handles `asian_arith` without another source edit.

2. **`src/vr_control.py`** — registered `"control_variate"` with the generic
   benchmark registry. It uses the discrete geometric Asian as the arithmetic-
   Asian control and the terminal European option as the barrier control.
   `b_hat = Cov(Y,X)/Var(X)` is fitted on an independent pilot of
   `max(256, ceil(0.1*N))` paths. Pilot work is timed and logged. `extra_json`
   contains `rho`, `b_hat`, control mean/type, pilot and total path counts,
   implied plain SE, observed within-run VRF, and theoretical `1/(1-rho²)`.

3. **Tests** — added `tests/test_control.py` (four tests), including the required
   synthetic correlated-normal coefficient/VRF smoke test. Replaced Stage 2's
   temporary arithmetic `NotImplementedError` assertion in `test_paths.py` with
   a positive dispatcher test. **All 50 tests pass.**

4. **`experiments/exp_control.py`** — plain vs control-variate sweeps on
   `asian_arith`, `paper_barrier`, and `deep_barrier` across `N_GRID × R=20`,
   plus a separate 2,000,000-path plain-MC arithmetic reference. Writes
   `results/raw/aritra_control.csv` (601 rows), three figures, and
   `results/tables/control_{summary.csv,findings.md}`.

5. **`experiments/exp_barrier_bias.py`** — control-variate barrier estimates at
   12, 52, and 252 monitoring dates, `N=65536`, `R=20`. Writes
   `results/raw/aritra_barrier_bias.csv` (60 rows),
   `barrier_bias_vs_steps.png`, and the barrier-bias summary/findings tables.

6. **`report/sections/04_control_variates.md`** — complete Stage-3 chapter:
   estimator theory, independent-pilot design, arithmetic validation, near/deep
   barrier comparison, theoretical-vs-observed VRF, BGK bias study, finite-
   sample coefficient note, and acceptance summary.

7. **Documentation** — `readme-AD.md` gives the concise reproducibility and
   handoff summary. README's four Stage-3 figure→CSV rows now correctly point
   to Aritra's raw files rather than Tamzeed's.

### Verified findings

At `N=65536`:

| Scenario | Price | rho | b_hat | within-run VRF |
|---|---:|---:|---:|---:|
| Arithmetic Asian call | 3.162776 | 0.999488 | 1.045681 | **975.7×** |
| Paper barrier (`B=110.6772`) | 7.094189 | 0.999675 | 1.000879 | **1542.4×** |
| Deep barrier (`B=140`) | 3.379102 | 0.830485 | 0.763175 | **3.2×** |

- The arithmetic result agrees with the 2M-path plain reference
  `3.162517 ± 0.008511` (95% CI); the difference is 0.06 combined SE.
- The barrier-control collapse is real and expected: the paper-barrier payoff
  almost equals the vanilla call, whereas many vanilla calls do not knock in at
  `B=140`.
- The 252-date control estimate is 7.093583: bias **−0.011945** from the
  continuous value. BGK is 7.092995, just 0.000588 lower. This independently
  confirms Stage 2's corrected ≈7.0941 target, not WORKPLAN's obsolete 7.076.
- The daily bias is resolved at 54.1 SE of the replicated mean; a same-size
  plain run has SE ≈0.0489 and cannot reveal it.
- BGK is excellent at 252 dates but only asymptotic: MC−BGK is +0.0343 at 12
  dates and +0.0061 at 52 dates.

### For Stage 4 (Tamzeed)

- Import `src.vr_control` before asking the registry for `control_variate`; its
  decorator performs registration without editing `benchmark.py`.
- Preserve `extra_json` when reusing these rows. `n_paths` means production
  paths; pilot/total path counts are explicitly logged for cost auditing.
- Use the within-run VRFs for precise variance-reduction claims. R=20 makes
  across-replicate variance ratios and bootstrap efficiency CIs noisy.
- Stage-3 timing is indicative only; Stage 5's one-machine master sweep remains
  authoritative under the frozen protocol.
- `exp_control.py --analyze-only` and `exp_barrier_bias.py --analyze-only`
  regenerate every Stage-3 table/figure without resimulation.

## Stage 4 — Tamzeed Mahfuz

_(fill in)_

## Stage 5 — Zaki Rehnoom Unmona

_(fill in)_

