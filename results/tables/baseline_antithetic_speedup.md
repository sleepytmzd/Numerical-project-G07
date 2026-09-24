# Baseline: paper-claim verification

## Claim 1 — convergence 'stabilizes' at ~5000 (barrier) / 750-1000 (Asian)

- paper_barrier/call: fitted RMSE~N slope = -0.48 (no plateau near N=5000; theory predicts -0.5 everywhere). RMSE keeps shrinking well past the paper's claimed 'stabilization' point — it is a visual impression from a log-x plot, not a real threshold.

- paper_asian_geo/call: fitted RMSE~N slope = -0.50 (no plateau near N=1000; theory predicts -0.5 everywhere). RMSE keeps shrinking well past the paper's claimed 'stabilization' point — it is a visual impression from a log-x plot, not a real threshold.

## Claim 2 — antithetic speedup 1.5x (barrier), 1.3x (Asian)

- paper_barrier/call: efficiency ratio (antithetic/plain) = 1.30 [0.66, 2.72] (95% bootstrap CI), paper claims 1.5x — consistent with the paper.

- paper_asian_geo/call: efficiency ratio (antithetic/plain) = 0.75 [0.40, 1.61] (95% bootstrap CI), paper claims 1.3x — consistent with the paper.

- paper_asian_geo/put: efficiency ratio (antithetic/plain) = 3.67 [1.83, 7.09] (95% bootstrap CI), paper claims 1.3x — DIFFERS from the paper.

## Claim 3 (implicit) — discrete monitoring bias is invisible at small N

- Continuous benchmark = 7.1055, BGK-corrected = 7.0930 (discrete monitoring bias ~= 0.0125).
- Large-N MC mean = 7.0913 (bias vs continuous = -0.0142), while a small-N run has SE ~= 0.390 — far larger than the bias. Plain MC at small N cannot distinguish 'discretely monitored' from 'continuously monitored'; the bias only becomes visible once variance reduction shrinks the CI below it (picked up again in Stage 3).

## Claim 4 — scheme choice (Euler/Euler-Maruyama/Milstein) is negligible

- Mean price by scheme: {'euler_maruyama': 7.1207, 'exact': 7.0686, 'milstein': 7.03}; max spread = 0.0907, typical MC SE = 0.793. The spread is within MC noise, confirming the paper's finding — but note 'Euler' and 'Euler-Maruyama' are mathematically the same scheme, and Milstein's strong-order gain (0.5->1.0) provably cannot change a European-style expectation. So 'scheme choice is negligible' is a tautology, not an empirical discovery (Stage 4 quantifies this properly via strong/weak discretization order).
