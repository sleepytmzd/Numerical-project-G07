"""
Stage 2 baseline experiment (Nafis Nahian, 2105007).

Replicates the base paper's setup: plain MC and antithetic variates on the
up-and-in barrier call (all 3 discretization schemes) and the geometric Asian
call/put (full average + last-30-day window), across N_GRID x R=20.

Usage
-----
    OMP_NUM_THREADS=1 python3 experiments/exp_baseline.py            # simulate + plot
    OMP_NUM_THREADS=1 python3 experiments/exp_baseline.py --analyze-only  # plot only
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytic import bs_call
from src.benchmark import (
    bootstrap_efficiency_ratio,
    exact_price,
    exact_price_bgk,
    fit_loglog_slope,
    run_sweep,
    summarize,
)
from src.config import N_GRID, N_STEPS, R, SCENARIOS
from src.estimators import make_seed_seq
from src.paths import SCHEMES, generate_paths
from src.payoffs import barrier_payoff, payoff_for
from src.plots import apply_style, ci_vs_n_plot, loglog_rmse_plot, save_fig
from src.results import _RESULTS_DIR, load_all_results

PERSON = "nafis"
EXPERIMENT_ID = "baseline"

TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

CASES = [
    ("paper_asian_geo", "call"),
    ("paper_asian_geo", "put"),
    ("asian_30d", "call"),
]
METHODS = ["plain", "antithetic"]


def simulate():
    """Barrier (3 schemes) x [plain, antithetic] x N_GRID x R, then the Asian cases."""
    # log_result appends, so start from a clean file or re-runs duplicate rows
    (_RESULTS_DIR / f"{PERSON}_{EXPERIMENT_ID}.csv").unlink(missing_ok=True)
    run_sweep(
        person=PERSON, experiment_id=EXPERIMENT_ID,
        cases=[("paper_barrier", "call")], methods=METHODS,
        n_grid=N_GRID, R=R, n_steps=N_STEPS, schemes=SCHEMES,
    )
    run_sweep(
        person=PERSON, experiment_id=EXPERIMENT_ID,
        cases=CASES, methods=METHODS,
        n_grid=N_GRID, R=R, n_steps=N_STEPS, schemes=["exact"],
    )


def _load():
    df = load_all_results()
    df = df[(df["person"] == PERSON) & (df["experiment_id"] == EXPERIMENT_ID)]
    if df.empty:
        raise RuntimeError("No baseline results found — run simulate() first.")
    return df


def _cell(summary, scenario, option, scheme):
    return summary[(summary["scenario"] == scenario) & (summary["option_type"] == option)
                   & (summary["scheme"] == scheme)]


def make_figures(df, barrier_ref=None):
    apply_style()
    summary = summarize(df)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES_DIR / "baseline_summary.csv", index=False)

    # --- baseline_barrier_ci.png: exact scheme, plain vs antithetic ---
    fig, ax = plt.subplots(figsize=(6, 4.2))
    sub = _cell(summary, "paper_barrier", "call", "exact")
    exact = exact_price(SCENARIOS["paper_barrier"], "call")
    ci_vs_n_plot(ax, sub, exact, METHODS, exact_label="Continuous closed form")
    if barrier_ref is not None:
        ax.axhline(barrier_ref, color="grey", linestyle=":", linewidth=1.2,
                   label=f"252-step discrete reference ({barrier_ref:.3f})")
        ax.legend(fontsize=8)
    ax.set_title("Up-and-in barrier call — CI vs N (exact-scheme MC)")
    save_fig(fig, "baseline_barrier_ci.png")
    plt.close(fig)

    # --- baseline_asian_call_ci.png / put ---
    for option, name in [("call", "baseline_asian_call_ci.png"),
                         ("put", "baseline_asian_put_ci.png")]:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        sub = _cell(summary, "paper_asian_geo", option, "exact")
        exact = exact_price(SCENARIOS["paper_asian_geo"], option)
        ci_vs_n_plot(ax, sub, exact, METHODS)
        ax.set_title(f"Geometric Asian {option} — CI vs N")
        save_fig(fig, name)
        plt.close(fig)

    # --- baseline_rmse_vs_n.png: RMSE-vs-N for barrier + both Asian cases ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, (scenario, option, title) in zip(axes, [
        ("paper_barrier", "call", "Barrier"),
        ("paper_asian_geo", "call", "Asian call"),
        ("paper_asian_geo", "put", "Asian put"),
    ]):
        sub = _cell(summary, scenario, option, "exact")
        loglog_rmse_plot(ax, sub, METHODS)
        ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, "baseline_rmse_vs_n.png")
    plt.close(fig)

    # --- baseline_scheme_comparison.png: exact vs euler_maruyama vs milstein (plain MC) ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    exact = exact_price(SCENARIOS["paper_barrier"], "call")
    for scheme in SCHEMES:
        sub = summary[(summary["scenario"] == "paper_barrier") & (summary["scheme"] == scheme)
                      & (summary["method"] == "plain")].sort_values("n_paths")
        ax.errorbar(sub["n_paths"], sub["mean_price"],
                    yerr=1.96 * sub["mean_std_error"], fmt="o-", capsize=3,
                    markersize=4, label=scheme)
    ax.axhline(exact, color="black", linestyle="--", linewidth=1, label="Continuous closed form")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Price")
    ax.set_title("Scheme comparison (plain MC, barrier, common random numbers)")
    ax.legend(fontsize=8)
    save_fig(fig, "baseline_scheme_comparison.png")
    plt.close(fig)

    # --- coverage_table.png ---
    make_coverage_table(summary)
    return summary


def make_coverage_table(summary):
    """95% CI coverage per scenario/option/method, pooled over N, rendered as a table."""
    rows = []
    for (scenario, option, method), g in summary[summary["scheme"] == "exact"].groupby(
        ["scenario", "option_type", "method"]
    ):
        n_covered = int(round((g["n_reps"] * g["ci_coverage"]).sum()))
        n_total = int(g["n_reps"].sum())
        cov = n_covered / n_total if n_total else float("nan")
        rows.append([scenario, option, method, f"{cov:.1%}", n_total])

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1))
    ax.axis("off")
    table = ax.table(cellText=rows,
                     colLabels=["Scenario", "Option", "Method", "CI coverage", "n obs"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("95% CI coverage vs closed form (target ~95%)", pad=20)
    save_fig(fig, "coverage_table.png")
    plt.close(fig)

    import csv
    with open(TABLES_DIR / "coverage_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "option", "method", "coverage", "n_obs"])
        w.writerows(rows)


# ---------------------------------------------------------------------------
# High-precision reference computations (independent of the N_GRID sweep)
# ---------------------------------------------------------------------------

def discrete_barrier_reference(n_paths=4_000_000, block=20_000):
    """Precise 252-step discretely-monitored up-and-in price via in-out parity:
    UIC = BS_call - E[UOC_discrete]. The UOC payoff is small and low-variance, so
    this is ~30x more precise than pricing the UIC directly."""
    scen = SCENARIOS["paper_barrier"]
    rng = np.random.default_rng(make_seed_seq("baseline_barrier_ref"))
    disc = math.exp(-scen.r * scen.T)
    ys = []
    for _ in range(n_paths // block):
        paths = generate_paths(scen, block, N_STEPS, rng.standard_normal((block, N_STEPS)))
        ys.append(disc * barrier_payoff(paths, scen.K, scen.B, "up_and_out", "call"))
    y = np.concatenate(ys)
    vanilla = bs_call(scen.S0, scen.K, scen.r, scen.sigma, scen.T)
    return vanilla - y.mean(), y.std(ddof=1) / math.sqrt(len(y))


def scheme_differences_crn(n_paths=1_000_000, block=20_000):
    """E[payoff_scheme - payoff_exact] on the barrier with COMMON random numbers,
    so the scheme effect is not buried under independent MC noise."""
    scen = SCENARIOS["paper_barrier"]
    rng = np.random.default_rng(make_seed_seq("baseline_scheme_crn"))
    payoff = payoff_for(scen)
    disc = math.exp(-scen.r * scen.T)
    diffs = {s: [] for s in SCHEMES if s != "exact"}
    for _ in range(n_paths // block):
        Z = rng.standard_normal((block, N_STEPS))
        y_exact = payoff(generate_paths(scen, block, N_STEPS, Z, "exact"))
        for s in diffs:
            diffs[s].append(disc * (payoff(generate_paths(scen, block, N_STEPS, Z, s)) - y_exact))
    out = {}
    for s, d in diffs.items():
        d = np.concatenate(d)
        out[s] = (d.mean(), d.std(ddof=1) / math.sqrt(len(d)))
    return out


# ---------------------------------------------------------------------------
# Paper-claim verification (Stage 2, task 8)
# ---------------------------------------------------------------------------

def _rows(df, scenario, option, method, n_paths=None, scheme="exact"):
    m = ((df["scenario"] == scenario) & (df["option_type"] == option)
         & (df["method"] == method) & (df["scheme"] == scheme))
    if n_paths is not None:
        m &= df["n_paths"] == n_paths
    return df[m]


def antithetic_table(df):
    """Antithetic vs plain, compared at EQUAL total path count, per N.

    VRF uses each replicate's own within-run SE (each estimated from N samples),
    which is far more precise than the across-replicate variance of R=20 prices;
    the across-replicate efficiency ratio (with bootstrap CI) is reported at the
    largest N as a cross-check.
    """
    out = []
    n_max = max(N_GRID)
    for scenario, option in [("paper_barrier", "call"), ("paper_asian_geo", "call"),
                             ("paper_asian_geo", "put"), ("asian_30d", "call")]:
        vrfs, time_ratios = [], []
        for n in N_GRID:
            p = _rows(df, scenario, option, "plain", n)
            a = _rows(df, scenario, option, "antithetic", n)
            vrfs.append((p["std_error"] ** 2).mean() / (a["std_error"] ** 2).mean())
            time_ratios.append(p["runtime_sec"].mean() / a["runtime_sec"].mean())
        a_max = _rows(df, scenario, option, "antithetic", n_max)
        rho = np.mean([json.loads(x)["rho_pair"] for x in a_max["extra_json"]])
        keys = dict(scenario=scenario, option_type=option, scheme="exact", n_paths=n_max)
        eff, lo, hi = bootstrap_efficiency_ratio(df, {**keys, "method": "antithetic"},
                                                 {**keys, "method": "plain"})
        vrf_max, t_max = vrfs[-1], time_ratios[-1]
        out.append(dict(scenario=scenario, option=option, rho_pair=rho,
                        vrf_by_n=vrfs, vrf=vrf_max, time_ratio=t_max,
                        eff_ratio=vrf_max * t_max, eff_boot=(eff, lo, hi)))
    return out


def stabilization_table(summary):
    """What does 'stabilized' at the paper's N actually mean in accuracy terms?"""
    out = []
    n_max = max(N_GRID)
    for scenario, option, paper_ns in [("paper_barrier", "call", [5000]),
                                       ("paper_asian_geo", "call", [750, 1000])]:
        sub = _cell(summary, scenario, option, "exact")
        plain = sub[sub["method"] == "plain"].sort_values("n_paths")
        sigma = float(plain[plain["n_paths"] == n_max]["mean_std_error"].iloc[0]) * math.sqrt(n_max)
        price = float(plain["exact_price"].iloc[0])
        slope, _ = fit_loglog_slope(plain["n_paths"], plain["rmse"])
        half = {n: 1.96 * sigma / math.sqrt(n) for n in paper_ns}
        n_1pct = (1.96 * sigma / (0.01 * price)) ** 2
        rmse_sqrt_n = (plain["rmse"] * np.sqrt(plain["n_paths"])).tolist()
        out.append(dict(scenario=scenario, option=option, sigma=sigma, price=price,
                        slope=slope, half=half, n_1pct=n_1pct, rmse_sqrt_n=rmse_sqrt_n))
    return out


def verify_paper_claims(df, summary, barrier_ref):
    L = ["# Baseline: verification of the base paper's claims\n",
         "Generated by `experiments/exp_baseline.py` from `results/raw/nafis_baseline.csv` "
         "plus two high-precision reference runs (in-out parity barrier price, "
         "common-random-number scheme differences).\n"]

    # ---- Claim 1: stabilization ----
    L.append("## Claim 1 — convergence 'stabilizes' at ~5000 (barrier) / 750–1000 (Asian)\n")
    L.append("| Option | Per-path SD | RMSE·√N by N (256→65536) | Fitted RMSE slope "
             "| 95% CI half-width at paper's N | N for ±1% |")
    L.append("|---|---|---|---|---|---|")
    for s in stabilization_table(summary):
        hw = "; ".join(f"N={n}: ±{h:.3f} (±{100 * h / s['price']:.1f}%)" for n, h in s["half"].items())
        L.append(f"| {s['scenario']}/{s['option']} | {s['sigma']:.2f} | "
                 f"{', '.join(f'{v:.1f}' for v in s['rmse_sqrt_n'])} | {s['slope']:.2f} | "
                 f"{hw} | ≈{s['n_1pct']:,.0f} |")
    L.append("\nVerdict: **not supported as a convergence property.** Error decays as N^-1/2 "
             "throughout the grid (RMSE·√N stays roughly constant — no plateau) with no threshold; at the paper's 'stabilized' N the 95% CI is "
             "still several percent of the price wide. The claim describes how the paper's "
             "log-x CI plots look, not a change in convergence behaviour.\n")

    # ---- Claim 2: antithetic ----
    L.append("## Claim 2 — antithetic 'accelerates' convergence 1.5x (barrier), 1.3x (Asian)\n")
    L.append("The paper does not define the factor. Below, all ratios are antithetic vs plain "
             "at the SAME total number of simulated paths (antithetic: N/2 pairs).\n")
    L.append("| Option | rho(Y, Y') | VRF by N (256→65536) | VRF @65536 | time ratio plain/anti "
             "| efficiency ratio | bootstrap eff. ratio @65536 [95% CI] |")
    L.append("|---|---|---|---|---|---|---|")
    for a in antithetic_table(df):
        e, lo, hi = a["eff_boot"]
        L.append(f"| {a['scenario']}/{a['option']} | {a['rho_pair']:+.2f} | "
                 f"{', '.join(f'{v:.2f}' for v in a['vrf_by_n'])} | {a['vrf']:.2f} | "
                 f"{a['time_ratio']:.2f} | {a['eff_ratio']:.2f} | {e:.2f} [{lo:.2f}, {hi:.2f}] |")
    L.append("\nVerdict: read as a variance ratio at equal path count, the paper's numbers are "
             "**confirmed** for the barrier (≈1.5) and the geometric Asian call (≈1.3–1.4). "
             "The Asian put's VRF is ≈3.9, well above 1.3 — but the paper's put figure (Fig. 8) "
             "appears to be a copy of its call figure (Fig. 6), so its put result was likely "
             "never separately measured. In our implementation antithetic is also ~1.2x "
             "*cheaper* per path (half the normal draws), so the efficiency gain exceeds the "
             "VRF; the paper instead reports 'increased computational time'. The across-"
             "replicate bootstrap ratio (last column) is the §1.7.3 cross-check: with R=20 a "
             "variance ratio has a ~[0.4, 2.5]x sampling range, so it cannot resolve 1.3 vs "
             "1.5; the within-run VRF (from N/2 iid pairs per replicate) can.\n")

    # ---- Claim 3: discrete-monitoring bias ----
    scen = SCENARIOS["paper_barrier"]
    cont = exact_price(scen, "call")
    bgk = exact_price_bgk(scen, "call", N_STEPS)
    ref, ref_se = barrier_ref
    p_max = _rows(df, "paper_barrier", "call", "plain", max(N_GRID))
    se_1024 = _rows(df, "paper_barrier", "call", "plain", 1024)["std_error"].mean()
    L.append("## Discrete-monitoring barrier bias (WORKPLAN trap #5)\n")
    L.append(f"- Continuous Reiner–Rubinstein value: {cont:.4f}")
    L.append(f"- BGK-corrected (252 steps): {bgk:.4f}")
    L.append(f"- **Precise 252-step discrete value (in-out parity, 4M paths): {ref:.4f} ± {ref_se:.4f}** "
             f"→ discretization bias = {ref - cont:+.4f}; BGK error = {bgk - ref:+.4f}")
    L.append(f"- Sweep, plain MC @65536 (mean of 20 reps): {p_max['price'].mean():.4f} ± "
             f"{p_max['price'].std(ddof=1) / math.sqrt(len(p_max)):.4f}")
    L.append(f"- Plain-MC SE at N=1024: {se_1024:.3f} — ~{se_1024 / abs(ref - cont):.0f}x the bias.")
    L.append("\nNote: WORKPLAN §2 lists the discrete value as ≈7.076 (bias ≈ −0.03); the precise "
             "value above shows the bias is ≈ −0.011 and BGK is accurate to ~0.001.\n")

    # ---- Claim 4: schemes ----
    crn = scheme_differences_crn()
    L.append("## Claim 4 — scheme choice (Euler / Euler–Maruyama / Milstein) is negligible\n")
    for s, (d, se) in crn.items():
        L.append(f"- {s} − exact (common random numbers, 1M paths): {d:+.5f} ± {se:.5f}")
    L.append(f"- For scale: plain-MC SE at N=65536 is {p_max['std_error'].mean():.3f}.")
    L.append("\nVerdict: **confirmed, and expected.** The scheme effect at dt=1/252 is O(dt) "
             "(~0.001, ~0.015% of the price) — statistically detectable only with common random "
             "numbers and ~50x smaller than the standard error of even a 65,536-path run. Both "
             "Euler–Maruyama and Milstein have weak order 1; Milstein improves only the strong "
             "(pathwise) order, so no accuracy gain on a price is expected. The paper never "
             "writes its schemes down, and its Figs. 2–4 show identical sample points, i.e. the "
             "same random numbers were reused, which makes near-identical plots unavoidable.\n")

    # ---- 30-day Asian ----
    s30 = SCENARIOS["asian_30d"]
    a30 = _rows(df, "asian_30d", "call", "antithetic", max(N_GRID))
    L.append("## 30-day-averaging geometric Asian call\n")
    L.append(f"- Discrete closed form, 30-point window: {exact_price(s30, 'call'):.4f}; "
             f"full 252-point window: {exact_price(SCENARIOS['paper_asian_geo'], 'call'):.4f}; "
             f"vanilla call: {bs_call(s30.S0, s30.K, s30.r, s30.sigma, s30.T):.4f}")
    L.append(f"- MC (antithetic @65536, mean of 20 reps): {a30['price'].mean():.4f} ± "
             f"{a30['price'].std(ddof=1) / math.sqrt(len(a30)):.4f}")
    L.append("\nThe paper states that for the 30-day window 'the exact closed-form solution "
             "remains identical to the previous examples'. Read literally (same value), this is "
             "false: shortening the window to 30 days more than doubles the price.\n")

    # ---- 3-SE acceptance check ----
    L.append("## Acceptance check — fraction of replicates within 3 SE of the closed form\n")
    ok = df[df["scheme"] == "exact"].copy()
    ok["within3"] = (ok["price"] - ok["exact_price"]).abs() <= 3 * ok["std_error"]
    for (sc, op, me), g in ok.groupby(["scenario", "option_type", "method"]):
        L.append(f"- {sc}/{op}/{me}: {g['within3'].mean():.1%} of {len(g)}")

    text = "\n".join(L) + "\n"
    (TABLES_DIR / "baseline_paper_claims.md").write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip simulation; only regenerate figures/tables from existing CSV.")
    args = parser.parse_args()

    if not args.analyze_only:
        simulate()

    df = _load()
    barrier_ref = discrete_barrier_reference()
    summary = make_figures(df, barrier_ref=barrier_ref[0])
    verify_paper_claims(df, summary, barrier_ref)
    print("Done. Figures in results/figures/, tables in results/tables/.")


if __name__ == "__main__":
    main()
