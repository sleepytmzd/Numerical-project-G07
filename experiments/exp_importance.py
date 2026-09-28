"""
Usage
-----
    OMP_NUM_THREADS=1 python experiments/exp_importance.py
    OMP_NUM_THREADS=1 python experiments/exp_importance.py --analyze-only
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _stage5_common import (TABLES_DIR, acceptance, cell_table, fmt_ci, references,
                            with_references)
from src.benchmark import exact_price_bgk, run_sweep
from src.config import N_GRID, N_STEPS, R, SCENARIOS
from src.estimators import make_seed_seq, seed_int
from src.paths import generate_paths
from src.payoffs import payoff_for
from src.plots import METHOD_COLORS, METHOD_LABELS, apply_style, ci_vs_n_plot, save_fig
from src.results import _RESULTS_DIR, load_all_results, log_result
from src.vr_importance import (effective_sample_size, heuristic_theta, importance_mc,
                               weighted_samples)

PERSON = "tamzeed"
EXPERIMENT_ID = "importance"

# Runtime-only extension scenario (config.py's frozen scenarios are untouched).
SCENARIOS.setdefault("rare_barrier",
                     replace(SCENARIOS["deep_barrier"], name="rare_barrier", B=160.0))

BARRIERS = ["deep_barrier", "paper_barrier", "rare_barrier"]
METHODS = ["plain", "antithetic", "importance"]
SCAN_METHOD = "importance_fixed_theta"
SCAN_FACTORS = np.linspace(0.0, 3.0, 25)
SCAN_N = 65_536
DIAG_N = 65_536

TITLES = {
    "paper_barrier": "Paper barrier (B=110.68)",
    "deep_barrier": "Deep barrier (B=140)",
    "rare_barrier": "Rare barrier (B=160, extension)",
}
SCEN_COLORS = {"deep_barrier": "#8172B2", "paper_barrier": "#4C72B0", "rare_barrier": "#C44E52"}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate():
    # log_result appends, so start from a clean file or re-runs duplicate rows
    (_RESULTS_DIR / f"{PERSON}_{EXPERIMENT_ID}.csv").unlink(missing_ok=True)
    run_sweep(person=PERSON, experiment_id=EXPERIMENT_ID,
              cases=[(b, "call") for b in BARRIERS], methods=METHODS,
              n_grid=N_GRID, R=R, n_steps=N_STEPS)

    for name in BARRIERS:
        scenario = SCENARIOS[name]
        theta0 = heuristic_theta(scenario)
        print(f"theta scan: {name}, {len(SCAN_FACTORS)} thetas x {SCAN_N}")
        for i, f in enumerate(SCAN_FACTORS):
            # same seed for every theta: common random numbers -> a smooth curve
            ss = make_seed_seq("importance_theta_scan", zlib.crc32(name.encode()))
            res = importance_mc(scenario, SCAN_N, N_STEPS, ss, option="call",
                                theta=float(theta0 * f))
            log_result(person=PERSON, experiment_id=EXPERIMENT_ID, scenario=name,
                       option_type="call", scheme="exact", method=SCAN_METHOD,
                       method_params=str({"theta_factor": float(f)}), n_paths=SCAN_N,
                       n_steps=N_STEPS, replicate_id=i, seed=seed_int(ss),
                       price=res.price, std_error=res.std_error, ci_low=res.ci_low,
                       ci_high=res.ci_high, exact_price=None, runtime_sec=res.runtime_sec,
                       extra={k: v for k, v in res.extra.items() if k != "theta_grid"})


def load_results():
    df = load_all_results()
    df = df[(df["person"] == PERSON) & (df["experiment_id"] == EXPERIMENT_ID)].copy()
    if df.empty:
        raise RuntimeError("No Stage-5 importance results — run the simulation first.")
    df["extra"] = df["extra_json"].fillna("{}").map(json.loads)
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def is_diagnostics(df):
    rows = []
    g_is = df[df["method"] == "importance"]
    for (scenario, n), g in g_is.groupby(["scenario", "n_paths"]):
        ex = pd.DataFrame(list(g["extra"]))
        rows.append(dict(scenario=scenario, n_paths=n, theta0=ex["theta0"].iloc[0],
                         theta_median=ex["theta"].median(),
                         theta_factor_values=sorted(set((ex["theta"] / ex["theta0"]).round(2))),
                         ess_frac_mean=ex["ESS_frac"].mean(), ess_frac_min=ex["ESS_frac"].min(),
                         max_weight_max=ex["max_weight"].max(),
                         max_weight_share_max=ex["max_weight_share"].max(),
                         hit_rate_proposal=ex["hit_rate_proposal"].mean()))
    return pd.DataFrame(rows)


def theta_scan(df):
    scan = df[df["method"] == SCAN_METHOD].copy()
    ex = pd.DataFrame(list(scan["extra"]), index=scan.index)
    scan["theta"] = ex["theta"]
    scan["theta0"] = ex["theta0"]
    scan["factor"] = scan["theta"] / scan["theta0"]
    scan["per_path_var"] = scan["std_error"] ** 2 * scan["n_paths"]
    scan["ess_frac"] = ex["ESS_frac"]
    return scan.sort_values(["scenario", "factor"])


def weight_samples(scenario_name, theta):
    sc = SCENARIOS[scenario_name]
    payoff = payoff_for(sc, "call")
    rng = np.random.default_rng(make_seed_seq("importance_diag", zlib.crc32(scenario_name.encode())))
    Y, w, _ = weighted_samples(sc, DIAG_N, N_STEPS, rng, theta, payoff)

    rng_p = np.random.default_rng(make_seed_seq("importance_diag_p", zlib.crc32(scenario_name.encode())))
    hit_p = pay_p = 0
    for k in (8192,) * (DIAG_N // 8192):
        Z = rng_p.standard_normal((k, N_STEPS))
        paths = generate_paths(sc, k, N_STEPS, Z)
        hit_p += int((paths.max(axis=1) >= sc.B).sum())
        pay_p += int((payoff(paths) > 0).sum())
    return dict(Y=Y, w=w, hit_rate_P=hit_p / DIAG_N, payoff_rate_P=pay_p / DIAG_N,
                payoff_rate_proposal=float(np.mean(Y > 0)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(table, scan, diag, weights, refs):
    apply_style()

    # --- is_deep_barrier.png ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    sub = table[table["scenario"] == "deep_barrier"]
    ci_vs_n_plot(a1, sub, refs["deep_barrier"][0], METHODS,
                 exact_label="252-date reference")
    a1.set_title("Deep barrier (B=140): estimate ± 95% CI vs N")
    for scen in BARRIERS:
        s = table[(table["scenario"] == scen) & (table["method"] == "importance")].sort_values("n_paths")
        x = s["n_paths"].to_numpy() * (1 + 0.04 * (BARRIERS.index(scen) - 1))
        y = s["eff"].to_numpy()
        a2.errorbar(x, y, yerr=[y - s["eff_ci_low"], s["eff_ci_high"] - y], fmt="o-",
                    capsize=3, markersize=4, color=SCEN_COLORS[scen], label=TITLES[scen])
        a2.plot(s["n_paths"], s["within_run_eff"], ":", color=SCEN_COLORS[scen], linewidth=1)
    a2.axhline(1.0, color="black", linestyle="--", linewidth=1)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("Number of paths (N)")
    a2.set_ylabel("Efficiency ratio vs plain MC")
    a2.set_title("IS efficiency vs plain (bootstrap 95% CI; dotted = within-run)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "is_deep_barrier.png")
    plt.close(fig)

    # --- is_weight_distribution.png ---
    fig, axes = plt.subplots(1, len(BARRIERS), figsize=(15, 4.2))
    for ax, scen in zip(axes, BARRIERS):
        d = weights[scen]
        w, Y = d["w"], d["Y"]
        bins = np.logspace(np.log10(w.min()), np.log10(w.max()), 60)
        ax.hist(w[Y > 0], bins=bins, color=SCEN_COLORS[scen], alpha=0.85, label="payoff > 0")
        ax.hist(w[Y == 0], bins=bins, color="grey", alpha=0.45, label="payoff = 0")
        ax.set_xscale("log")
        ess = effective_sample_size(w) / len(w)
        contrib = w[Y > 0] * Y[Y > 0]
        top1 = np.sort(contrib)[::-1][: max(1, len(contrib) // 100)].sum() / contrib.sum()
        ax.set_title(f"{TITLES[scen]}\ntheta={d['theta']:.2f} (theta0={d['theta0']:.2f})", fontsize=9)
        ax.text(0.02, 0.97, f"ESS = {ess:.1%} of N\nmax w = {w.max():.1f}\n"
                            f"top 1% of paths = {top1:.0%} of price",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("Likelihood ratio w (log scale)")
        ax.set_ylabel("Paths")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Likelihood-ratio distribution at the tuned drift (N={DIAG_N:,})", y=1.02)
    fig.tight_layout()
    save_fig(fig, "is_weight_distribution.png")
    plt.close(fig)

    # --- is_theta_scan.png (extra) ---
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for scen in BARRIERS:
        s = scan[scan["scenario"] == scen]
        base = s[s["factor"] == 0]["per_path_var"].iloc[0]
        ax.plot(s["factor"], s["per_path_var"] / base, "o-", markersize=3,
                color=SCEN_COLORS[scen], label=TITLES[scen])
        chosen = diag[(diag["scenario"] == scen) & (diag["n_paths"] == max(N_GRID))]
        if len(chosen):
            ax.axvline(chosen["theta_median"].iloc[0] / chosen["theta0"].iloc[0],
                       color=SCEN_COLORS[scen], linestyle=":", linewidth=1)
    ax.axvspan(0, 1.5, color="grey", alpha=0.08, label="7-point pilot grid range")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, label=r"heuristic $\theta_0$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta / \theta_0$")
    ax.set_ylabel("Per-path variance / plain MC")
    ax.set_title("Variance vs drift shift (fixed theta, common random numbers)")
    ax.legend(fontsize=8)
    save_fig(fig, "is_theta_scan.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def write_findings(df, table, scan, diag, weights, acc, refs):
    n_max = max(N_GRID)
    L = ["# Stage 5: importance sampling findings", "",
         f"Generated by `experiments/exp_importance.py` from `results/raw/{PERSON}_{EXPERIMENT_ID}.csv` "
         f"(R={R} replicates per cell, {N_STEPS} steps, exact scheme, machine "
         f"`{df['machine_id'].iloc[0]}`). Timings here are indicative; the master sweep "
         "(`master_findings.md`) is authoritative.", ""]

    L += ["## References (252-date discrete monitoring)", "",
          "| Scenario | Reference | SE | Source | BGK |", "|---|---:|---:|---|---:|"]
    for scen in BARRIERS:
        bgk = exact_price_bgk(SCENARIOS[scen], "call", N_STEPS)
        if scen in refs:
            ref, se, src = refs[scen]
            L.append(f"| {scen} | {ref:.6f} | {se:.6f} | {src} | {bgk:.6f} |")
        else:
            L.append(f"| {scen} | — | — | none (extension; variance metrics only) | {bgk:.6f} |")
    L.append("")

    L += ["## Drift selection", "",
          "theta0 = [log(B/S0)/T − (r − σ²/2)] / σ (WORKPLAN's formula without the /σ is a "
          "log-drift, not the Brownian shift). The 7-point grid theta0·{0, 0.25, …, 1.5} is "
          "scored on one independent 10,000-path pilot per estimate.", "",
          "| Scenario | theta0 | tuned theta (median, N=65,536) | theta/theta0 values chosen across all N×R | "
          "scan optimum theta/theta0 | scan variance at optimum / at tuned |",
          "|---|---:|---:|---|---:|---:|"]
    for scen in BARRIERS:
        d = diag[(diag["scenario"] == scen) & (diag["n_paths"] == n_max)].iloc[0]
        allf = sorted(set(np.concatenate(diag[diag["scenario"] == scen]["theta_factor_values"].to_list())))
        s = scan[scan["scenario"] == scen]
        opt = s.loc[s["per_path_var"].idxmin()]
        tuned_f = d["theta_median"] / d["theta0"]
        tuned_var = np.interp(tuned_f, s["factor"], s["per_path_var"])
        L.append(f"| {scen} | {d['theta0']:.3f} | {d['theta_median']:.3f} | "
                 f"{', '.join(f'{v:g}' for v in allf)} | {opt['factor']:.3g} | "
                 f"{opt['per_path_var'] / tuned_var:.2f} |")
    L.append("")

    L += ["## Knock-in rates and weights (diagnostic draw, N=65,536, tuned theta)", "",
          "| Scenario | P(hit B) under P | P(payoff>0) under P | P(payoff>0) under proposal "
          "| ESS / N | max w | max w / Σw |", "|---|---:|---:|---:|---:|---:|---:|"]
    for scen in BARRIERS:
        d = weights[scen]
        w = d["w"]
        L.append(f"| {scen} | {d['hit_rate_P']:.4f} | {d['payoff_rate_P']:.4f} | "
                 f"{d['payoff_rate_proposal']:.4f} | {effective_sample_size(w) / len(w):.3f} | "
                 f"{w.max():.1f} | {w.max() / w.sum():.2e} |")
    L.append("")

    for scen in BARRIERS:
        L += [f"## {TITLES[scen]}: per-N comparison", "",
              "| N | method | mean price | RMSE | VRF vs plain [95% F-CI] | within-run VRF | time ratio "
              "| efficiency vs plain [bootstrap 95% CI] | efficiency vs antithetic [95% CI] | ESS/N |",
              "|---:|---|---:|---:|---|---:|---:|---|---|---:|"]
        for n in N_GRID:
            for m in METHODS:
                r = table[(table["scenario"] == scen) & (table["method"] == m)
                          & (table["n_paths"] == n)].iloc[0]
                ess = "—"
                if m == "importance":
                    ess = f"{diag[(diag['scenario'] == scen) & (diag['n_paths'] == n)]['ess_frac_mean'].iloc[0]:.3f}"
                rmse = "—" if math.isnan(r["rmse"]) else f"{r['rmse']:.4f}"
                L.append(f"| {n:,} | {m} | {r['mean_price']:.4f} | {rmse} | "
                         f"{fmt_ci(r['vrf'], r['vrf_ci_low'], r['vrf_ci_high'], 2)} | "
                         f"{r['vrf_within_run']:.2f} | {r['time_ratio']:.2f} | "
                         f"{fmt_ci(r['eff'], r['eff_ci_low'], r['eff_ci_high'], 2)} | "
                         f"{fmt_ci(r['eff_vs_antithetic'], r['eff_vs_antithetic_ci_low'], r['eff_vs_antithetic_ci_high'], 2)} | "
                         f"{ess} |")
        L.append("")

    L += ["## Acceptance (§1.7): replicated mean vs reference, per cell", "",
          "| Scenario | Method | cells within 3 SE | t-interval covers | per-replicate CI coverage (mean) |",
          "|---|---|---|---|---:|"]
    for (scen, m), g in acc.groupby(["scenario", "method"]):
        L.append(f"| {scen} | {m} | {int(g['within3'].sum())}/{len(g)} | "
                 f"{int(g['t_covered'].sum())}/{len(g)} | {g['coverage'].mean():.3f} |")
    L.append("")

    text = "\n".join(L)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "importance_findings.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip simulation; rebuild figures/tables from the existing CSV.")
    args = ap.parse_args()
    if not args.analyze_only:
        simulate()

    df = load_results()
    refs = references()
    sweep = with_references(df[df["method"].isin(METHODS)], refs)
    table = cell_table(sweep, baseline="plain", compare_to=("antithetic",))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "importance_summary.csv", index=False)

    diag = is_diagnostics(df)
    scan = theta_scan(df)
    weights = {}
    for scen in BARRIERS:
        d = diag[(diag["scenario"] == scen) & (diag["n_paths"] == max(N_GRID))].iloc[0]
        weights[scen] = {**weight_samples(scen, float(d["theta_median"])),
                         "theta": float(d["theta_median"]), "theta0": float(d["theta0"])}
    acc = acceptance(sweep, refs)
    make_figures(table, scan, diag, weights, refs)
    write_findings(df, table, scan, diag, weights, acc, refs)
    print("Importance-sampling experiment complete.")


if __name__ == "__main__":
    main()
