"""
Stage 4: randomized QMC vs plain Monte Carlo (Zaki Rehnoom Unmona, 2105016).

Plain MC, RQMC with Brownian-bridge construction ("rqmc") and RQMC with
incremental construction ("rqmc_incremental") on the geometric Asian call, the
paper barrier (B=110.6772) and the deep barrier (B=140), across N_GRID x R=20.

Barrier RMSE is measured against the DISCRETELY monitored (252-date) price, not
the continuous Reiner–Rubinstein formula: the -0.011 monitoring bias would
otherwise put a floor under the RQMC RMSE and fake a flat slope. That reference
is estimated here from 64 extra independent bridge scrambles at N=65536 per
barrier (logged as method "rqmc_reference") and cross-checked against the
Stage-2 and Stage-3 values.

Usage
-----
    OMP_NUM_THREADS=1 python experiments/exp_qmc.py                 # simulate + analyse
    OMP_NUM_THREADS=1 python experiments/exp_qmc.py --analyze-only  # analyse only
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import math
import sys
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import bootstrap_efficiency_ratio, exact_price, run_sweep, summarize
from src.config import N_GRID, N_STEPS, R, SCENARIOS
from src.estimators import make_seed_seq, seed_int
from src.plots import (METHOD_COLORS, METHOD_LABELS, apply_style, loglog_rmse_plot,
                       save_fig)
from src.results import _RESULTS_DIR, load_all_results, log_result
from src.vr_qmc import bridge_matrix, rqmc_mc

PERSON = "zaki"
EXPERIMENT_ID = "qmc"
TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

CASES = [
    ("paper_asian_geo", "call"),
    ("paper_barrier", "call"),
    ("deep_barrier", "call"),
]
METHODS = ["plain", "rqmc", "rqmc_incremental"]
RQMC_METHODS = ["rqmc", "rqmc_incremental"]
BARRIERS = ["paper_barrier", "deep_barrier"]
REF_METHOD = "rqmc_reference"
REF_SCRAMBLES = 64
REF_N = 65_536

# Independent checks of the 252-date paper-barrier price from earlier stages.
STAGE2_PARITY_REF = (7.0941, 0.0002)     # plain MC, in-out parity, 4M paths
STAGE3_CV_REF = (7.093583, 0.00028)      # control variate, 20 x 65536 paths

TITLES = {
    "paper_asian_geo": "Geometric Asian call",
    "paper_barrier": "Paper barrier (B=110.68)",
    "deep_barrier": "Deep barrier (B=140)",
}

# Style for the extra construction (plots.py reserves one colour per method).
METHOD_COLORS.setdefault("rqmc_incremental", "#E8A33D")
METHOD_LABELS.setdefault("rqmc_incremental", "RQMC (Sobol, incremental)")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate():
    # log_result appends, so start from a clean file or re-runs duplicate rows
    (_RESULTS_DIR / f"{PERSON}_{EXPERIMENT_ID}.csv").unlink(missing_ok=True)
    run_sweep(person=PERSON, experiment_id=EXPERIMENT_ID, cases=CASES, methods=METHODS,
              n_grid=N_GRID, R=R, n_steps=N_STEPS, schemes=["exact"])

    for name in BARRIERS:
        scenario = SCENARIOS[name]
        print(f"reference: {name}, {REF_SCRAMBLES} scrambles x {REF_N}")
        for rep in range(REF_SCRAMBLES):
            ss = make_seed_seq("qmc_reference", zlib.crc32(name.encode()), rep)
            res = rqmc_mc(scenario, REF_N, N_STEPS, ss, option="call")
            log_result(person=PERSON, experiment_id=EXPERIMENT_ID, scenario=name,
                       option_type="call", scheme="exact", method=REF_METHOD,
                       method_params="{'construction': 'bridge'}", n_paths=REF_N,
                       n_steps=N_STEPS, replicate_id=rep, seed=seed_int(ss),
                       price=res.price, std_error=res.std_error, ci_low=res.ci_low,
                       ci_high=res.ci_high, exact_price=None, runtime_sec=res.runtime_sec,
                       extra={**res.extra, "purpose": "discrete_monitoring_reference"})


def load_results():
    df = load_all_results()
    df = df[(df["person"] == PERSON) & (df["experiment_id"] == EXPERIMENT_ID)].copy()
    if df.empty:
        raise RuntimeError("No Stage-4 QMC results found — run the simulation first.")
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def references(df):
    """{scenario: (reference price, its SE)} — the discrete 252-date target."""
    refs = {"paper_asian_geo": (exact_price(SCENARIOS["paper_asian_geo"], "call", N_STEPS), 0.0)}
    for name in BARRIERS:
        p = df[(df["method"] == REF_METHOD) & (df["scenario"] == name)]["price"]
        refs[name] = (float(p.mean()), float(p.std(ddof=1) / math.sqrt(len(p))))
    return refs


def _sweep(df, refs):
    sweep = df[df["method"].isin(METHODS)].copy()
    sweep["exact_price"] = sweep["scenario"].map(lambda s: refs[s][0])
    return sweep


def _rmse(prices, ref):
    return math.sqrt((prices.mean() - ref) ** 2 + prices.var(ddof=1))


def slope_with_ci(sweep, scenario, method, ref, n_boot=2000, seed=402):
    """RMSE-vs-N log-log slope, with a bootstrap CI from resampling replicates per N."""
    cells = [sweep[(sweep["scenario"] == scenario) & (sweep["method"] == method)
                   & (sweep["n_paths"] == n)]["price"].to_numpy() for n in N_GRID]
    x = np.log(N_GRID)
    point = np.polyfit(x, np.log([_rmse(c, ref) for c in cells]), 1)[0]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        y = [_rmse(c[rng.integers(0, len(c), len(c))], ref) for c in cells]
        boots[b] = np.polyfit(x, np.log(y), 1)[0]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def build_summary(df, refs):
    sweep = _sweep(df, refs)
    summary = summarize(sweep)
    # RQMC rows have no per-run CI by design; coverage there is not defined.
    summary.loc[summary["method"].isin(RQMC_METHODS), "ci_coverage"] = np.nan

    # The same RMSE against the CONTINUOUS formula, to expose the bias floor.
    cont = {name: exact_price(SCENARIOS[name], "call", N_STEPS) for name, _ in CASES}
    summary["continuous_price"] = summary["scenario"].map(cont)
    summary["rmse_vs_continuous"] = np.sqrt(
        (summary["mean_price"] - summary["continuous_price"]) ** 2 + summary["var_across_reps"])

    rows = []
    for _, row in summary.iterrows():
        plain = summary[(summary["scenario"] == row["scenario"]) & (summary["method"] == "plain")
                        & (summary["n_paths"] == row["n_paths"])].iloc[0]
        vrf = plain["var_across_reps"] / row["var_across_reps"]
        d1, d2 = int(plain["n_reps"]) - 1, int(row["n_reps"]) - 1
        keys = dict(scenario=row["scenario"], option_type="call", scheme="exact",
                    n_paths=row["n_paths"])
        if row["method"] == "plain":
            eff = (1.0, 1.0, 1.0)
        else:
            eff = bootstrap_efficiency_ratio(sweep, {**keys, "method": row["method"]},
                                             {**keys, "method": "plain"})
        rows.append({
            "vrf_vs_plain": vrf,
            # sample-variance ratio of normal estimates ~ F(d1, d2)
            "vrf_ci_low": vrf / f_dist.ppf(0.975, d1, d2),
            "vrf_ci_high": vrf / f_dist.ppf(0.025, d1, d2),
            "time_ratio_vs_plain": row["mean_runtime_sec"] / plain["mean_runtime_sec"],
            "efficiency_ratio_vs_plain": eff[0],
            "efficiency_ci_low": eff[1],
            "efficiency_ci_high": eff[2],
        })
    summary = pd.concat([summary.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    summary = summary.sort_values(["scenario", "method", "n_paths"]).reset_index(drop=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES_DIR / "qmc_summary.csv", index=False)
    return sweep, summary


def acceptance(sweep, refs):
    """§1.7.1 / §1.7.2 for RQMC: one t-interval per cell from its R scrambles."""
    out = []
    tcrit = t_dist.ppf(0.975, R - 1)
    for (scenario, method, n), g in sweep.groupby(["scenario", "method", "n_paths"]):
        ref = refs[scenario][0]
        p = g["price"]
        se = p.std(ddof=1) / math.sqrt(len(p))
        z = abs(p.mean() - ref) / se
        out.append(dict(scenario=scenario, method=method, n_paths=n, z=z,
                        covered=z <= tcrit, within3=z <= 3.0))
    return pd.DataFrame(out)


def variance_share(n_steps=N_STEPS):
    """Share of Var(log S_T) and Var(log G) carried by each Sobol dimension.

    Both are linear in the Brownian path, so the share of dimension k is exact:
    (c . A[:, k])^2 / sum, with A the map from input normals to W on the grid.
    """
    shares = {}
    for construction in ("incremental", "bridge"):
        M = np.eye(n_steps) if construction == "incremental" else bridge_matrix(n_steps)
        W = np.cumsum(M, axis=1)                   # W[k, j]: weight of z_k in W_{j+1}
        for label, c in [("log S_T", np.eye(n_steps)[-1]),
                         ("log G (252-date average)", np.full(n_steps, 1.0 / n_steps))]:
            w2 = (W @ c) ** 2
            shares[(construction, label)] = np.cumsum(w2) / w2.sum()
    return shares


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _cell(summary, scenario, methods):
    return summary[(summary["scenario"] == scenario) & (summary["method"].isin(methods))]


def _vrf_panel(ax, summary, scenario, methods):
    for i, method in enumerate(methods):
        sub = _cell(summary, scenario, [method]).sort_values("n_paths")
        x = sub["n_paths"].to_numpy() * (1.0 + 0.04 * (i - (len(methods) - 1) / 2))
        y = sub["vrf_vs_plain"].to_numpy()
        yerr = [y - sub["vrf_ci_low"].to_numpy(), sub["vrf_ci_high"].to_numpy() - y]
        ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, markersize=4,
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths (N)")
    ax.set_ylabel("Variance reduction vs plain MC (95% CI)")
    ax.legend(fontsize=8)


def make_figures(summary, refs):
    apply_style()

    # --- qmc_vs_mc_asian.png ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    loglog_rmse_plot(a1, _cell(summary, "paper_asian_geo", ["plain", "rqmc"]), ["plain", "rqmc"])
    a1.set_title("Geometric Asian call — RMSE vs N")
    _vrf_panel(a2, summary, "paper_asian_geo", ["rqmc"])
    a2.set_title("Variance reduction factor")
    fig.tight_layout()
    save_fig(fig, "qmc_vs_mc_asian.png")
    plt.close(fig)

    # --- qmc_vs_mc_barrier.png: RMSE vs the discrete reference; paper panel shows the floor ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, scenario in zip(axes, BARRIERS):
        sub = _cell(summary, scenario, ["plain", "rqmc"])
        loglog_rmse_plot(ax, sub, ["plain", "rqmc"])
        if scenario == "paper_barrier":
            rq = sub[sub["method"] == "rqmc"].sort_values("n_paths")
            ax.plot(rq["n_paths"], rq["rmse_vs_continuous"], "x:", color=METHOD_COLORS["rqmc"],
                    label="RQMC RMSE vs continuous formula (bias floor)")
            ax.axhline(abs(refs[scenario][0] - rq["continuous_price"].iloc[0]), color="grey",
                       linestyle="--", linewidth=0.8, label="|discrete − continuous| = 0.011")
            ax.legend(fontsize=7)
        ax.set_title(f"{TITLES[scenario]} — RMSE vs 252-date reference")
    fig.tight_layout()
    save_fig(fig, "qmc_vs_mc_barrier.png")
    plt.close(fig)

    # --- qmc_bridge_vs_incremental.png ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (scenario, _) in zip(axes, CASES):
        loglog_rmse_plot(ax, _cell(summary, scenario, METHODS), METHODS)
        ax.set_title(TITLES[scenario])
    fig.suptitle("Brownian bridge vs incremental path construction (RQMC)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "qmc_bridge_vs_incremental.png")
    plt.close(fig)

    # --- qmc_effective_dimension.png (extra): where the variance lives ---
    shares = variance_share()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    k = np.arange(1, N_STEPS + 1)
    styles = {"bridge": "-", "incremental": "--"}
    colors = {"log S_T": "#C44E52", "log G (252-date average)": "#4C72B0"}
    for (construction, label), cum in shares.items():
        ax.plot(k, cum, styles[construction], color=colors[label],
                label=f"{label}, {construction}")
    ax.set_xscale("log")
    ax.set_xlabel("First k Sobol dimensions")
    ax.set_ylabel("Cumulative share of variance")
    ax.set_title("Brownian bridge concentrates variance in the leading dimensions")
    ax.legend(fontsize=8)
    save_fig(fig, "qmc_effective_dimension.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def write_findings(df, sweep, summary, refs):
    n_max = max(N_GRID)
    L = ["# Stage 4: randomized QMC findings", "",
         f"Generated by `experiments/exp_qmc.py` from `results/raw/{PERSON}_{EXPERIMENT_ID}.csv` "
         f"(R={R} independent scrambles / replicates per cell, {N_STEPS} steps, exact scheme).", ""]

    L += ["## Reference prices (252-date discrete monitoring)", "",
          "| Scenario | Reference | SE | Continuous formula | Source |", "|---|---:|---:|---:|---|"]
    for name, _ in CASES:
        ref, se = refs[name]
        cont = exact_price(SCENARIOS[name], "call", N_STEPS)
        src = ("discrete Kemna–Vorst-style closed form" if name == "paper_asian_geo" else
               f"{REF_SCRAMBLES} bridge scrambles × {REF_N:,} paths")
        L.append(f"| {name} | {ref:.6f} | {se:.6f} | {cont:.6f} | {src} |")
    ref, se = refs["paper_barrier"]
    for label, (v, s) in [("Stage 2 in-out parity", STAGE2_PARITY_REF),
                          ("Stage 3 control variate", STAGE3_CV_REF)]:
        L.append(f"\nCross-check, paper barrier vs {label} {v:.6f} ± {s:.6f}: "
                 f"difference {ref - v:+.6f} = {(ref - v) / math.hypot(se, s):+.2f} combined SE.")
    L.append("")

    L += ["## RMSE-vs-N slopes (bootstrap 95% CI over replicates)", "",
          "| Scenario | Method | slope | 95% CI |", "|---|---|---:|---|"]
    slopes = {}
    for name, _ in CASES:
        for method in METHODS:
            s, lo, hi = slope_with_ci(sweep, name, method, refs[name][0])
            slopes[(name, method)] = (s, lo, hi)
            L.append(f"| {name} | {method} | {s:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    cont = summary[(summary["scenario"] == "paper_barrier") & (summary["method"] == "rqmc")]
    cont = cont.sort_values("n_paths")
    s_cont = np.polyfit(np.log(cont["n_paths"]), np.log(cont["rmse_vs_continuous"]), 1)[0]
    L.append(f"\nPaper barrier, rqmc, RMSE measured against the continuous formula instead: "
             f"slope {s_cont:+.2f} — the 0.011 monitoring bias floors the error.\n")

    L += [f"## At N={n_max:,}", "",
          "| Scenario | Method | mean price | SD across reps | RMSE | VRF vs plain [95% CI] "
          "| time ratio | efficiency ratio [bootstrap 95% CI] |",
          "|---|---|---:|---:|---:|---|---:|---|"]
    top = summary[summary["n_paths"] == n_max]
    for name, _ in CASES:
        for method in METHODS:
            r = top[(top["scenario"] == name) & (top["method"] == method)].iloc[0]
            L.append(f"| {name} | {method} | {r['mean_price']:.5f} | "
                     f"{math.sqrt(r['var_across_reps']):.5f} | {r['rmse']:.5f} | "
                     f"{r['vrf_vs_plain']:.1f} [{r['vrf_ci_low']:.1f}, {r['vrf_ci_high']:.1f}] | "
                     f"{r['time_ratio_vs_plain']:.2f} | {r['efficiency_ratio_vs_plain']:.1f} "
                     f"[{r['efficiency_ci_low']:.1f}, {r['efficiency_ci_high']:.1f}] |")
    L.append("")

    L += ["## Bridge vs incremental (variance ratio incremental / bridge, by N)", ""]
    for name, _ in CASES:
        rb = _cell(summary, name, ["rqmc"]).sort_values("n_paths")["var_across_reps"].to_numpy()
        ri = _cell(summary, name, ["rqmc_incremental"]).sort_values("n_paths")["var_across_reps"].to_numpy()
        L.append(f"- {name}: " + ", ".join(f"{v:.1f}" for v in ri / rb))
    L.append("")

    shares = variance_share()
    L += ["## Variance share of the first Sobol dimension(s) (exact, linear functionals)", "",
          "| Functional | construction | k=1 | k=4 | k=16 |", "|---|---|---:|---:|---:|"]
    for (construction, label), cum in shares.items():
        L.append(f"| {label} | {construction} | {cum[0]:.3f} | {cum[3]:.3f} | {cum[15]:.3f} |")
    L.append("")

    acc = acceptance(sweep, refs)
    L += ["## Acceptance (one t-interval per cell from its R scrambles/replicates)", "",
          "| Method | cells within 3 SE | 95% t-interval covers reference |", "|---|---|---|"]
    for method in METHODS:
        g = acc[acc["method"] == method]
        L.append(f"| {method} | {int(g['within3'].sum())}/{len(g)} | "
                 f"{int(g['covered'].sum())}/{len(g)} |")
    plain = summary[summary["method"] == "plain"]
    L.append(f"\nPlain-MC per-replicate 95% CI coverage (vs the discrete reference), by cell: "
             f"{plain['ci_coverage'].min():.0%}–{plain['ci_coverage'].max():.0%}.")
    L.append("\nTimings are indicative (Stage 5's master sweep is authoritative); machine: "
             f"{df['machine_id'].iloc[0]}.\n")

    text = "\n".join(L)
    (TABLES_DIR / "qmc_findings.md").write_text(text, encoding="utf-8")
    print(text)
    return slopes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip simulation; rebuild figures/tables from the existing CSV.")
    args = parser.parse_args()
    if not args.analyze_only:
        simulate()
    df = load_results()
    refs = references(df)
    sweep, summary = build_summary(df, refs)
    make_figures(summary, refs)
    write_findings(df, sweep, summary, refs)
    print("QMC experiment complete.")


if __name__ == "__main__":
    main()
