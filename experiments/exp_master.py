"""
Usage
-----
    OMP_NUM_THREADS=1 python experiments/exp_master.py                  # simulate + analyse
    OMP_NUM_THREADS=1 python experiments/exp_master.py --simulate-only
    OMP_NUM_THREADS=1 python experiments/exp_master.py --analyze-only
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.vr_control  
import src.vr_importance   
import src.vr_qmc          
from _stage5_common import (TABLES_DIR, acceptance, cell_table, fmt_ci, references,
                            run_arith_reference, with_references)
from src.benchmark import ESTIMATORS, exact_price, fit_loglog_slope
from src.config import BASE_SEED, N_GRID, N_STEPS, R, SCENARIOS
from src.estimators import make_seed_seq, seed_int
from src.plots import METHOD_COLORS, METHOD_LABELS, apply_style, loglog_rmse_plot, save_fig
from src.results import _RESULTS_DIR, load_all_results, log_result

PERSON = "tamzeed"
EXPERIMENT_ID = "master"

ALL_METHODS = ["plain", "antithetic", "control_variate", "rqmc", "importance"]
BARRIER_CASES = [("paper_barrier", "call"), ("deep_barrier", "call")]
ARITH_CASES = [("asian_arith", "call")]
ARITH_METHODS = ["plain", "antithetic", "control_variate", "rqmc"]
GEO_CASES = [("paper_asian_geo", "call")]
GEO_METHODS = ["plain", "antithetic", "rqmc"]
SWEEPS = [(BARRIER_CASES, ALL_METHODS), (ARITH_CASES, ARITH_METHODS), (GEO_CASES, GEO_METHODS)]
SCENARIO_ORDER = [c[0] for cases, _ in SWEEPS for c in cases]
TITLES = {
    "paper_barrier": "Paper barrier (B=110.68)",
    "deep_barrier": "Deep barrier (B=140)",
    "asian_arith": "Arithmetic Asian call",
    "paper_asian_geo": "Geometric Asian call",
}
# Target accuracies for the time-to-accuracy table (absolute RMSE, price units).
EPSILONS = (1e-2, 1e-3)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _cells():
    return [(scen, opt, m) for cases, methods in SWEEPS for scen, opt in cases for m in methods]


def _seed(scen, opt, method, n_paths, rep):
    # Exactly run_sweep's key, so every price equals what run_sweep would log.
    return make_seed_seq(EXPERIMENT_ID, zlib.crc32(scen.encode()), zlib.crc32(opt.encode()),
                         zlib.crc32(method.encode()), n_paths, rep)


def simulate():
    # log_result appends, so start from a clean file or re-runs duplicate rows
    (_RESULTS_DIR / f"{PERSON}_{EXPERIMENT_ID}.csv").unlink(missing_ok=True)
    cells = _cells()

    # Untimed warm-up: imports, allocator, RQMC's cached bridge matrix.
    for scen, opt, method in cells:
        ESTIMATORS[method](SCENARIOS[scen], 256, N_STEPS, _seed("warmup", opt, method, 256, 0),
                           option=opt)

    order_rng = np.random.default_rng(BASE_SEED)
    t0 = time.perf_counter()
    for rep in range(R):
        for n_paths in N_GRID:
            for i in order_rng.permutation(len(cells)):
                scen, opt, method = cells[i]
                scenario = SCENARIOS[scen]
                ss = _seed(scen, opt, method, n_paths, rep)
                res = ESTIMATORS[method](scenario, n_paths, N_STEPS, ss, option=opt, scheme="exact")
                log_result(person=PERSON, experiment_id=EXPERIMENT_ID, scenario=scen,
                           option_type=opt, scheme="exact", method=method, method_params="{}",
                           n_paths=n_paths, n_steps=N_STEPS, replicate_id=rep,
                           seed=seed_int(ss), price=res.price, std_error=res.std_error,
                           ci_low=res.ci_low, ci_high=res.ci_high,
                           exact_price=exact_price(scenario, opt, N_STEPS),
                           runtime_sec=res.runtime_sec, extra=res.extra)
        print(f"replicate {rep + 1}/{R} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"master sweep done in {time.perf_counter() - t0:.0f}s")
    run_arith_reference()


def load_results():
    df = load_all_results()
    df = df[(df["person"] == PERSON) & (df["experiment_id"] == EXPERIMENT_ID)].copy()
    if df.empty:
        raise RuntimeError("No master-sweep results — run the simulation first.")
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def slope_with_ci(sweep, scenario, method, ref, n_boot=2000, seed=402):
    cells = [sweep[(sweep["scenario"] == scenario) & (sweep["method"] == method)
                   & (sweep["n_paths"] == n)]["price"].to_numpy() for n in N_GRID]

    def rmse(c):
        return math.sqrt((c.mean() - ref) ** 2 + c.var(ddof=1))

    x = np.log(N_GRID)
    point = np.polyfit(x, np.log([rmse(c) for c in cells]), 1)[0]
    rng = np.random.default_rng(seed)
    boots = [np.polyfit(x, np.log([rmse(c[rng.integers(0, len(c), len(c))]) for c in cells]), 1)[0]
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def time_to_accuracy(table, scenario, method, eps):
    s = table[(table["scenario"] == scenario) & (table["method"] == method)].sort_values("n_paths")
    if s.empty or s["rmse"].isna().any():
        return (np.nan, np.nan, True)
    slope, intercept = fit_loglog_slope(s["n_paths"], s["rmse"])
    n_needed = math.exp((math.log(eps) - intercept) / slope)
    d, c = np.polyfit(s["n_paths"], s["mean_runtime_sec"], 1)
    seconds = c + d * n_needed
    extrapolated = not (N_GRID[0] <= n_needed <= N_GRID[-1])
    return (float(seconds), float(n_needed), extrapolated)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _methods_for(scenario):
    return next(methods for cases, methods in SWEEPS if scenario in dict(cases))


def make_figures(table):
    apply_style()
    n_max = max(N_GRID)

    # --- master_rmse_vs_n.png ---
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    for ax, scen in zip(axes.flat, SCENARIO_ORDER):
        loglog_rmse_plot(ax, table[table["scenario"] == scen], _methods_for(scen))
        ax.set_title(f"{TITLES[scen]} — RMSE vs 252-date reference")
    fig.tight_layout()
    save_fig(fig, "master_rmse_vs_n.png")
    plt.close(fig)

    # --- master_efficiency_bars.png ---
    top = table[table["n_paths"] == n_max]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.16
    for j, m in enumerate(ALL_METHODS):
        xs, ys, lo, hi = [], [], [], []
        for i, scen in enumerate(SCENARIO_ORDER):
            r = top[(top["scenario"] == scen) & (top["method"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            xs.append(i + (j - 2) * width)
            ys.append(r["eff"])
            lo.append(r["eff"] - r["eff_ci_low"])
            hi.append(r["eff_ci_high"] - r["eff"])
        ax.bar(xs, ys, width, yerr=[lo, hi], capsize=2, color=METHOD_COLORS[m],
               label=METHOD_LABELS[m], error_kw=dict(linewidth=0.8))
        for x, y, h in zip(xs, ys, hi):
            text = f"{y:,.0f}" if y >= 100 else f"{y:.2g}" if y < 10 else f"{y:.0f}"
            ax.text(x, (y + h) * 1.25, text, ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.3, ax.get_ylim()[1] * 8)
    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels([TITLES[s] for s in SCENARIO_ORDER])
    ax.set_ylabel("Efficiency vs plain MC  (1/(Var×time), log)")
    ax.set_title(f"Efficiency relative to plain MC at N={n_max:,} (bootstrap 95% CI, R={R})")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_fig(fig, "master_efficiency_bars.png")
    plt.close(fig)

    # --- master_efficiency_table.png ---
    cell_text, colours = [], []
    for m in ALL_METHODS:
        row_t, row_c = [], []
        for scen in SCENARIO_ORDER:
            r = top[(top["scenario"] == scen) & (top["method"] == m)]
            if r.empty:
                row_t.append("n/a")
                row_c.append("#f2f2f2")
                continue
            r = r.iloc[0]
            row_t.append(f"{r['eff']:.3g}\n[{r['eff_ci_low']:.3g}, {r['eff_ci_high']:.3g}]\n"
                         f"RMSE {r['rmse']:.2e}")
            level = min(1.0, max(0.0, math.log10(max(r["eff"], 1e-3)) / 4))
            row_c.append(plt.cm.Greens(0.15 + 0.6 * level))
        cell_text.append(row_t)
        colours.append(row_c)
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, cellColours=colours,
                   rowLabels=[METHOD_LABELS[m] for m in ALL_METHODS],
                   colLabels=[TITLES[s] for s in SCENARIO_ORDER], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 3.1)
    ax.set_title(f"Efficiency ratio vs plain MC at N={n_max:,} [bootstrap 95% CI], "
                 "with RMSE — one machine, single-threaded", fontsize=10)
    save_fig(fig, "master_efficiency_table.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def write_findings(df, sweep, table, acc, refs):
    n_max = max(N_GRID)
    machines = sorted(df["machine_id"].unique())
    L = ["# Stage 5: master comparison findings", "",
         f"Generated by `experiments/exp_master.py` from `results/raw/{PERSON}_{EXPERIMENT_ID}.csv` "
         f"({len(df)} rows, R={R} replicates per cell, {N_STEPS} steps, exact scheme, "
         f"`OMP_NUM_THREADS=1`). **Authoritative timings** — machine: `{', '.join(machines)}`.", "",
         "Efficiency = 1/(across-replicate variance × mean runtime per estimate); every "
         "ratio is against plain MC in the same (scenario, N) cell, with a 2,000-resample "
         "bootstrap 95% CI over replicates. Runtimes include pilot runs (control variate, "
         "importance sampling) and the Sobol scramble (RQMC).", ""]

    L += ["## Reference prices (252-date discrete monitoring)", "",
          "| Scenario | Reference | SE | Source |", "|---|---:|---:|---|"]
    for scen in SCENARIO_ORDER:
        ref, se, src = refs[scen]
        L.append(f"| {scen} | {ref:.6f} | {se:.6f} | {src} |")
    L.append("")

    top = table[table["n_paths"] == n_max]
    L += [f"## Headline: efficiency vs plain MC at N={n_max:,}", "",
          "| Scenario | Method | mean price | RMSE | VRF [95% F-CI] | within-run VRF | time ratio "
          "| efficiency [bootstrap 95% CI] |", "|---|---|---:|---:|---|---:|---:|---|"]
    for scen in SCENARIO_ORDER:
        for m in _methods_for(scen):
            r = top[(top["scenario"] == scen) & (top["method"] == m)].iloc[0]
            wr = "—" if math.isnan(r["vrf_within_run"]) else f"{r['vrf_within_run']:.1f}"
            L.append(f"| {scen} | {m} | {r['mean_price']:.5f} | {r['rmse']:.2e} | "
                     f"{fmt_ci(r['vrf'], r['vrf_ci_low'], r['vrf_ci_high'])} | {wr} | "
                     f"{r['time_ratio']:.2f} | {fmt_ci(r['eff'], r['eff_ci_low'], r['eff_ci_high'])} |")
    L.append("")

    L += ["## Winner per scenario and N (highest efficiency point estimate)", "",
          "| Scenario | " + " | ".join(f"N={n:,}" for n in N_GRID) + " |",
          "|---|" + "---|" * len(N_GRID)]
    for scen in SCENARIO_ORDER:
        cells = []
        for n in N_GRID:
            s = table[(table["scenario"] == scen) & (table["n_paths"] == n)]
            best = s.loc[s["eff"].idxmax()]
            second = s[s["method"] != best["method"]].loc[lambda x: x["eff"].idxmax()]
            separated = best["eff_ci_low"] > second["eff_ci_high"]
            cells.append(f"{best['method']} ({best['eff']:.3g}{'' if separated else ', CI overlaps ' + second['method']})")
        L.append(f"| {scen} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Efficiency vs plain MC at every N (bootstrap 95% CI)", "",
          "| Scenario | Method | " + " | ".join(f"N={n:,}" for n in N_GRID) + " |",
          "|---|---|" + "---|" * len(N_GRID)]
    for scen in SCENARIO_ORDER:
        for m in _methods_for(scen):
            s = table[(table["scenario"] == scen) & (table["method"] == m)].sort_values("n_paths")
            L.append(f"| {scen} | {m} | " + " | ".join(
                fmt_ci(r["eff"], r["eff_ci_low"], r["eff_ci_high"], 2) for _, r in s.iterrows()) + " |")
    L.append("")

    L += ["## RMSE-vs-N slopes (bootstrap 95% CI)", "",
          "| Scenario | " + " | ".join(ALL_METHODS) + " |", "|---|" + "---|" * len(ALL_METHODS)]
    slopes = {}
    for scen in SCENARIO_ORDER:
        cells = []
        for m in ALL_METHODS:
            if m not in _methods_for(scen):
                cells.append("n/a")
                continue
            s, lo, hi = slope_with_ci(sweep, scen, m, refs[scen][0])
            slopes[(scen, m)] = s
            cells.append(f"{s:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        L.append(f"| {scen} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Time for one estimate to reach RMSE ε", "",
          "From the fitted RMSE(N) = a·N^s and runtime(N) = c + d·N over N_GRID. "
          "`*` = extrapolated beyond N=65,536 (for RQMC, whose slope is steeper than −0.5, "
          "extrapolation is only indicative).", "",
          "| Scenario | ε | " + " | ".join(ALL_METHODS) + " | fastest |",
          "|---|---:|" + "---|" * (len(ALL_METHODS) + 1)]
    tta = {}
    for scen in SCENARIO_ORDER:
        for eps in EPSILONS:
            cells, best = [], (np.inf, None)
            for m in ALL_METHODS:
                if m not in _methods_for(scen):
                    cells.append("n/a")
                    continue
                sec, n_need, extra = time_to_accuracy(table, scen, m, eps)
                tta[(scen, eps, m)] = (sec, n_need, extra)
                cells.append(f"{sec:.3g} s (N≈{n_need:.2g}){'*' if extra else ''}")
                if sec < best[0]:
                    best = (sec, m)
            L.append(f"| {scen} | {eps:g} | " + " | ".join(cells) + f" | **{best[1]}** |")
    L.append("")

    L += ["## Acceptance (§1.7)", "",
          "Replicated mean of each cell vs the 252-date reference (t-interval, reference SE "
          "included); per-replicate 95% CI coverage for methods with a valid within-run SE.", "",
          "| Scenario | Method | cells within 3 SE | t-interval covers | CI coverage (mean over N) |",
          "|---|---|---|---|---:|"]
    for scen in SCENARIO_ORDER:
        for m in _methods_for(scen):
            g = acc[(acc["scenario"] == scen) & (acc["method"] == m)]
            cov = "n/a (RQMC)" if g["coverage"].isna().all() else f"{g['coverage'].mean():.3f}"
            L.append(f"| {scen} | {m} | {int(g['within3'].sum())}/{len(g)} | "
                     f"{int(g['t_covered'].sum())}/{len(g)} | {cov} |")
    L.append(f"\nOverall: {int(acc['within3'].sum())}/{len(acc)} cells within 3 SE; "
             f"{int(acc['t_covered'].sum())}/{len(acc)} t-intervals cover the reference.\n")

    L += ["## Cross-checks against earlier stages (indicative timings there)", ""]
    cv_a = top[(top["scenario"] == "asian_arith") & (top["method"] == "control_variate")].iloc[0]
    ex = sweep[(sweep["scenario"] == "asian_arith") & (sweep["method"] == "control_variate")
               & (sweep["n_paths"] == n_max)]["extra_json"].map(json.loads)
    rho = np.mean([e["rho"] for e in ex])
    L.append(f"- Control variate on the arithmetic Asian: mean rho = {rho:.6f}, within-run VRF "
             f"{cv_a['vrf_within_run']:.0f} (Stage 3: rho 0.999488, VRF 975.7).")
    for scen, stage4 in [("deep_barrier", "45"), ("paper_barrier", "6988")]:
        r = top[(top["scenario"] == scen) & (top["method"] == "rqmc")].iloc[0]
        L.append(f"- RQMC on {scen}: VRF {r['vrf']:.0f} (Stage 4: {stage4}).")
    for scen in ("deep_barrier", "paper_barrier"):
        r = top[(top["scenario"] == scen) & (top["method"] == "importance")].iloc[0]
        L.append(f"- Importance sampling on {scen}: within-run VRF {r['vrf_within_run']:.1f}, "
                 f"efficiency {r['eff']:.2f}.")
    L.append("")

    text = "\n".join(L)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "master_findings.md").write_text(text, encoding="utf-8")
    print(text)


def analyze():
    df = load_results()
    refs = references()
    missing = [s for s in SCENARIO_ORDER if s not in refs]
    if missing:
        raise RuntimeError(f"Missing 252-date references for {missing} — run the simulation "
                           "(the arithmetic reference is produced after the sweep).")
    sweep = with_references(df, refs)
    table = cell_table(sweep, baseline="plain")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "master_summary.csv", index=False)
    eff_cols = ["scenario", "method", "n_paths", "mean_price", "rmse", "bias", "vrf",
                "vrf_ci_low", "vrf_ci_high", "vrf_within_run", "time_ratio", "eff",
                "eff_ci_low", "eff_ci_high", "within_run_eff", "mean_runtime_sec"]
    table[eff_cols].to_csv(TABLES_DIR / "master_efficiency.csv", index=False)
    acc = acceptance(sweep, refs)
    make_figures(table)
    write_findings(df, sweep, table, acc, refs)
    print("Master comparison complete.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--analyze-only", action="store_true",
                   help="Skip simulation; rebuild figures/tables from the existing CSVs.")
    g.add_argument("--simulate-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        simulate()
    if not args.simulate_only:
        analyze()


if __name__ == "__main__":
    main()
