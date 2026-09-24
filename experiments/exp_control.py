"""Stage 3: control-variate experiments (Aritra Debnath, 2105010).

Runs plain MC and independently piloted control variates for the arithmetic
Asian call and for near/deep up-and-in barriers.  It also computes a large
plain-MC arithmetic-Asian reference because that option has no closed form.

Usage
-----
    python experiments/exp_control.py
    python experiments/exp_control.py --analyze-only
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import bootstrap_efficiency_ratio, run_sweep
from src.config import N_GRID, N_STEPS, R, SCENARIOS
from src.estimators import make_seed_seq, plain_mc, seed_int
from src.plots import METHOD_COLORS, apply_style, save_fig
from src.results import _RESULTS_DIR, load_all_results, log_result
from src.vr_control import control_variate_mc  # noqa: F401; registers estimator

PERSON = "aritra"
EXPERIMENT_ID = "control"
METHODS = ["plain", "control_variate"]
CASES = [
    ("asian_arith", "call"),
    ("paper_barrier", "call"),
    ("deep_barrier", "call"),
]
REFERENCE_PATHS = 2_000_000
TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"


def simulate(reference_paths=REFERENCE_PATHS):
    """Run the frozen grid and add one large plain arithmetic-Asian reference."""
    output = _RESULTS_DIR / f"{PERSON}_{EXPERIMENT_ID}.csv"
    output.unlink(missing_ok=True)
    run_sweep(
        person=PERSON,
        experiment_id=EXPERIMENT_ID,
        cases=CASES,
        methods=METHODS,
        n_grid=N_GRID,
        R=R,
        n_steps=N_STEPS,
        schemes=["exact"],
    )

    scenario = SCENARIOS["asian_arith"]
    seed_seq = make_seed_seq("control_arithmetic_plain_reference", reference_paths)
    result = plain_mc(
        scenario,
        reference_paths,
        N_STEPS,
        seed_seq,
        option="call",
        scheme="exact",
    )
    log_result(
        person=PERSON,
        experiment_id=EXPERIMENT_ID,
        scenario=scenario.name,
        option_type="call",
        scheme="exact",
        method="plain_reference",
        method_params=f"n_paths={reference_paths}",
        n_paths=reference_paths,
        n_steps=N_STEPS,
        replicate_id=0,
        seed=seed_int(seed_seq),
        price=result.price,
        std_error=result.std_error,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        exact_price=None,
        runtime_sec=result.runtime_sec,
        extra={"purpose": "arithmetic_asian_no_closed_form_reference"},
    )


def load_results():
    df = load_all_results()
    df = df[(df["person"] == PERSON) & (df["experiment_id"] == EXPERIMENT_ID)].copy()
    if df.empty:
        raise RuntimeError("No Stage-3 control results found; run the simulation first.")
    return df


def _extras(rows):
    return [json.loads(value) for value in rows["extra_json"]]


def build_summary(df):
    """One analysis row per scenario and N, keeping precise within-run VRFs."""
    rows = []
    sweep = df[df["method"].isin(METHODS)]
    for (scenario, n_paths), group in sweep.groupby(["scenario", "n_paths"]):
        plain = group[group["method"] == "plain"]
        control = group[group["method"] == "control_variate"]
        if len(plain) != R or len(control) != R:
            raise RuntimeError(
                f"Expected {R} replicates for {scenario}, N={n_paths}; "
                f"found plain={len(plain)}, control={len(control)}"
            )
        extra = _extras(control)
        across_vrf = plain["price"].var(ddof=1) / control["price"].var(ddof=1)
        time_ratio = plain["runtime_sec"].mean() / control["runtime_sec"].mean()
        keys = dict(
            scenario=scenario,
            option_type="call",
            scheme="exact",
            n_paths=n_paths,
        )
        efficiency, efficiency_lo, efficiency_hi = bootstrap_efficiency_ratio(
            sweep,
            {**keys, "method": "control_variate"},
            {**keys, "method": "plain"},
        )
        rows.append({
            "scenario": scenario,
            "n_paths": int(n_paths),
            "n_reps": len(control),
            "plain_mean_price": plain["price"].mean(),
            "control_mean_price": control["price"].mean(),
            "control_mean_se": control["price"].std(ddof=1) / math.sqrt(len(control)),
            "plain_mean_std_error": plain["std_error"].mean(),
            "control_mean_std_error": control["std_error"].mean(),
            "rho": np.mean([item["rho"] for item in extra]),
            "b_hat": np.mean([item["b_hat"] for item in extra]),
            "pilot_paths": int(round(np.mean([item["pilot_paths"] for item in extra]))),
            "observed_within_vrf": np.mean([item["observed_vrf"] for item in extra]),
            "theoretical_vrf": np.mean([item["theoretical_vrf"] for item in extra]),
            "across_replicate_vrf": across_vrf,
            "plain_over_control_time": time_ratio,
            "within_run_efficiency_ratio": (
                np.mean([item["observed_vrf"] for item in extra]) * time_ratio
            ),
            "bootstrap_efficiency_ratio": efficiency,
            "bootstrap_efficiency_ci_low": efficiency_lo,
            "bootstrap_efficiency_ci_high": efficiency_hi,
        })
    summary = pd.DataFrame(rows).sort_values(["scenario", "n_paths"])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES_DIR / "control_summary.csv", index=False)
    return summary


def make_figures(summary):
    apply_style()
    color = METHOD_COLORS["control_variate"]

    # Arithmetic-Asian headline: enormous, stable VRF over the shared N grid.
    asian = summary[summary["scenario"] == "asian_arith"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(asian["n_paths"], asian["observed_within_vrf"], "o-", color=color,
            label="Observed within-run VRF")
    ax.plot(asian["n_paths"], asian["theoretical_vrf"], "s--", color="#333333",
            label=r"Theory $1/(1-\rho^2)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Production paths (N)")
    ax.set_ylabel("Variance-reduction factor")
    ax.set_title("Arithmetic Asian: geometric-Asian control")
    ax.legend()
    save_fig(fig, "cv_asian_vrf.png")
    plt.close(fig)

    # Theory-vs-observation diagnostic across all three use cases.
    labels = {
        "asian_arith": "Arithmetic Asian",
        "paper_barrier": "Near barrier (B=110.68)",
        "deep_barrier": "Deep barrier (B=140)",
    }
    markers = {"asian_arith": "o", "paper_barrier": "s", "deep_barrier": "^"}
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for scenario, group in summary.groupby("scenario"):
        ax.scatter(
            group["rho"].abs(),
            group["observed_within_vrf"],
            s=38,
            marker=markers[scenario],
            label=labels[scenario],
        )
    rho_min = max(0.0, float(summary["rho"].abs().min()) - 0.04)
    rho_max = min(0.9999999, float(summary["rho"].abs().max()) + 0.00001)
    # Resolve the sharp rise near rho=1 by spacing points logarithmically in
    # (1-rho), rather than using a linear grid that visually skips that region.
    curve = 1.0 - np.geomspace(1.0 - rho_min, 1.0 - rho_max, 1000)
    ax.plot(curve, 1.0 / (1.0 - curve**2), "k--", linewidth=1.2,
            label=r"$1/(1-\rho^2)$")
    ax.set_yscale("log")
    ax.set_xlabel(r"Absolute target-control correlation $|\rho|$")
    ax.set_ylabel("Observed within-run VRF")
    ax.set_title("Control-variate theory matches observed reduction")
    ax.legend(fontsize=8)
    save_fig(fig, "cv_rho_vs_vrf.png")
    plt.close(fig)

    # Same vanilla control, radically different effectiveness by barrier level.
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for scenario, label, marker in [
        ("paper_barrier", "B=110.6772 (near)", "o"),
        ("deep_barrier", "B=140 (deep)", "s"),
    ]:
        group = summary[summary["scenario"] == scenario]
        ax.plot(group["n_paths"], group["observed_within_vrf"], marker + "-",
                label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Production paths (N)")
    ax.set_ylabel("Variance-reduction factor")
    ax.set_title("European-call control weakens for a deep barrier")
    ax.legend()
    save_fig(fig, "cv_barrier_collapse.png")
    plt.close(fig)


def write_findings(df, summary):
    n_max = max(N_GRID)
    reference = df[df["method"] == "plain_reference"].iloc[0]
    asian = summary[(summary["scenario"] == "asian_arith") &
                    (summary["n_paths"] == n_max)].iloc[0]
    combined_se = math.sqrt(reference["std_error"]**2 + asian["control_mean_se"]**2)
    z_score = abs(asian["control_mean_price"] - reference["price"]) / combined_se

    lines = [
        "# Stage 3 control-variate findings",
        "",
        "Generated by `experiments/exp_control.py` from "
        "`results/raw/aritra_control.csv`.",
        "",
        "## Arithmetic Asian validation",
        "",
        f"- Large plain-MC reference ({int(reference['n_paths']):,} paths): "
        f"{reference['price']:.6f} ± {1.96 * reference['std_error']:.6f} (95% CI).",
        f"- Control-variate mean at N={n_max:,} over R={R}: "
        f"{asian['control_mean_price']:.6f} ± {1.96 * asian['control_mean_se']:.6f} "
        "(95% CI for the replicated mean).",
        f"- Difference is {z_score:.2f} combined standard errors; "
        f"rho={asian['rho']:.6f}, b_hat={asian['b_hat']:.6f}, "
        f"within-run VRF={asian['observed_within_vrf']:.1f}.",
        "",
        "## Largest-N comparison",
        "",
        "| Scenario | rho | b_hat | within-run VRF | theory | efficiency ratio* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in ["asian_arith", "paper_barrier", "deep_barrier"]:
        row = summary[(summary["scenario"] == scenario) &
                      (summary["n_paths"] == n_max)].iloc[0]
        lines.append(
            f"| {scenario} | {row['rho']:.6f} | {row['b_hat']:.6f} | "
            f"{row['observed_within_vrf']:.1f} | {row['theoretical_vrf']:.1f} | "
            f"{row['within_run_efficiency_ratio']:.1f} |"
        )
    lines += [
        "",
        "\\* Pilot cost is included in control-variate runtime. The bootstrap "
        "efficiency estimates and 95% CIs are in `control_summary.csv`; R=20 makes "
        "across-replicate variance ratios much noisier than within-run VRFs.",
        "",
    ]
    (TABLES_DIR / "control_findings.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--reference-paths", type=int, default=REFERENCE_PATHS)
    args = parser.parse_args()
    if not args.analyze_only:
        simulate(reference_paths=args.reference_paths)
    df = load_results()
    summary = build_summary(df)
    make_figures(summary)
    write_findings(df, summary)
    print("Control-variate experiment complete.")


if __name__ == "__main__":
    main()
