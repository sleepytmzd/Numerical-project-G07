"""
Shared plotting style — every stage's figures should look like one document.

Stage 2 (Nafis Nahian, 2105007).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

# One color/label per method, shared across all stages' figures.
METHOD_COLORS = {
    "plain": "#4C72B0",
    "antithetic": "#DD8452",
    "control_variate": "#55A868",
    "rqmc": "#C44E52",
    "importance": "#8172B2",
}
METHOD_LABELS = {
    "plain": "Plain MC",
    "antithetic": "Antithetic",
    "control_variate": "Control variate",
    "rqmc": "RQMC (Sobol + bridge)",
    "importance": "Importance sampling",
}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def save_fig(fig, name):
    """Save ``fig`` to results/figures/<name> (adds .png if missing)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".png"):
        name += ".png"
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    return path


def ci_vs_n_plot(ax, summary_df, exact, methods, n_col="n_paths",
                 price_col="mean_price", se_col="mean_std_error",
                 exact_label="Exact (closed form)"):
    """Estimate +/- 95% CI vs N (log-x), one offset series per method.

    ``summary_df`` is the output of ``benchmark.summarize`` restricted to one
    (scenario, option, scheme) — i.e. it still has one row per method x N.
    """
    n_methods = len(methods)
    for i, method in enumerate(methods):
        sub = summary_df[summary_df["method"] == method].sort_values(n_col)
        if sub.empty:
            continue
        # small log-space offset so overlapping series are visible
        offset = 1.0 + 0.03 * (i - (n_methods - 1) / 2)
        x = sub[n_col].to_numpy() * offset
        y = sub[price_col].to_numpy()
        ci = 1.959963984540054 * sub[se_col].to_numpy()
        ax.errorbar(x, y, yerr=ci, fmt="o-", capsize=3, markersize=4,
                    color=METHOD_COLORS.get(method), label=METHOD_LABELS.get(method, method))
    if exact is not None:
        ax.axhline(exact, color="black", linestyle="--", linewidth=1, label=exact_label)
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths (N)")
    ax.set_ylabel("Estimated Payoff")
    ax.legend(fontsize=8)


def loglog_rmse_plot(ax, summary_df, methods, n_col="n_paths", rmse_col="rmse"):
    """RMSE vs N on log-log axes, with a fitted slope in the legend and an
    N^-1/2 reference line."""
    for method in methods:
        sub = summary_df[summary_df["method"] == method].sort_values(n_col)
        if sub.empty:
            continue
        x = sub[n_col].to_numpy(dtype=float)
        y = sub[rmse_col].to_numpy(dtype=float)
        mask = (x > 0) & (y > 0)
        label = METHOD_LABELS.get(method, method)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
            label += f" (slope={slope:.2f})"
        ax.plot(x, y, "o-", markersize=4, color=METHOD_COLORS.get(method), label=label)

    all_n = summary_df[n_col].dropna()
    if len(all_n):
        n_ref = np.array(sorted(all_n.unique()), dtype=float)
        ref = summary_df[rmse_col].median() * np.sqrt(n_ref[0] / n_ref) if len(n_ref) else None
        if ref is not None:
            ax.plot(n_ref, ref, "k--", linewidth=1, alpha=0.6, label=r"$N^{-1/2}$ reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths (N)")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)
