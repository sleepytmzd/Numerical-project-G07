import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import fit_loglog_slope
from src.config import R, SCENARIOS
from src.estimators import DEFAULT_CHUNK, make_seed_seq, seed_int
from src.paths import generate_paths
from src.payoffs import barrier_payoff
from src.plots import apply_style, save_fig
from src.results import _RESULTS_DIR, load_all_results, log_result

PERSON = "zaki"
EXPERIMENT_ID = "scheme_order"
TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

SCENARIO = SCENARIOS["paper_barrier"]
FINE_STEPS = 512
STEP_GRID = [4, 8, 16, 32, 64, 128, 256, 512]
N_PER_REP = 65_536
SCHEMES = ["euler_maruyama", "milstein"]
SCHEME_COLORS = {"euler_maruyama": "#4C72B0", "milstein": "#C44E52"}
SCHEME_LABELS = {"euler_maruyama": "Euler–Maruyama", "milstein": "Milstein"}
PAPER_DT = 1 / 252


# ---------------------------------------------------------------------------
# Analytic weak errors (noise-free check of the MC machinery)
# ---------------------------------------------------------------------------

def analytic_terminal_mean_error(m, scenario=SCENARIO):
    S0, r, T = scenario.S0, scenario.r, scenario.T
    return S0 * (1 + r * T / m) ** m - S0 * math.exp(r * T)


def analytic_second_moment_error(m, scheme, scenario=SCENARIO):
    # E[(S_T^scheme)^2] - E[S_T^2]; per-step E[F^2] differs between schemes.
    S0, r, sigma, T = scenario.S0, scenario.r, scenario.sigma, scenario.T
    dt = T / m
    step = (1 + r * dt) ** 2 + sigma**2 * dt
    if scheme == "milstein":
        step += 0.5 * sigma**4 * dt**2       # 0.25 sigma^4 dt^2 E[(Z^2-1)^2]
    return S0**2 * step**m - S0**2 * math.exp((2 * r + sigma**2) * T)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _coarsen(Zf, m):
    k = Zf.shape[1] // m
    return Zf.reshape(len(Zf), m, k).sum(axis=2) / math.sqrt(k)


class _Acc:
    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0

    def add(self, x):
        self.n += len(x)
        self.s += float(x.sum())
        self.ss += float((x * x).sum())

    def mean_se(self):
        mean = self.s / self.n
        var = max(self.ss / self.n - mean**2, 0.0) * self.n / (self.n - 1)
        return mean, math.sqrt(var / self.n)


def simulate():
    (_RESULTS_DIR / f"{PERSON}_{EXPERIMENT_ID}.csv").unlink(missing_ok=True)
    sc = SCENARIO
    disc = math.exp(-sc.r * sc.T)
    for rep in range(R):
        t_rep = time.perf_counter()
        ss = make_seed_seq(EXPERIMENT_ID, rep)
        rng = np.random.default_rng(ss)
        acc, timer = {}, {}
        for pos in range(0, N_PER_REP, DEFAULT_CHUNK):
            k = min(DEFAULT_CHUNK, N_PER_REP - pos)
            Zf = rng.standard_normal((k, FINE_STEPS))
            for m in STEP_GRID:
                Z = _coarsen(Zf, m)
                t0 = time.perf_counter()
                ex = generate_paths(sc, k, m, Z, "exact")
                timer[("exact", m)] = timer.get(("exact", m), 0.0) + time.perf_counter() - t0
                call_ex = disc * np.maximum(ex[:, -1] - sc.K, 0.0)
                bar_ex = disc * barrier_payoff(ex, sc.K, sc.B, sc.barrier_kind, "call")
                acc.setdefault(("exact", "price_call", m), _Acc()).add(call_ex)
                acc.setdefault(("exact", "price_barrier", m), _Acc()).add(bar_ex)
                for scheme in SCHEMES:
                    t0 = time.perf_counter()
                    p = generate_paths(sc, k, m, Z, scheme)
                    timer[(scheme, m)] = timer.get((scheme, m), 0.0) + time.perf_counter() - t0
                    d = p[:, -1] - ex[:, -1]
                    call = disc * np.maximum(p[:, -1] - sc.K, 0.0)
                    bar = disc * barrier_payoff(p, sc.K, sc.B, sc.barrier_kind, "call")
                    acc.setdefault((scheme, "strong_abs_terminal", m), _Acc()).add(np.abs(d))
                    acc.setdefault((scheme, "weak_terminal_mean", m), _Acc()).add(d)
                    acc.setdefault((scheme, "weak_call", m), _Acc()).add(call - call_ex)
                    acc.setdefault((scheme, "weak_barrier", m), _Acc()).add(bar - bar_ex)

        for (scheme, metric, m), a in acc.items():
            mean, se = a.mean_se()
            exact = analytic_terminal_mean_error(m) if metric == "weak_terminal_mean" else None
            log_result(person=PERSON, experiment_id=EXPERIMENT_ID, scenario=sc.name,
                       option_type="call", scheme=scheme, method=metric,
                       method_params=f"{{'fine_steps': {FINE_STEPS}}}", n_paths=a.n,
                       n_steps=m, replicate_id=rep, seed=seed_int(ss), price=mean,
                       std_error=se, ci_low=mean - 1.96 * se, ci_high=mean + 1.96 * se,
                       exact_price=exact, runtime_sec=timer[(scheme, m)],
                       extra={"dt": sc.T / m})
        print(f"replicate {rep + 1}/{R} done in {time.perf_counter() - t_rep:.1f}s")


def load_results():
    df = load_all_results()
    df = df[(df["person"] == PERSON) & (df["experiment_id"] == EXPERIMENT_ID)].copy()
    if df.empty:
        raise RuntimeError("No scheme-order results found — run the simulation first.")
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def build_summary(df):
    rows = []
    for (scheme, metric, m), g in df.groupby(["scheme", "method", "n_steps"]):
        mean = g["price"].mean()
        se = g["price"].std(ddof=1) / math.sqrt(len(g))
        exact = g["exact_price"].dropna()
        rows.append(dict(scheme=scheme, metric=metric, n_steps=int(m), dt=1.0 / m,
                         n_reps=len(g), total_paths=int(g["n_paths"].sum()), mean=mean,
                         se=se, resolved=abs(mean) > 3 * se,
                         analytic=float(exact.iloc[0]) if len(exact) else np.nan,
                         mean_within_se=g["std_error"].mean(),
                         path_time_sec=g["runtime_sec"].mean()))
    summary = pd.DataFrame(rows).sort_values(["metric", "scheme", "n_steps"])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES_DIR / "scheme_order_summary.csv", index=False)
    return summary


def _series(summary, scheme, metric):
    return summary[(summary["scheme"] == scheme) & (summary["metric"] == metric)].sort_values("dt")


def order_slope(sub):
    # Log-log slope of |error| vs dt over the statistically resolved points.
    sub = sub[sub["resolved"]]
    if len(sub) < 3:
        return float("nan"), 0
    return fit_loglog_slope(sub["dt"], sub["mean"].abs())[0], len(sub)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _ref_line(ax, dts, y0, order, label):
    dts = np.array(sorted(dts))
    ax.plot(dts, y0 * (dts / dts[-1]) ** order, "k:", linewidth=1, alpha=0.7, label=label)


def _error_panel(ax, summary, metric, title):
    for scheme in SCHEMES:
        sub = _series(summary, scheme, metric)
        slope, n_fit = order_slope(sub)
        c = SCHEME_COLORS[scheme]
        y = sub["mean"].abs().to_numpy()
        res = sub["resolved"].to_numpy()
        ax.errorbar(sub["dt"], y, yerr=1.96 * sub["se"], fmt="-", color=c, capsize=2,
                    linewidth=1.2, label=f"{SCHEME_LABELS[scheme]} (slope={slope:.2f})")
        ax.plot(sub["dt"][res], y[res], "o", color=c, markersize=5)
        ax.plot(sub["dt"][~res], y[~res], "o", color=c, markersize=5, markerfacecolor="white")
    ax.axvline(PAPER_DT, color="grey", linestyle="--", linewidth=0.8, label="dt = 1/252 (paper)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time step dt (years)")
    ax.set_ylabel("|weak error|")
    ax.set_title(title)


def make_figures(summary):
    apply_style()
    dts = summary["dt"].unique()

    # --- scheme_strong_order.png ---
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for scheme in SCHEMES:
        sub = _series(summary, scheme, "strong_abs_terminal")
        slope = fit_loglog_slope(sub["dt"], sub["mean"])[0]
        ax.errorbar(sub["dt"], sub["mean"], yerr=1.96 * sub["se"], fmt="o-", capsize=2,
                    markersize=5, color=SCHEME_COLORS[scheme],
                    label=f"{SCHEME_LABELS[scheme]} (slope={slope:.2f})")
    em = _series(summary, "euler_maruyama", "strong_abs_terminal")
    mil = _series(summary, "milstein", "strong_abs_terminal")
    _ref_line(ax, dts, em["mean"].iloc[-1] * 1.6, 0.5, r"$\Delta t^{1/2}$ reference")
    _ref_line(ax, dts, mil["mean"].iloc[-1] * 0.5, 1.0, r"$\Delta t^{1}$ reference")
    ax.axvline(PAPER_DT, color="grey", linestyle="--", linewidth=0.8, label="dt = 1/252 (paper)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time step dt (years)")
    ax.set_ylabel(r"Strong error  $E|S_T^{\Delta t} - S_T|$")
    ax.set_title("Strong (pathwise) convergence order")
    ax.legend(fontsize=8)
    save_fig(fig, "scheme_strong_order.png")
    plt.close(fig)

    # --- scheme_weak_order.png ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    _error_panel(ax, summary, "weak_terminal_mean", r"$E[S_T]$: MC (points) vs analytic (line)")
    grid = np.array(sorted(STEP_GRID))
    ax.plot(1.0 / grid, [abs(analytic_terminal_mean_error(m)) for m in grid], "k--",
            linewidth=1.2, zorder=5, label="analytic (both schemes)")
    ax.legend(fontsize=7)

    _error_panel(axes[1], summary, "weak_call", "European call (K=105)")
    _ref_line(axes[1], dts, _series(summary, "milstein", "weak_call")["mean"].abs().iloc[-1],
              1.0, r"$\Delta t^{1}$ reference")
    axes[1].legend(fontsize=7)

    ax = axes[2]
    _error_panel(ax, summary, "weak_barrier", "Up-and-in barrier (paper's option)")
    price = _series(summary, "exact", "price_barrier")
    sd = price["mean_within_se"].iloc[0] * math.sqrt(N_PER_REP)
    for n, style in [(65_536, "-."), (5_000, "--")]:
        ax.axhline(sd / math.sqrt(n), color="#55A868", linestyle=style, linewidth=1,
                   label=f"plain-MC SE at N={n:,}")
    ax.legend(fontsize=7)
    fig.suptitle("Weak convergence: both schemes are first order (hollow = not resolved from 0)",
                 y=1.02)
    fig.tight_layout()
    save_fig(fig, "scheme_weak_order.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def write_findings(df, summary):
    L = ["# Stage 4: SDE discretization order", "",
         f"Generated by `experiments/exp_scheme_order.py` from `results/raw/{PERSON}_{EXPERIMENT_ID}.csv`. "
         f"{R} replicates × {N_PER_REP:,} paths per step size (common random numbers: one "
         f"{FINE_STEPS}-step Brownian path per sample, coarsened to each dt). Errors are "
         "relative to the exact log-GBM scheme on the same grid.", "",
         "## Fitted orders (log-log slope vs dt; weak slopes use resolved points, |mean| > 3 SE)", "",
         "| Metric | Euler–Maruyama | Milstein | theory |", "|---|---:|---:|---|"]
    theory = {"strong_abs_terminal": "0.5 / 1.0", "weak_terminal_mean": "1 / 1",
              "weak_call": "1 / 1", "weak_barrier": "1 / 1"}
    orders = {}
    for metric in theory:
        cells = []
        for scheme in SCHEMES:
            sub = _series(summary, scheme, metric)
            if metric == "strong_abs_terminal":
                s, n = fit_loglog_slope(sub["dt"], sub["mean"])[0], len(sub)
            else:
                s, n = order_slope(sub)
            orders[(scheme, metric)] = s
            cells.append(f"{s:.2f} ({n} pts)")
        L.append(f"| {metric} | {cells[0]} | {cells[1]} | {theory[metric]} |")
    L.append("")

    L += ["## Analytic check: E[S_T^scheme] − E[S_T]", "",
          "| m | analytic | EM (MC) | Milstein (MC) | within 3 SE |", "|---:|---:|---:|---:|---|"]
    ok = True
    for m in STEP_GRID:
        a = analytic_terminal_mean_error(m)
        vals = [_series(summary, s, "weak_terminal_mean").set_index("n_steps").loc[m] for s in SCHEMES]
        within = all(abs(v["mean"] - a) <= 3 * v["se"] for v in vals)
        ok &= within
        L.append(f"| {m} | {a:+.6f} | {vals[0]['mean']:+.6f} ± {vals[0]['se']:.6f} | "
                 f"{vals[1]['mean']:+.6f} ± {vals[1]['se']:.6f} | {'yes' if within else 'NO'} |")
    L.append(f"\nAll within 3 SE: **{ok}**. Second-moment weak error (analytic, dt=1/256): "
             f"EM {analytic_second_moment_error(256, 'euler_maruyama'):+.5f}, "
             f"Milstein {analytic_second_moment_error(256, 'milstein'):+.5f}.\n")

    m0 = 256
    L += [f"## At dt = 1/{m0} (closest grid point to the paper's 1/252)", "",
          "| Metric | Euler–Maruyama | Milstein |", "|---|---:|---:|"]
    for metric in ["strong_abs_terminal", "weak_call", "weak_barrier"]:
        v = [_series(summary, s, metric).set_index("n_steps").loc[m0] for s in SCHEMES]
        L.append(f"| {metric} | {v[0]['mean']:+.6f} ± {v[0]['se']:.6f} | "
                 f"{v[1]['mean']:+.6f} ± {v[1]['se']:.6f} |")
    price = _series(summary, "exact", "price_barrier").set_index("n_steps").loc[m0]
    sd = price["mean_within_se"] * math.sqrt(N_PER_REP)
    worst = max(abs(_series(summary, s, "weak_barrier").set_index("n_steps").loc[m0]["mean"])
                for s in SCHEMES)
    L.append(f"\nBarrier payoff SD per path ≈ {sd:.2f}: plain-MC SE is {sd / math.sqrt(65536):.4f} "
             f"at N=65,536 and {sd / math.sqrt(5000):.4f} at the paper's N=5,000 — "
             f"{sd / math.sqrt(65536) / worst:.0f}× and {sd / math.sqrt(5000) / worst:.0f}× the "
             "larger scheme bias. Stage 2 measured −0.0006 (EM) / −0.0011 (Milstein) at 1/252.")

    cost = summary[summary["n_steps"] == max(STEP_GRID)].set_index(["scheme", "metric"])
    t = {s: cost.loc[(s, "strong_abs_terminal" if s != "exact" else "price_call"), "path_time_sec"]
         for s in ["exact", *SCHEMES]}
    L.append(f"\nPath-generation time per replicate at m={max(STEP_GRID)}: exact {t['exact']:.2f}s, "
             f"EM {t['euler_maruyama']:.2f}s, Milstein {t['milstein']:.2f}s (indicative; machine "
             f"{df['machine_id'].iloc[0]}).\n")

    text = "\n".join(L)
    (TABLES_DIR / "scheme_order_findings.md").write_text(text, encoding="utf-8")
    print(text)
    return orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        simulate()
    df = load_results()
    summary = build_summary(df)
    make_figures(summary)
    write_findings(df, summary)
    print("Scheme-order experiment complete.")


if __name__ == "__main__":
    main()
