"""Shared analysis helpers for the Stage-5 experiments (Tamzeed Mahfuz, 2105012).

Used by ``exp_importance.py`` and ``exp_master.py``.  Everything here is
derived from raw per-replicate rows, per WORKPLAN §1.6.
"""

import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The findings contain ε, σ, θ...; Windows consoles default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.benchmark import bootstrap_efficiency_ratio, exact_price, summarize
from src.config import N_STEPS, SCENARIOS
from src.estimators import make_seed_seq, seed_int
from src.results import _RESULTS_DIR, load_all_results, log_result

TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

# Methods whose per-run std_error is NOT a valid error bar (RQMC: one scramble).
NO_WITHIN_RUN_SE = {"rqmc", "rqmc_incremental"}

REF_PERSON, REF_EXPERIMENT = "tamzeed", "master_reference"
REF_SCRAMBLES, REF_N = 64, 65_536


def run_arith_reference():
    """64 independent Brownian-bridge scrambles x 65,536 paths on ``asian_arith``,
    mirroring Stage 4's barrier references.  Logged to its own CSV so it never
    mixes with sweep rows (and never distorts the master sweep's timings)."""
    from src.vr_qmc import rqmc_mc

    (_RESULTS_DIR / f"{REF_PERSON}_{REF_EXPERIMENT}.csv").unlink(missing_ok=True)
    name = "asian_arith"
    scenario = SCENARIOS[name]
    print(f"reference: {name}, {REF_SCRAMBLES} scrambles x {REF_N}")
    for rep in range(REF_SCRAMBLES):
        ss = make_seed_seq("master_reference", zlib.crc32(name.encode()), rep)
        res = rqmc_mc(scenario, REF_N, N_STEPS, ss, option="call")
        log_result(person=REF_PERSON, experiment_id=REF_EXPERIMENT, scenario=name,
                   option_type="call", scheme="exact", method="rqmc_reference",
                   method_params="{'construction': 'bridge'}", n_paths=REF_N,
                   n_steps=N_STEPS, replicate_id=rep, seed=seed_int(ss),
                   price=res.price, std_error=res.std_error, ci_low=res.ci_low,
                   ci_high=res.ci_high, exact_price=None, runtime_sec=res.runtime_sec,
                   extra={**res.extra, "purpose": "arithmetic_asian_reference"})


def references():
    """{scenario: (price, SE, source)} — 252-date discrete-monitoring targets.

    Barriers: Stage 4's 64-scramble RQMC reference rows (``zaki_qmc.csv``).
    Geometric Asian: the discrete closed form.  Arithmetic Asian: Stage 5's own
    64-scramble reference (``tamzeed_master_reference.csv``), when present.
    """
    df = load_all_results()
    ref_rows = df[df["method"] == "rqmc_reference"]
    refs = {}
    for name, who in [("paper_barrier", "zaki"), ("deep_barrier", "zaki"),
                      ("asian_arith", REF_PERSON)]:
        p = ref_rows[(ref_rows["scenario"] == name) & (ref_rows["person"] == who)]["price"]
        if len(p):
            refs[name] = (float(p.mean()), float(p.std(ddof=1) / math.sqrt(len(p))),
                          f"{len(p)} RQMC bridge scrambles x {REF_N:,} paths ({who})")
    geo = exact_price(SCENARIOS["paper_asian_geo"], "call", N_STEPS)
    refs["paper_asian_geo"] = (float(geo), 0.0, "discrete geometric closed form")
    return refs


def with_references(df, refs):
    """Copy of ``df`` whose ``exact_price`` is the 252-date reference (NaN if none)."""
    out = df.copy()
    out["exact_price"] = out["scenario"].map(lambda s: refs.get(s, (np.nan,))[0])
    return out


def cell_table(df, baseline="plain", compare_to=(), n_boot=2000):
    """``summarize`` plus VRF (with F-distribution CI), within-run VRF, time
    ratio and bootstrap efficiency ratios, every ratio taken against
    ``baseline`` in the SAME (scenario, option, scheme, N) cell.

    ``compare_to`` adds extra efficiency-ratio columns against other methods
    (e.g. ``("antithetic",)`` gives ``eff_vs_antithetic``).
    """
    summary = summarize(df)
    summary.loc[summary["method"].isin(NO_WITHIN_RUN_SE), "ci_coverage"] = np.nan
    mean_se2 = (df.assign(se2=df["std_error"] ** 2)
                  .groupby(["scenario", "option_type", "scheme", "method", "n_paths"])["se2"]
                  .mean())

    rows = []
    for _, row in summary.iterrows():
        cell = dict(scenario=row["scenario"], option_type=row["option_type"],
                    scheme=row["scheme"], n_paths=row["n_paths"])
        out = {}
        for other in (baseline, *compare_to):
            ref = summary[(summary["scenario"] == row["scenario"])
                          & (summary["option_type"] == row["option_type"])
                          & (summary["scheme"] == row["scheme"])
                          & (summary["n_paths"] == row["n_paths"])
                          & (summary["method"] == other)]
            tag = "" if other == baseline else f"_vs_{other}"
            if ref.empty:
                continue
            ref = ref.iloc[0]
            if other == baseline:
                vrf = ref["var_across_reps"] / row["var_across_reps"]
                d1, d2 = int(ref["n_reps"]) - 1, int(row["n_reps"]) - 1
                out["vrf"] = vrf
                out["vrf_ci_low"] = vrf / f_dist.ppf(0.975, d1, d2)
                out["vrf_ci_high"] = vrf / f_dist.ppf(0.025, d1, d2)
                out["time_ratio"] = row["mean_runtime_sec"] / ref["mean_runtime_sec"]
                if row["method"] in NO_WITHIN_RUN_SE:
                    out["vrf_within_run"] = np.nan
                else:
                    k = (row["scenario"], row["option_type"], row["scheme"])
                    out["vrf_within_run"] = (mean_se2[(*k, other, row["n_paths"])]
                                             / mean_se2[(*k, row["method"], row["n_paths"])])
            if row["method"] == other:
                eff = (1.0, 1.0, 1.0)
            else:
                eff = bootstrap_efficiency_ratio(df, {**cell, "method": row["method"]},
                                                 {**cell, "method": other}, n_boot=n_boot)
            name = "eff" if other == baseline else f"eff{tag}"
            out[name], out[f"{name}_ci_low"], out[f"{name}_ci_high"] = eff
        rows.append(out)
    table = pd.concat([summary.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    table["within_run_eff"] = table["vrf_within_run"] / table["time_ratio"]
    return table.sort_values(["scenario", "method", "n_paths"]).reset_index(drop=True)


def acceptance(df, refs):
    """§1.7.1/§1.7.2 per cell: the replicated mean's t-interval vs the reference,
    and the per-replicate 95% CI coverage (valid-SE methods only)."""
    out = []
    for (scenario, method, n), g in df.groupby(["scenario", "method", "n_paths"]):
        if scenario not in refs:
            continue
        ref, ref_se, _ = refs[scenario]
        p = g["price"]
        tcrit = t_dist.ppf(0.975, len(p) - 1)
        se = math.sqrt(p.var(ddof=1) / len(p) + ref_se**2)
        z = abs(p.mean() - ref) / se
        cov = (np.nan if method in NO_WITHIN_RUN_SE
               else float(((g["ci_low"] <= ref) & (ref <= g["ci_high"])).mean()))
        out.append(dict(scenario=scenario, method=method, n_paths=n, z=z,
                        within3=z <= 3.0, t_covered=z <= tcrit, coverage=cov))
    return pd.DataFrame(out)


def fmt_ci(point, lo, hi, digits=1):
    if point is None or (isinstance(point, float) and math.isnan(point)):
        return "—"
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
