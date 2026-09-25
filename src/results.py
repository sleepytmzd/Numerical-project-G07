import csv
import glob
import json
import os
import platform
import time
import uuid
from pathlib import Path

import pandas as pd

# Where raw CSVs live, relative to the project root
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"

# Canonical column order
COLUMNS = [
    "run_id",
    "timestamp",
    "person",
    "experiment_id",
    "scenario",
    "option_type",
    "scheme",
    "method",
    "method_params",
    "n_paths",
    "n_steps",
    "replicate_id",
    "seed",
    "machine_id",
    "price",
    "std_error",
    "ci_low",
    "ci_high",
    "exact_price",
    "abs_error",
    "runtime_sec",
    "extra_json",
]


def _machine_id() -> str:
    """A short, stable machine identifier (hostname + processor tag)."""
    return f"{platform.node()}_{platform.machine()}"


def log_result(
    *,
    person: str,
    experiment_id: str,
    scenario: str,
    option_type: str,
    scheme: str,
    method: str,
    method_params: str = "",
    n_paths: int,
    n_steps: int,
    replicate_id: int,
    seed: int,
    price: float,
    std_error: float,
    ci_low: float,
    ci_high: float,
    exact_price: float | None = None,
    runtime_sec: float = 0.0,
    extra: dict | None = None,
) -> None:
    """Append one row to ``results/raw/<person>_<experiment_id>.csv``.
    Creates the file with a header row if it does not exist yet.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = _RESULTS_DIR / f"{person}_{experiment_id}.csv"

    abs_error = abs(price - exact_price) if exact_price is not None else None

    row = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "person": person,
        "experiment_id": experiment_id,
        "scenario": scenario,
        "option_type": option_type,
        "scheme": scheme,
        "method": method,
        "method_params": method_params,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "replicate_id": replicate_id,
        "seed": seed,
        "machine_id": _machine_id(),
        "price": price,
        "std_error": std_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "exact_price": exact_price if exact_price is not None else "",
        "abs_error": abs_error if abs_error is not None else "",
        "runtime_sec": runtime_sec,
        "extra_json": json.dumps(extra) if extra else "",
    }

    write_header = not filepath.exists()
    with open(filepath, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_all_results(results_dir: Path | None = None) -> pd.DataFrame:
    """Glob all ``results/raw/*.csv`` files into one DataFrame.
    Numeric columns are coerced appropriately; ``extra_json`` is left as a
    string (callers can ``json.loads`` it when needed).
    """
    d = results_dir or _RESULTS_DIR
    files = sorted(glob.glob(str(d / "*.csv")))
    if not files:
        return pd.DataFrame(columns=COLUMNS)

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # Coerce numerics that might have been read as strings
    numeric_cols = [
        "n_paths", "n_steps", "replicate_id", "seed",
        "price", "std_error", "ci_low", "ci_high",
        "exact_price", "abs_error", "runtime_sec",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
