"""
Tests for estimators.py and benchmark.py.

Stage 2 (Nafis Nahian, 2105007).
"""

import math

import numpy as np
import pytest

import src.results as results_mod
from src.benchmark import exact_price, run_sweep, summarize
from src.config import SCENARIOS
from src.estimators import antithetic_mc, make_seed_seq, plain_mc, seed_int

ASIAN = SCENARIOS["paper_asian_geo"]


class TestSeeding:
    def test_deterministic(self):
        a = make_seed_seq("exp", 1, 2, 3)
        b = make_seed_seq("exp", 1, 2, 3)
        assert seed_int(a) == seed_int(b)

    def test_distinct_keys_differ(self):
        a = make_seed_seq("exp", 1, 2, 3)
        b = make_seed_seq("exp", 1, 2, 4)
        assert seed_int(a) != seed_int(b)


class TestPlainAndAntithetic:
    @pytest.mark.parametrize("option", ["call", "put"])
    def test_within_3se_of_closed_form(self, option):
        ss = make_seed_seq("test_estimators", 0)
        result = plain_mc(ASIAN, 20_000, 252, ss, option=option, scheme="exact")
        exact = exact_price(ASIAN, option, 252)
        assert abs(result.price - exact) < 3 * result.std_error

    def test_antithetic_reduces_se(self):
        ss1 = make_seed_seq("test_estimators", 1)
        ss2 = make_seed_seq("test_estimators", 2)
        plain = plain_mc(ASIAN, 20_000, 252, ss1, option="call", scheme="exact")
        anti = antithetic_mc(ASIAN, 20_000, 252, ss2, option="call", scheme="exact")
        assert anti.std_error < plain.std_error

    def test_antithetic_odd_paths_raises(self):
        ss = make_seed_seq("test_estimators", 3)
        with pytest.raises(ValueError):
            antithetic_mc(ASIAN, 101, 10, ss, option="call")

    def test_chunking_matches_single_block(self):
        ss_a = make_seed_seq("test_estimators", 4)
        ss_b = make_seed_seq("test_estimators", 4)
        r_chunked = plain_mc(ASIAN, 5000, 50, ss_a, option="call", scheme="exact", chunk=500)
        r_single = plain_mc(ASIAN, 5000, 50, ss_b, option="call", scheme="exact", chunk=50_000)
        assert r_chunked.price == pytest.approx(r_single.price, rel=1e-12)


class TestRunSweep:
    def test_smoke(self, tmp_path, monkeypatch):
        monkeypatch.setattr(results_mod, "_RESULTS_DIR", tmp_path)
        run_sweep(
            person="test", experiment_id="smoke",
            cases=[("paper_asian_geo", "call")],
            methods=["plain", "antithetic"],
            n_grid=[64, 256], R=3, n_steps=20, schemes=["exact"], verbose=False,
        )
        files = list(tmp_path.glob("*.csv"))
        assert len(files) == 1
        df = results_mod.load_all_results(tmp_path)
        assert len(df) == 2 * 2 * 3  # methods x N x R
        assert set(df["method"]) == {"plain", "antithetic"}

        summary = summarize(df)
        assert (summary["n_reps"] == 3).all()
        assert not summary["ci_coverage"].isna().any()
