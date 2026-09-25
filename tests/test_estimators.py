# Tests for estimators.py and benchmark.py.
import os
import subprocess
import sys
from pathlib import Path

import pytest

import src.results as results_mod
from src.benchmark import bootstrap_efficiency_ratio, exact_price, run_sweep, summarize
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

    def test_antithetic_chunking_matches_single_block(self):
        r1 = antithetic_mc(ASIAN, 5000, 50, make_seed_seq("t", 5), option="call", chunk=500)
        r2 = antithetic_mc(ASIAN, 5000, 50, make_seed_seq("t", 5), option="call", chunk=50_000)
        assert r1.price == pytest.approx(r2.price, rel=1e-12)
        assert r1.std_error == pytest.approx(r2.std_error, rel=1e-12)

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


def _tiny_sweep(tmp_path, monkeypatch, cases):
    monkeypatch.setattr(results_mod, "_RESULTS_DIR", tmp_path)
    run_sweep(person="test", experiment_id="seeds", cases=cases,
              methods=["plain", "antithetic"], n_grid=[64], R=3, n_steps=10,
              schemes=["exact", "milstein"], verbose=False)
    return results_mod.load_all_results(tmp_path)


class TestSweepSeeding:
    def test_no_seed_shared_across_cases_or_methods(self, tmp_path, monkeypatch):
        df = _tiny_sweep(tmp_path, monkeypatch,
                         [("paper_barrier", "call"), ("paper_asian_geo", "call"),
                          ("paper_asian_geo", "put")])
        exact_rows = df[df["scheme"] == "exact"]
        assert exact_rows["seed"].is_unique

    def test_schemes_share_normals(self, tmp_path, monkeypatch):
        df = _tiny_sweep(tmp_path, monkeypatch, [("paper_barrier", "call")])
        key = ["method", "n_paths", "replicate_id"]
        a = df[df["scheme"] == "exact"].set_index(key)["seed"].sort_index()
        b = df[df["scheme"] == "milstein"].set_index(key)["seed"].sort_index()
        assert (a == b).all()

    def test_seeds_stable_across_processes(self):
        # str hash() is salted per process; seeds must not depend on it.
        code = ("from src.benchmark import *; import src.results as r, tempfile, pathlib;"
                "d = pathlib.Path(tempfile.mkdtemp()); r._RESULTS_DIR = d;"
                "run_sweep(person='t', experiment_id='x', cases=[('paper_barrier','call')],"
                "methods=['plain'], n_grid=[16], R=2, n_steps=4, verbose=False);"
                "print(list(r.load_all_results(d)['seed']))")
        root = Path(__file__).resolve().parent.parent
        outs = set()
        for salt in ("1", "2"):
            env = {**os.environ, "PYTHONHASHSEED": salt}
            outs.add(subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                                    capture_output=True, text=True, check=True).stdout)
        assert len(outs) == 1


class TestBootstrap:
    @pytest.fixture
    def two_n_df(self, tmp_path, monkeypatch):
        monkeypatch.setattr(results_mod, "_RESULTS_DIR", tmp_path)
        run_sweep(person="test", experiment_id="boot", cases=[("paper_asian_geo", "call")],
                  methods=["plain", "antithetic"], n_grid=[64, 256], R=5, n_steps=10,
                  verbose=False)
        return results_mod.load_all_results(tmp_path)

    def test_refuses_to_pool_across_n(self, two_n_df):
        keys = dict(scenario="paper_asian_geo", option_type="call", scheme="exact")
        with pytest.raises(ValueError, match="several cells"):
            bootstrap_efficiency_ratio(two_n_df, {**keys, "method": "antithetic"},
                                       {**keys, "method": "plain"})

    def test_single_cell_ok(self, two_n_df):
        keys = dict(scenario="paper_asian_geo", option_type="call", scheme="exact", n_paths=256)
        ratio, lo, hi = bootstrap_efficiency_ratio(two_n_df, {**keys, "method": "antithetic"},
                                                   {**keys, "method": "plain"}, n_boot=200)
        assert ratio > 0 and 0 < lo <= hi
