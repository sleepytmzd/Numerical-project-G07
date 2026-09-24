"""
Tests for analytic closed-form pricers.

Stage 1 (Arnob Biswas, 2105015).

Pins every value in WORKPLAN §2; asserts put–call parity; asserts in-out
parity at both paper_barrier and mid_barrier (trap #6).
"""

import math

import pytest

from src.analytic import (
    barrier_closed_form,
    barrier_closed_form_bgk,
    bs_call,
    bs_put,
    geometric_asian_closed_form,
)
from src.config import N_STEPS, SCENARIOS

# Common parameters shared by all scenarios
S0, r, sigma, T = 100.0, 0.03, 0.2, 1.0
K = 105.0


# ============================================================================
# §2 Reference values
# ============================================================================

class TestReferenceValues:
    """Pin every value in WORKPLAN §2."""

    def test_vanilla_european_call(self):
        """Vanilla European call, K=105 → 7.12806."""
        price = bs_call(S0, K, r, sigma, T)
        assert price == pytest.approx(7.12806, abs=1e-4), f"Got {price}"

    def test_up_and_in_call_paper_barrier(self):
        """Up-and-in call, B=110.6772, continuous → 7.105528."""
        B = 110.6772
        price = barrier_closed_form(S0, K, B, r, sigma, T,
                                    kind="up_and_in", option="call")
        assert price == pytest.approx(7.105528, abs=1e-4), f"Got {price}"

    def test_up_and_out_call_paper_barrier(self):
        """Up-and-out call (in-out parity) → 0.02254."""
        B = 110.6772
        price = barrier_closed_form(S0, K, B, r, sigma, T,
                                    kind="up_and_out", option="call")
        assert price == pytest.approx(0.02254, abs=1e-3), f"Got {price}"

    def test_geometric_asian_call_continuous(self):
        """Geometric Asian call, continuous averaging → 2.98488."""
        price = geometric_asian_closed_form(S0, K, r, sigma, T,
                                            option="call", m=None)
        assert price == pytest.approx(2.98488, abs=1e-3), f"Got {price}"


# ============================================================================
# Put–Call Parity
# ============================================================================

class TestPutCallParity:
    """C - P = S0 - K*exp(-rT) for European options."""

    def test_bs_put_call_parity(self):
        parity_rhs = S0 - K * math.exp(-r * T)
        c = bs_call(S0, K, r, sigma, T)
        p = bs_put(S0, K, r, sigma, T)
        assert (c - p) == pytest.approx(parity_rhs, abs=1e-10)

    def test_geometric_asian_put_call_parity(self):
        """The geometric-average put–call parity:
        C_G - P_G = e^{-rT} * (S0 * e^{bT} - K)
        where b is the Kemna–Vorst cost of carry.
        """
        c = geometric_asian_closed_form(S0, K, r, sigma, T, option="call")
        p = geometric_asian_closed_form(S0, K, r, sigma, T, option="put")
        b = 0.5 * (r - sigma**2 / 6)
        expected = math.exp(-r * T) * (S0 * math.exp(b * T) - K)
        assert (c - p) == pytest.approx(expected, abs=1e-10)

    def test_geometric_asian_discrete_put_call_parity(self):
        """Same parity for the discrete variant (m=252)."""
        m = 252
        c = geometric_asian_closed_form(S0, K, r, sigma, T, option="call", m=m)
        p = geometric_asian_closed_form(S0, K, r, sigma, T, option="put", m=m)
        sigma_G_sq = sigma**2 * (m + 1) * (2 * m + 1) / (6 * m**2)
        b = 0.5 * sigma_G_sq + (r - 0.5 * sigma**2) * (m + 1) / (2 * m)
        expected = math.exp(-r * T) * (S0 * math.exp(b * T) - K)
        assert (c - p) == pytest.approx(expected, abs=1e-10)


# ============================================================================
# In-Out Parity for barrier options
# ============================================================================

class TestInOutParity:
    """C_ui + C_uo = C_vanilla for barrier options.

    Checked at paper_barrier AND mid_barrier (trap #6: the paper_barrier
    test is weak because C_uo ≈ 0.023 is only 0.3% of C_vanilla).
    """

    def test_in_out_parity_paper_barrier(self):
        """In-out parity at B=110.6772."""
        B = 110.6772
        c_ui = barrier_closed_form(S0, K, B, r, sigma, T,
                                   kind="up_and_in", option="call")
        c_uo = barrier_closed_form(S0, K, B, r, sigma, T,
                                   kind="up_and_out", option="call")
        c_van = bs_call(S0, K, r, sigma, T)
        assert (c_ui + c_uo) == pytest.approx(c_van, abs=1e-10)

    def test_in_out_parity_mid_barrier(self):
        """In-out parity at B=130 — a stronger test (trap #6)."""
        B = 130.0
        c_ui = barrier_closed_form(S0, K, B, r, sigma, T,
                                   kind="up_and_in", option="call")
        c_uo = barrier_closed_form(S0, K, B, r, sigma, T,
                                   kind="up_and_out", option="call")
        c_van = bs_call(S0, K, r, sigma, T)
        assert (c_ui + c_uo) == pytest.approx(c_van, abs=1e-10)

    def test_in_out_parity_deep_barrier(self):
        """In-out parity at B=140 — even stronger test."""
        B = 140.0
        c_ui = barrier_closed_form(S0, K, B, r, sigma, T,
                                   kind="up_and_in", option="call")
        c_uo = barrier_closed_form(S0, K, B, r, sigma, T,
                                   kind="up_and_out", option="call")
        c_van = bs_call(S0, K, r, sigma, T)
        assert (c_ui + c_uo) == pytest.approx(c_van, abs=1e-10)


# ============================================================================
# BGK correction sanity
# ============================================================================

class TestBGKCorrection:
    """The BGK-corrected value should be between the continuous value and
    the vanilla call (for an up-and-in call with B near S0)."""

    def test_bgk_paper_barrier(self):
        """BGK-corrected continuous value ≈ 7.093 (WORKPLAN §2)."""
        B = 110.6772
        price = barrier_closed_form_bgk(S0, K, B, r, sigma, T,
                                        n_steps=N_STEPS,
                                        kind="up_and_in", option="call")
        # The BGK shifts the barrier up, so the UIC price drops slightly
        continuous = barrier_closed_form(S0, K, B, r, sigma, T,
                                         kind="up_and_in", option="call")
        assert price < continuous, (
            f"BGK price {price} should be < continuous {continuous} "
            "for an up-and-in (barrier shifted outward reduces knock-in probability)"
        )
        # §2 says ≈7.093
        assert price == pytest.approx(7.093, abs=0.01), f"Got {price}"

    def test_bgk_converges_to_continuous(self):
        """As n_steps → ∞, BGK should approach the continuous price."""
        B = 110.6772
        continuous = barrier_closed_form(S0, K, B, r, sigma, T,
                                         kind="up_and_in", option="call")
        bgk_fine = barrier_closed_form_bgk(S0, K, B, r, sigma, T,
                                           n_steps=100_000,
                                           kind="up_and_in", option="call")
        assert bgk_fine == pytest.approx(continuous, abs=1e-3)


# ============================================================================
# Discrete vs continuous geometric Asian (trap #1)
# ============================================================================

class TestDiscreteVsContinuousAsian:
    """The discrete Kemna–Vorst variant should differ from the continuous
    one and converge as m → ∞."""

    def test_discrete_differs_from_continuous(self):
        c_cont = geometric_asian_closed_form(S0, K, r, sigma, T, option="call")
        c_disc = geometric_asian_closed_form(S0, K, r, sigma, T, option="call", m=252)
        # They should differ (the discrete version is slightly different)
        assert c_disc != pytest.approx(c_cont, abs=1e-6)
        # But be close
        assert c_disc == pytest.approx(c_cont, abs=0.05)

    def test_discrete_converges_to_continuous(self):
        c_cont = geometric_asian_closed_form(S0, K, r, sigma, T, option="call")
        c_disc = geometric_asian_closed_form(S0, K, r, sigma, T, option="call",
                                             m=100_000)
        assert c_disc == pytest.approx(c_cont, abs=1e-3)
