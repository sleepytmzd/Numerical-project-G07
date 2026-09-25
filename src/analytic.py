import math
from typing import Optional
import numpy as np
from scipy.stats import norm

# Black–Scholes European options
def bs_call(S0: float, K: float, r: float, sigma: float, T: float,
            q: float = 0.0) -> float:
    """Black–Scholes European call price (continuous dividend yield q)."""
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put(S0: float, K: float, r: float, sigma: float, T: float,
           q: float = 0.0) -> float:
    """Black–Scholes European put price (continuous dividend yield q)."""
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * math.exp(-q * T) * norm.cdf(-d1)

def _barrier_mu(r, q, sigma):
    """mu = (b - 0.5*sigma^2) / sigma^2 where b = r - q."""
    return (r - q - 0.5 * sigma**2) / sigma**2

def _A(S0, K, r, q, sigma, T, phi):
    """Component A: standard BS-like term using ln(S/K)."""
    sqT = sigma * math.sqrt(T)
    mu = _barrier_mu(r, q, sigma)
    x1 = math.log(S0 / K) / sqT + (1 + mu) * sqT
    return phi * (S0 * math.exp(-q * T) * norm.cdf(phi * x1)
                  - K * math.exp(-r * T) * norm.cdf(phi * (x1 - sqT)))

def _B(S0, K, B, r, q, sigma, T, phi):
    """Component B: BS-like term truncated at barrier, using ln(S/H)."""
    sqT = sigma * math.sqrt(T)
    mu = _barrier_mu(r, q, sigma)
    x2 = math.log(S0 / B) / sqT + (1 + mu) * sqT
    return phi * (S0 * math.exp(-q * T) * norm.cdf(phi * x2)
                  - K * math.exp(-r * T) * norm.cdf(phi * (x2 - sqT)))

def _C(S0, K, B, r, q, sigma, T, eta, phi):
    """Component C: reflected term using y1 = ln(H^2/(S*K))."""
    sqT = sigma * math.sqrt(T)
    mu = _barrier_mu(r, q, sigma)
    HS = B / S0
    powHS0 = HS ** (2 * mu)
    powHS1 = powHS0 * HS * HS

    y1 = math.log(B * HS / K) / sqT + (1 + mu) * sqT  # = ln(H^2/(S*K))/sqT + muSigma
    N1 = norm.cdf(eta * y1)
    N2 = norm.cdf(eta * (y1 - sqT))

    return phi * (S0 * math.exp(-q * T) * (0.0 if N1 == 0 else powHS1 * N1)
                  - K * math.exp(-r * T) * (0.0 if N2 == 0 else powHS0 * N2))


def _D(S0, K, B, r, q, sigma, T, eta, phi):
    """Component D: reflected term using y2 = ln(H/S)."""
    sqT = sigma * math.sqrt(T)
    mu = _barrier_mu(r, q, sigma)
    HS = B / S0
    powHS0 = HS ** (2 * mu)
    powHS1 = powHS0 * HS * HS

    y2 = math.log(B / S0) / sqT + (1 + mu) * sqT  # = ln(H/S)/sqT + muSigma
    N1 = norm.cdf(eta * y2)
    N2 = norm.cdf(eta * (y2 - sqT))

    return phi * (S0 * math.exp(-q * T) * (0.0 if N1 == 0 else powHS1 * N1)
                  - K * math.exp(-r * T) * (0.0 if N2 == 0 else powHS0 * N2))

def barrier_closed_form(
    S0: float,
    K: float,
    B: float,
    r: float,
    sigma: float,
    T: float,
    kind: str = "up_and_in",
    option: str = "call",
    q: float = 0.0,
) -> float:
    """Reiner–Rubinstein barrier option price (continuous monitoring).
    Follows the exact QuantLib component decomposition. 
    Parameters:
    kind : str
        "up_and_in" or "up_and_out".
    option : str
        "call" or "put".
    Returns:
    float: The analytic option price.
    """
    if S0 >= B:
        # Already above the barrier
        if kind == "up_and_in":
            return bs_call(S0, K, r, sigma, T, q) if option == "call" else bs_put(S0, K, r, sigma, T, q)
        else:
            return 0.0  # up-and-out is worthless

    if option == "call":
        if kind == "up_and_in":
            if K >= B:
                return _A(S0, K, r, q, sigma, T, phi=1)
            else:
                return (_B(S0, K, B, r, q, sigma, T, phi=1)
                        - _C(S0, K, B, r, q, sigma, T, eta=-1, phi=1)
                        + _D(S0, K, B, r, q, sigma, T, eta=-1, phi=1))
        elif kind == "up_and_out":
            if K >= B:
                return 0.0  # worthless: can't be ITM without crossing barrier
            else:
                return (_A(S0, K, r, q, sigma, T, phi=1)
                        - _B(S0, K, B, r, q, sigma, T, phi=1)
                        + _C(S0, K, B, r, q, sigma, T, eta=-1, phi=1)
                        - _D(S0, K, B, r, q, sigma, T, eta=-1, phi=1))
    elif option == "put":
        if kind == "up_and_in":
            if K >= B:
                return (_A(S0, K, r, q, sigma, T, phi=-1)
                        - _B(S0, K, B, r, q, sigma, T, phi=-1)
                        + _D(S0, K, B, r, q, sigma, T, eta=-1, phi=-1))
            else:
                return _C(S0, K, B, r, q, sigma, T, eta=-1, phi=-1)
        elif kind == "up_and_out":
            if K >= B:
                return (_B(S0, K, B, r, q, sigma, T, phi=-1)
                        - _D(S0, K, B, r, q, sigma, T, eta=-1, phi=-1))
            else:
                return (_A(S0, K, r, q, sigma, T, phi=-1)
                        - _C(S0, K, B, r, q, sigma, T, eta=-1, phi=-1))

    raise ValueError(f"Unsupported barrier kind={kind!r}, option={option!r}")

def barrier_closed_form_bgk(
    S0: float,
    K: float,
    B: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    kind: str = "up_and_in",
    option: str = "call",
    q: float = 0.0,
) -> float:
    """Barrier price with BGK continuity correction for discrete monitoring.
    Broadie, Glasserman & Kou (1997): shift the barrier by
        B_adj = B * exp(+beta * sigma * sqrt(dt))   for up-barriers
        B_adj = B * exp(-beta * sigma * sqrt(dt))   for down-barriers
    where beta ≈ 0.5826 = -zeta(1/2) / sqrt(2*pi).
    Then use the continuous Reiner–Rubinstein formula with B_adj.
    """
    beta = 0.5826
    dt = T / n_steps

    # For up-barriers, shift outward (upward): + sign
    B_adj = B * math.exp(beta * sigma * math.sqrt(dt))

    return barrier_closed_form(S0, K, B_adj, r, sigma, T,
                               kind=kind, option=option, q=q)

def geometric_asian_closed_form(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option: str = "call",
    m: Optional[int] = None,
    q: float = 0.0,
) -> float:
    """Kemna–Vorst geometric-average Asian option price.
    Parameters:
    m : int or None
        Number of discrete averaging points.  ``None`` → continuous averaging.
    option : str
        "call" or "put".
    Returns:
    float: The analytic option price.
    """
    if m is None:
        # Continuous averaging
        sigma_G = sigma / math.sqrt(3)
        b = 0.5 * (r - q - sigma**2 / 6)
    else:
        # Discrete averaging over m equally-spaced points
        sigma_G_sq = sigma**2 * (m + 1) * (2 * m + 1) / (6 * m**2)
        sigma_G = math.sqrt(sigma_G_sq)
        b = 0.5 * sigma_G_sq + (r - q - 0.5 * sigma**2) * (m + 1) / (2 * m)

    sqT = sigma_G * math.sqrt(T)
    d1 = (math.log(S0 / K) + (b + 0.5 * sigma_G**2) * T) / sqT
    d2 = d1 - sqT

    disc = math.exp(-r * T)
    fwd = S0 * math.exp(b * T)

    if option == "call":
        return disc * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    elif option == "put":
        return disc * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    else:
        raise ValueError(f"Unsupported option type: {option!r}")
