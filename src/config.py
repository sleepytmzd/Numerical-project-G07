from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Scenario:
    """A frozen option-pricing scenario.
    Parameters:
    name : str
        Unique identifier used in filenames and log rows.
    S0 : float
        Initial spot price.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility (annualized).
    T : float
        Time to maturity in years.
    K : float
        Strike price.
    B : float or None
        Barrier level (None for non-barrier options).
    option_type : str
        "call" or "put".
    exotic_type : str
        "barrier" | "geometric_asian" | "arithmetic_asian".
    barrier_kind : str or None
        "up_and_in" | "up_and_out" | None.
    avg_start_idx : int
        For Asian options, the index at which averaging begins (1 = from the
        first simulated point, i.e. full averaging). For the last-30-day
        variant, this is set so that only the final 30 time steps are averaged.
    """
    name: str
    S0: float
    r: float
    sigma: float
    T: float
    K: float
    B: Optional[float] = None
    option_type: str = "call"
    exotic_type: str = "barrier"
    barrier_kind: Optional[str] = "up_and_in"
    avg_start_idx: int = 1

# Experiment constants 
N_GRID = [256, 1024, 4096, 16384, 65536]   # powers of 2 (Sobol requirement)
N_STEPS = 252                                # daily monitoring, 1 year
R = 20                                       # replications per (method, N) cell
BASE_SEED = 402                              # np.random.SeedSequence(402)

_COMMON = dict(S0=100.0, r=0.03, sigma=0.2, T=1.0)

SCENARIOS = {
    # Barrier family 
    "paper_barrier": Scenario(
        name="paper_barrier",
        K=105.0,
        B=110.6772,
        option_type="call",
        exotic_type="barrier",
        barrier_kind="up_and_in",
        **_COMMON,
    ),
    "mid_barrier": Scenario(
        name="mid_barrier",
        K=105.0,
        B=130.0,
        option_type="call",
        exotic_type="barrier",
        barrier_kind="up_and_in",
        **_COMMON,
    ),
    "deep_barrier": Scenario(
        name="deep_barrier",
        K=105.0,
        B=140.0,
        option_type="call",
        exotic_type="barrier",
        barrier_kind="up_and_in",
        **_COMMON,
    ),
    # Asian family (geometric) 
    "paper_asian_geo": Scenario(
        name="paper_asian_geo",
        K=105.0,
        option_type="call",      
        exotic_type="geometric_asian",
        barrier_kind=None,
        B=None,
        **_COMMON,
    ),
    "asian_30d": Scenario(
        name="asian_30d",
        K=105.0,
        option_type="call",
        exotic_type="geometric_asian",
        barrier_kind=None,
        B=None,
        avg_start_idx=N_STEPS - 30 + 1,   # average over the final 30 trading days
        **_COMMON,
    ),
    # Asian family (arithmetic) 
    "asian_arith": Scenario(
        name="asian_arith",
        K=105.0,
        option_type="call",
        exotic_type="arithmetic_asian",
        barrier_kind=None,
        B=None,
        **_COMMON,
    ),
}


def get_scenario(name: str) -> Scenario:
    """Return a frozen scenario by name; raise KeyError if not found."""
    return SCENARIOS[name]
