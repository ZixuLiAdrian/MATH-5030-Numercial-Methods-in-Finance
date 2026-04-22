"""Plain and antithetic Monte Carlo estimators for arithmetic Asian options."""
from __future__ import annotations

import time

import numpy as np

from .models import AsianOptionParams
from .paths import PathMethod, build_paths, payoff_from_paths
from .utils import discount_factor


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility with earlier imports.
# ---------------------------------------------------------------------------
def _paths_from_z(params: AsianOptionParams, z: np.ndarray) -> np.ndarray:
    """Deprecated alias preserved for older callers/tests."""
    return build_paths(params, z, method="incremental")


def _payoff_from_paths(paths: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """Deprecated alias preserved for older callers/tests."""
    return payoff_from_paths(paths, K, option_type)


# ---------------------------------------------------------------------------
# Estimators.
# ---------------------------------------------------------------------------
def standard_mc_price(
    params: AsianOptionParams,
    n_paths: int,
    seed: int = 123,
    path_method: PathMethod = "incremental",
) -> dict:
    """Plain Monte Carlo estimator for an arithmetic-average Asian option.

    Returns a dict with the price, the Monte Carlo standard error, elapsed
    wall-clock time, and the number of paths used.
    """
    params.validate()
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, params.N))
    paths = build_paths(params, z, method=path_method)
    pv = discount_factor(params) * payoff_from_paths(paths, params.K, params.option_type)
    elapsed = time.perf_counter() - t0
    return {
        "price": float(pv.mean()),
        "std_err": float(pv.std(ddof=1) / np.sqrt(n_paths)),
        "runtime_s": elapsed,
        "n_paths": n_paths,
    }


def antithetic_mc_price(
    params: AsianOptionParams,
    n_paths: int,
    seed: int = 123,
    path_method: PathMethod = "incremental",
) -> dict:
    """Antithetic-variates Monte Carlo.

    Uses ``n_paths // 2`` independent normal draws ``z`` plus their negation
    ``-z``, and averages the pair payoffs. The reported ``n_paths`` is the
    total number of GBM paths simulated (``2 * half``), which is the fair unit
    for CPU comparisons with plain MC.
    """
    params.validate()
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2 for antithetic variates.")

    half = n_paths // 2
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((half, params.N))
    paths_pos = build_paths(params, z, method=path_method)
    paths_neg = build_paths(params, -z, method=path_method)
    df = discount_factor(params)
    pv_pos = df * payoff_from_paths(paths_pos, params.K, params.option_type)
    pv_neg = df * payoff_from_paths(paths_neg, params.K, params.option_type)
    # The antithetic estimator's iid unit is the pair average.
    pv_pair = 0.5 * (pv_pos + pv_neg)
    elapsed = time.perf_counter() - t0
    return {
        "price": float(pv_pair.mean()),
        "std_err": float(pv_pair.std(ddof=1) / np.sqrt(half)),
        "runtime_s": elapsed,
        "n_paths": 2 * half,
    }
