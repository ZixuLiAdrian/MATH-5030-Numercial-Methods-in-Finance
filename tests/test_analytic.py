"""Tests for the closed-form / semi-analytic prices."""
from __future__ import annotations

import numpy as np
import pytest

from asian_option_pricer import (
    AsianOptionParams,
    build_paths,
    geometric_asian_call_price,
    geometric_payoff_from_paths,
    levy_approx_call_price,
)
from asian_option_pricer.utils import discount_factor


def test_prices_are_non_negative():
    params = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=52)
    assert geometric_asian_call_price(params) >= 0.0
    assert levy_approx_call_price(params) >= 0.0


def test_geometric_closed_form_matches_mc():
    """Closed-form discrete geometric Asian should match antithetic MC to
    well within three standard errors."""
    params = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.20, T=1.0, N=52)
    closed = geometric_asian_call_price(params)

    rng = np.random.default_rng(2024)
    n_pairs = 400_000
    z = rng.standard_normal((n_pairs, params.N))
    df = discount_factor(params)
    pv_pos = df * geometric_payoff_from_paths(
        build_paths(params, z, method="incremental"), params.K, params.option_type
    )
    pv_neg = df * geometric_payoff_from_paths(
        build_paths(params, -z, method="incremental"), params.K, params.option_type
    )
    pair = 0.5 * (pv_pos + pv_neg)
    mc_price = pair.mean()
    mc_se = pair.std(ddof=1) / np.sqrt(n_pairs)
    assert abs(closed - mc_price) < 3.0 * mc_se, (
        f"closed={closed}, mc={mc_price} +/- {mc_se}"
    )


def test_geometric_single_monitor_reduces_to_black_scholes():
    """With ``N = 1`` the geometric average reduces to ``S_T``, so our price
    must equal the Black-Scholes call price."""
    from scipy.stats import norm

    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    params = AsianOptionParams(S0=S0, K=K, r=r, sigma=sigma, T=T, N=1)
    geom = geometric_asian_call_price(params)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    assert abs(geom - bs) < 1e-10


def test_geometric_zero_volatility_limit():
    """At ``sigma = 0`` the discrete geometric price is deterministic."""
    params = AsianOptionParams(S0=100.0, K=100.0, r=0.05, sigma=0.0, T=1.0, N=52)
    price = geometric_asian_call_price(params)
    t_bar = params.T * (params.N + 1) / (2 * params.N)
    forward = params.S0 * np.exp(params.r * t_bar)
    expected = np.exp(-params.r * params.T) * max(forward - params.K, 0.0)
    assert abs(price - expected) < 1e-10


def test_levy_approx_close_to_mc_at_moderate_vol():
    """Levy approximation is known to be accurate at sigma <= 0.3. Cross-check
    against Monte Carlo within a generous tolerance."""
    params = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=52)
    levy = levy_approx_call_price(params)
    # Tight MC via antithetic + moderate sample size
    rng = np.random.default_rng(11)
    n_pairs = 200_000
    z = rng.standard_normal((n_pairs, params.N))
    paths_p = build_paths(params, z, method="incremental")
    paths_n = build_paths(params, -z, method="incremental")
    payoff = 0.5 * (paths_p.mean(1) + paths_n.mean(1))  # paired average payoff pre-max
    # actual payoffs (max with 0) averaged
    p_pos = np.maximum(paths_p.mean(1) - params.K, 0.0)
    p_neg = np.maximum(paths_n.mean(1) - params.K, 0.0)
    df = discount_factor(params)
    pair = 0.5 * df * (p_pos + p_neg)
    mc_price = pair.mean()
    mc_se = pair.std(ddof=1) / np.sqrt(n_pairs)
    # Levy is an *approximation*; the relative error at sigma=0.2 is usually
    # below 0.5%, which is well inside 100 bps.
    assert abs(levy - mc_price) / mc_price < 0.01
    # And it sits near the MC value for reporting sake.
    assert abs(levy - mc_price) < 10 * mc_se + 0.05


def test_put_not_implemented_is_explicit():
    params = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=52, option_type="put")
    with pytest.raises(NotImplementedError):
        geometric_asian_call_price(params)
    with pytest.raises(NotImplementedError):
        levy_approx_call_price(params)
