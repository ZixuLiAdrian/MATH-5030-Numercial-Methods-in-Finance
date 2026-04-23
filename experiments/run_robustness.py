"""Robustness experiments for the Asian-option estimators.

Three independent checks are produced:

1. ``robustness_random.csv``
   Sample ``N_SCENARIOS`` random parameter sets over reasonable ranges and
   run every estimator. Flag NaNs, negative prices, prices violating the
   no-arbitrage bounds, or results that differ from the Levy approximation
   by an implausible margin.

2. ``robustness_monotonicity.csv``
   Sweep the strike ``K`` and the volatility ``sigma`` over dense grids.
   Record whether the estimated call price is monotonically decreasing in
   ``K`` and monotonically increasing in ``sigma`` -- both exact properties
   of the true price under GBM.

3. ``robustness_edge_cases.csv``
   Limit behaviour: ``sigma -> 0`` (payoff becomes deterministic), ``K``
   far out-of-the-money, and very large ``N`` (dense monitoring).

Summary statistics are printed and full tables are written under
``results/tables/``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _common import (
    TABLES_DIR,
    AsianOptionParams,
)
from asian_option_pricer import (
    antithetic_cv_price,
    antithetic_mc_price,
    control_variate_price,
    geometric_asian_call_price,
    levy_approx_call_price,
    rqmc_sobol_price,
    sobol_qmc_price,
    standard_mc_price,
)


METHODS = {
    "standard_mc": lambda p, n, s: standard_mc_price(p, n, seed=s),
    "antithetic": lambda p, n, s: antithetic_mc_price(p, n, seed=s),
    "control_variate": lambda p, n, s: control_variate_price(p, n, seed=s),
    "antithetic_cv": lambda p, n, s: antithetic_cv_price(p, n, seed=s),
    "sobol_bridge": lambda p, n, s: sobol_qmc_price(
        p, n, seed=s, path_method="brownian_bridge"
    ),
    "rqmc_bridge": lambda p, n, s: rqmc_sobol_price(
        p, n, seed=s, n_replications=8, path_method="brownian_bridge"
    ),
}

N_SCENARIOS = 200
N_PATHS_SCENARIO = 32_768
SEED_BASE = 20240420


def _no_arb_bounds(params: AsianOptionParams) -> tuple[float, float]:
    return 0.0, params.S0


def _run_all(params: AsianOptionParams, n_paths: int, seed: int) -> dict:
    results: dict[str, dict] = {}
    for name, fn in METHODS.items():
        try:
            results[name] = fn(params, n_paths, seed)
        except Exception as exc:
            results[name] = {"price": float("nan"), "error": str(exc)}
    return results


def random_scenarios() -> pd.DataFrame:
    """Random parameter sweep."""
    rng = np.random.default_rng(SEED_BASE)
    rows = []
    for i in range(N_SCENARIOS):
        sigma = float(rng.uniform(0.05, 0.80))
        T = float(rng.uniform(0.25, 3.0))
        moneyness = float(rng.uniform(0.7, 1.3))
        K = 100.0 * moneyness
        r = float(rng.uniform(0.0, 0.10))
        N = int(rng.choice([12, 26, 52, 100, 250]))
        params = AsianOptionParams(S0=100.0, K=K, r=r, sigma=sigma, T=T, N=N)

        runs = _run_all(params, N_PATHS_SCENARIO, seed=SEED_BASE + i)
        levy = levy_approx_call_price(params)
        lo, hi = _no_arb_bounds(params)

        for name, out in runs.items():
            price = out.get("price", float("nan"))
            is_nan = not np.isfinite(price)
            out_of_bounds = (not is_nan) and (price < lo - 1e-8 or price > hi + 1e-8)
            levy_diff_bps = (
                1.0e4 * (price - levy) / max(levy, 1e-8) if not is_nan else float("nan")
            )
            rows.append(
                {
                    "scenario": i,
                    "sigma": sigma,
                    "T": T,
                    "K": K,
                    "r": r,
                    "N": N,
                    "method": name,
                    "price": price,
                    "std_err": out.get("std_err", float("nan")),
                    "runtime_s": out.get("runtime_s", float("nan")),
                    "is_nan": is_nan,
                    "out_of_bounds": out_of_bounds,
                    "levy_ref": levy,
                    "diff_from_levy_bps": levy_diff_bps,
                }
            )
    return pd.DataFrame(rows)


def monotonicity_sweep() -> pd.DataFrame:
    """Check strike- and volatility-monotonicity of the call price."""
    base = dict(S0=100.0, r=0.05, T=1.0, N=52)
    methods = ["control_variate", "antithetic_cv", "rqmc_bridge"]
    rows = []

    K_grid = np.linspace(70, 130, 31)
    for K in K_grid:
        params = AsianOptionParams(sigma=0.2, K=float(K), **base)
        runs = {m: METHODS[m](params, N_PATHS_SCENARIO, SEED_BASE) for m in methods}
        row = {"sweep": "strike", "x_name": "K", "x_value": float(K)}
        for m, out in runs.items():
            row[f"{m}_price"] = out["price"]
        rows.append(row)

    sigma_grid = np.linspace(0.05, 0.80, 31)
    for sigma in sigma_grid:
        params = AsianOptionParams(sigma=float(sigma), K=100.0, **base)
        runs = {m: METHODS[m](params, N_PATHS_SCENARIO, SEED_BASE) for m in methods}
        row = {"sweep": "sigma", "x_name": "sigma", "x_value": float(sigma)}
        for m, out in runs.items():
            row[f"{m}_price"] = out["price"]
        rows.append(row)

    return pd.DataFrame(rows)


def edge_cases() -> pd.DataFrame:
    """Limit-case sanity checks."""
    rows = []

    for sigma in [1e-6, 1e-4, 1e-2]:
        params = AsianOptionParams(S0=100.0, K=100.0, r=0.05, sigma=sigma, T=1.0, N=52)
        runs = _run_all(params, 4096, seed=SEED_BASE)
        row = {"case": f"low_sigma={sigma}"}
        for m, o in runs.items():
            row[m] = o["price"]
        row["levy"] = levy_approx_call_price(params)
        row["geometric_closed"] = geometric_asian_call_price(params)
        rows.append(row)

    for K in [200.0, 400.0]:
        params = AsianOptionParams(S0=100.0, K=K, r=0.05, sigma=0.2, T=1.0, N=52)
        runs = _run_all(params, N_PATHS_SCENARIO, seed=SEED_BASE)
        row = {"case": f"deep_otm_K={K}"}
        for m, o in runs.items():
            row[m] = o["price"]
        row["levy"] = levy_approx_call_price(params)
        rows.append(row)

    params = AsianOptionParams(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=500)
    runs = _run_all(params, N_PATHS_SCENARIO, seed=SEED_BASE)
    row = {"case": "dense_N=500"}
    for m, o in runs.items():
        row[m] = o["price"]
    row["levy"] = levy_approx_call_price(params)
    rows.append(row)

    return pd.DataFrame(rows)


def _summarise_random(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("method").agg(
        scenarios=("scenario", "nunique"),
        n_nan=("is_nan", "sum"),
        n_out_of_bounds=("out_of_bounds", "sum"),
        median_abs_diff_bps=("diff_from_levy_bps", lambda s: float(np.nanmedian(np.abs(s)))),
        p95_abs_diff_bps=("diff_from_levy_bps", lambda s: float(np.nanpercentile(np.abs(s), 95))),
        mean_runtime_s=("runtime_s", "mean"),
        mean_std_err=("std_err", "mean"),
    )
    return grouped


def _monotonicity_violations(df: pd.DataFrame) -> pd.DataFrame:
    methods = [c.replace("_price", "") for c in df.columns if c.endswith("_price")]
    out = []
    for sweep, sub in df.groupby("sweep"):
        sub = sub.sort_values("x_value")
        expected = "decreasing" if sweep == "strike" else "increasing"
        for m in methods:
            values = sub[f"{m}_price"].to_numpy()
            diffs = np.diff(values)
            if expected == "decreasing":
                violations = int(np.sum(diffs > 0))
                max_violation = float(diffs.max())
            else:
                violations = int(np.sum(diffs < 0))
                max_violation = float(-diffs.min())
            out.append(
                {
                    "sweep": sweep,
                    "method": m,
                    "expected": expected,
                    "violations": violations,
                    "max_violation_magnitude": max_violation,
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 78)
    print(f"RANDOM SCENARIO SWEEP ({N_SCENARIOS} scenarios, {N_PATHS_SCENARIO} paths each)")
    print("=" * 78)
    rnd = random_scenarios()
    rnd.to_csv(TABLES_DIR / "robustness_random.csv", index=False)
    summary = _summarise_random(rnd)
    print(summary.to_string(float_format=lambda x: f"{x: .4f}"))
    total_nans = int(rnd["is_nan"].sum())
    total_oob = int(rnd["out_of_bounds"].sum())
    print(f"\nTotal NaNs across methods: {total_nans}")
    print(f"Total out-of-bounds prices: {total_oob}")

    print()
    print("=" * 78)
    print("MONOTONICITY SWEEPS (strike: expected decreasing; sigma: expected increasing)")
    print("=" * 78)
    mono = monotonicity_sweep()
    mono.to_csv(TABLES_DIR / "robustness_monotonicity.csv", index=False)
    viol = _monotonicity_violations(mono)
    print(viol.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    print()
    print("=" * 78)
    print("EDGE CASES")
    print("=" * 78)
    edge = edge_cases()
    edge.to_csv(TABLES_DIR / "robustness_edge_cases.csv", index=False)
    print(edge.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    elapsed = time.perf_counter() - t0
    print(f"\nRobustness sweep complete in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
