"""Efficiency comparison: RMSE and standard error vs CPU time.

For a single canonical parameter set we sweep the Monte Carlo budget from a
few thousand paths up to ~500k, run every estimator at each budget, and
measure:

* ``rmse`` -- root-mean-square error against a high-precision reference,
  estimated from ``N_REPS`` independent repetitions with disjoint seeds.
* ``mean_runtime_s`` -- wall-clock cost per run.

Results are written to ``results/tables/efficiency.csv`` and plotted on a
log-log axis as ``results/figures/efficiency_rmse_vs_time.png``.
"""
from __future__ import annotations

import time
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # headless backend for unattended execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    FIGURES_DIR,
    TABLES_DIR,
    AsianOptionParams,
    high_precision_reference,
)
from asian_option_pricer import (
    antithetic_cv_price,
    antithetic_mc_price,
    control_variate_price,
    rqmc_sobol_price,
    sobol_qmc_price,
    standard_mc_price,
)


PARAMS = AsianOptionParams(S0=100.0, K=100.0, r=0.05, sigma=0.30, T=1.0, N=50)
N_PATH_GRID = [2_048, 8_192, 32_768, 131_072, 524_288]
N_REPS = 12
SEED_BASE = 20240420


Method = tuple[str, Callable[[AsianOptionParams, int, int], dict]]


def _methods() -> list[Method]:
    return [
        ("standard_mc", lambda p, n, s: standard_mc_price(p, n, seed=s)),
        ("antithetic", lambda p, n, s: antithetic_mc_price(p, n, seed=s)),
        ("control_variate", lambda p, n, s: control_variate_price(p, n, seed=s)),
        ("antithetic_cv", lambda p, n, s: antithetic_cv_price(p, n, seed=s)),
        (
            "sobol_incremental",
            lambda p, n, s: sobol_qmc_price(p, n, seed=s, path_method="incremental"),
        ),
        (
            "sobol_bridge",
            lambda p, n, s: sobol_qmc_price(p, n, seed=s, path_method="brownian_bridge"),
        ),
        (
            "rqmc_bridge",
            lambda p, n, s: rqmc_sobol_price(
                p, n, seed=s, n_replications=8, path_method="brownian_bridge"
            ),
        ),
    ]


def run_grid(reference: float) -> pd.DataFrame:
    rows = []
    for name, fn in _methods():
        for n in N_PATH_GRID:
            prices = np.empty(N_REPS)
            runtimes = np.empty(N_REPS)
            for r_idx in range(N_REPS):
                out = fn(PARAMS, n, SEED_BASE + 1_000 * r_idx)
                prices[r_idx] = out["price"]
                runtimes[r_idx] = out["runtime_s"]
            errors = prices - reference
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            bias = float(np.mean(errors))
            std = float(np.std(prices, ddof=1))
            rows.append(
                {
                    "method": name,
                    "n_paths_requested": n,
                    "rmse": rmse,
                    "bias": bias,
                    "empirical_std": std,
                    "mean_runtime_s": float(np.mean(runtimes)),
                    "mean_price": float(np.mean(prices)),
                }
            )
            print(
                f"  {name:<20s} n={n:>7d}  rmse={rmse: .5f}  "
                f"std={std:.5f}  runtime={np.mean(runtimes)*1e3: .1f}ms"
            )
    return pd.DataFrame(rows)


_MARKERS = {
    "standard_mc": ("o", "C0"),
    "antithetic": ("s", "C1"),
    "control_variate": ("D", "C2"),
    "antithetic_cv": ("v", "C3"),
    "sobol_incremental": ("^", "C4"),
    "sobol_bridge": ("P", "C5"),
    "rqmc_bridge": ("X", "C6"),
}


def plot_efficiency(df: pd.DataFrame, out_path) -> None:
    fig, (ax_n, ax_t) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for name, sub in df.groupby("method"):
        sub = sub.sort_values("n_paths_requested")
        marker, color = _MARKERS.get(name, ("o", None))
        ax_n.plot(
            sub["n_paths_requested"].to_numpy(),
            sub["rmse"].to_numpy(),
            marker=marker,
            color=color,
            label=name,
            linewidth=1.4,
        )
        ax_t.plot(
            sub["mean_runtime_s"].to_numpy(),
            sub["rmse"].to_numpy(),
            marker=marker,
            color=color,
            label=name,
            linewidth=1.4,
        )

    for ax in (ax_n, ax_t):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)

    ax_n.set_xlabel("Paths requested (log)")
    ax_n.set_ylabel("RMSE vs reference (log)")
    ax_n.set_title(
        f"Convergence  (S0={PARAMS.S0}, K={PARAMS.K}, sigma={PARAMS.sigma}, "
        f"N={PARAMS.N}, reps={N_REPS})"
    )
    ax_t.set_xlabel("Mean runtime per call [s] (log)")
    ax_t.set_title("Efficiency (RMSE vs CPU time)")
    ax_t.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    print("Computing high-precision reference...")
    ref_price, ref_se = high_precision_reference(PARAMS)
    print(f"Reference = {ref_price:.6f}  (se = {ref_se:.2e})")

    print("\nRunning efficiency grid...")
    df = run_grid(ref_price)
    df["reference_price"] = ref_price
    df["reference_std_err"] = ref_se
    table_path = TABLES_DIR / "efficiency.csv"
    df.to_csv(table_path, index=False)

    fig_path = FIGURES_DIR / "efficiency_rmse_vs_time.png"
    plot_efficiency(df, fig_path)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  table   -> {table_path}")
    print(f"  figure  -> {fig_path}")


if __name__ == "__main__":
    main()
