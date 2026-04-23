"""Shared helpers for the experiment scripts.

Each script in this directory stays self-contained and runnable via
``python experiments/<name>.py``. Centralising a few utilities here keeps
boilerplate (path resolution, reference-price computation) out of the
individual scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from asian_option_pricer import (  # noqa: E402
    AsianOptionParams,
    antithetic_cv_price,
    rqmc_sobol_price,
)

RESULTS_DIR = _ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def high_precision_reference(
    params: AsianOptionParams,
    n_paths: int = 2_097_152,
    n_replications: int = 32,
    seed: int = 20240420,
) -> tuple[float, float]:
    """Return ``(price, std_err)`` for a tight reference price on ``params``.

    Uses randomised-QMC with Brownian-bridge construction, which gives the
    lowest variance per CPU-second of any estimator in this package, together
    with an honest cross-replication standard error. A second antithetic-CV
    run with a disjoint seed is averaged in to further reduce variance and
    protect against a single-method systematic bias.
    """
    rqmc = rqmc_sobol_price(
        params,
        n_paths=n_paths,
        seed=seed,
        n_replications=n_replications,
        path_method="brownian_bridge",
    )
    acv = antithetic_cv_price(params, n_paths=n_paths // 2, seed=seed + 1)
    variances = [rqmc["std_err"] ** 2, acv["std_err"] ** 2]
    weights = [1.0 / v for v in variances]
    wsum = sum(weights)
    price = (rqmc["price"] * weights[0] + acv["price"] * weights[1]) / wsum
    std_err = (1.0 / wsum) ** 0.5
    return float(price), float(std_err)
