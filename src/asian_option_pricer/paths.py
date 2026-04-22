"""Path construction utilities for GBM under equally-spaced monitoring.

Two constructions from N iid standard normals to a discrete Brownian path are
supported:

* ``incremental`` (a.k.a. forward / random-walk construction) builds each
  Brownian increment independently. This is the default for plain Monte Carlo
  because all coordinates contribute equal variance.

* ``brownian_bridge`` places the terminal value on the first coordinate and
  then fills in the remaining points by recursive bisection. Early coordinates
  therefore capture the coarse shape of the path and carry most of the
  variance. With low-discrepancy sequences (Sobol), whose first coordinates
  are the best distributed, this can give dramatic variance reduction on
  path-dependent payoffs (Moro 1995, Caflisch-Morokoff-Owen 1997).

The two helpers return log-price paths of shape ``(n_paths, N)`` so callers
can share downstream payoff code regardless of the construction.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from .models import AsianOptionParams
from .utils import monitoring_times


PathMethod = str  # "incremental" | "brownian_bridge"


@lru_cache(maxsize=32)
def brownian_bridge_matrix(N: int, T: float) -> np.ndarray:
    """Return a matrix ``B`` of shape ``(N, N)`` with ``W = B @ z`` where
    ``z`` is an iid standard-normal vector and ``W`` is the Brownian motion at
    the ``N`` equally-spaced times ``T/N, 2T/N, ..., T`` constructed in
    bridge order.

    The construction follows Glasserman (2004), Monte Carlo Methods in
    Financial Engineering, section 3.1. Coordinate ``z[0]`` sets the terminal
    point; subsequent coordinates fill mid-points of already-determined
    intervals by breadth-first bisection, so the earliest ``z``-coordinates
    control the coarsest scales of the path.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")

    dt = T / N
    # t_ext[0] = 0, t_ext[k] = k * dt for k = 1..N.
    t_ext = np.arange(N + 1) * dt

    B = np.zeros((N, N))
    # Terminal point: W(T) = sqrt(T) * z[0].
    B[N - 1, 0] = np.sqrt(T)

    # Breadth-first bisection on index intervals (l, r) in t_ext.
    queue: deque[tuple[int, int]] = deque()
    queue.append((0, N))
    z_idx = 1
    while queue and z_idx < N:
        l, r = queue.popleft()
        m = (l + r) // 2
        if m == l or m == r:
            continue
        t_l, t_m, t_r = t_ext[l], t_ext[m], t_ext[r]
        # Bridge parameters.
        coef_l = (t_r - t_m) / (t_r - t_l)
        coef_r = (t_m - t_l) / (t_r - t_l)
        std = np.sqrt((t_m - t_l) * (t_r - t_m) / (t_r - t_l))

        # Row i in B corresponds to W at t_ext[i+1].
        row = np.zeros(N)
        if l > 0:
            row += coef_l * B[l - 1, :]
        if r > 0:
            row += coef_r * B[r - 1, :]
        row[z_idx] = std
        B[m - 1, :] = row

        z_idx += 1
        queue.append((l, m))
        queue.append((m, r))

    return B


def build_paths(
    params: AsianOptionParams,
    z: np.ndarray,
    method: PathMethod = "incremental",
) -> np.ndarray:
    """Turn ``(n_paths, N)`` standard normals into GBM price paths.

    Parameters
    ----------
    params
        Option parameters (provides ``S0``, ``r``, ``sigma``, ``T``, ``N``).
    z
        Shape ``(n_paths, N)`` array of iid standard normals.
    method
        ``"incremental"`` (default) or ``"brownian_bridge"``.

    Returns
    -------
    paths : np.ndarray
        Shape ``(n_paths, N)`` with ``paths[:, i] = S(t_{i+1})``.
    """
    if z.ndim != 2 or z.shape[1] != params.N:
        raise ValueError(f"z must have shape (n_paths, {params.N}); got {z.shape}.")

    t = monitoring_times(params)  # shape (N,)
    drift_term = (params.r - 0.5 * params.sigma ** 2) * t

    if method == "incremental":
        dt = params.T / params.N
        # W at monitoring times is sqrt(dt) * cumsum(z).
        W = np.sqrt(dt) * np.cumsum(z, axis=1)
    elif method == "brownian_bridge":
        B = brownian_bridge_matrix(params.N, params.T)
        # W = z @ B.T so that each row is W evaluated at the monitoring times.
        W = z @ B.T
    else:
        raise ValueError(
            f"Unknown path construction '{method}'. "
            "Use 'incremental' or 'brownian_bridge'."
        )

    log_paths = np.log(params.S0) + drift_term + params.sigma * W
    return np.exp(log_paths)


def payoff_from_paths(paths: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """Arithmetic-average Asian payoff for each path."""
    avg = paths.mean(axis=1)
    if option_type == "call":
        return np.maximum(avg - K, 0.0)
    if option_type == "put":
        return np.maximum(K - avg, 0.0)
    raise ValueError(f"Unknown option_type '{option_type}'.")


def geometric_payoff_from_paths(paths: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """Geometric-average Asian payoff for each path (used by control variates)."""
    geom = np.exp(np.log(paths).mean(axis=1))
    if option_type == "call":
        return np.maximum(geom - K, 0.0)
    if option_type == "put":
        return np.maximum(K - geom, 0.0)
    raise ValueError(f"Unknown option_type '{option_type}'.")
