# Efficient Numerical Methods for Pricing Arithmetic Asian Options

*MATH 5030 — Numerical Methods — Final Project Report*

This repository studies the pricing of **discretely monitored arithmetic
Asian call options** under the Black–Scholes / GBM model and compares several
published numerical methods on accuracy, computational efficiency, and
robustness.

The arithmetic average of lognormal prices does not admit a closed-form
distribution, so the exact price must be obtained numerically. We implement
seven estimators from the literature and measure how tightly each one prices
the option per unit of CPU time; we then show that two classical
variance-reduction ideas — **geometric-Asian control variate** and
**Brownian-bridge Sobol QMC** — improve root-mean-square error by **one to
two orders of magnitude** over plain Monte Carlo on this problem.

---

## 1. Results at a glance

Reference parameter set: `S0 = 100`, `K = 100`, `r = 5%`, `sigma = 30%`,
`T = 1 year`, `N = 50` monitoring dates. Reference price obtained from
randomised-QMC (2M Brownian-bridge paths, 32 scrambles) combined with a
second 1M-path antithetic+control-variate run; reference standard error is
below **2 × 10⁻⁴**.

| Method | RMSE @ 524k paths | Runtime | Gain over plain MC |
| --- | ---: | ---: | ---: |
| Standard Monte Carlo | 1.9 × 10⁻² | 0.18 s | 1.0 × |
| Antithetic variates | 8.2 × 10⁻³ | 0.16 s | ≈ 2.6 × |
| Sobol QMC — incremental | 1.8 × 10⁻³ | 0.46 s | ≈ 10 × |
| Control variate (geometric Asian) | 7.3 × 10⁻⁴ | 0.24 s | ≈ 26 × |
| Antithetic + Control variate | 7.8 × 10⁻⁴ | 0.19 s | ≈ 24 × |
| Randomised QMC — Brownian bridge | 3.5 × 10⁻⁴ | 0.43 s | ≈ 54 × |
| **Sobol QMC — Brownian bridge** | **2.3 × 10⁻⁴** | 0.43 s | **≈ 82 ×** |

*RMSE is measured across 12 independent replications with disjoint seeds;
see `experiments/run_efficiency.py`.*

![Efficiency: RMSE vs paths (left) and vs CPU time (right)](results/figures/efficiency_rmse_vs_time.png)

**Two classical methods dominate the efficiency curve:**

1. **Geometric-Asian control variate** leverages the closed-form price of
   the geometric Asian (Kemna–Vorst 1990) to subtract the dominant
   component of the arithmetic payoff's variance. On this problem it
   yields roughly a **25× variance reduction** with essentially no
   additional CPU cost.
2. **Brownian-bridge Sobol QMC** reorganises the random draws so that the
   first (best-equidistributed) Sobol coordinates control the coarse shape
   of the path (Glasserman 2004 §4.3). Combined with scrambling, it
   delivers an **~80× RMSE reduction** versus plain MC on this arithmetic
   Asian payoff.

---

## 2. Mathematical setup

Under the risk-neutral measure the underlying follows geometric Brownian
motion

$$ \frac{dS_t}{S_t} = r\,dt + \sigma\,dW_t, \qquad S_0 \text{ given}. $$

For equally-spaced monitoring dates $t_i = i\,\Delta t,\; i = 1, \dots, N,\;\Delta t = T/N$
the **arithmetic average** and **arithmetic Asian call payoff** are

$$ A_N \;=\; \frac{1}{N}\sum_{i=1}^N S(t_i), \qquad \text{payoff} \;=\; \bigl(A_N - K\bigr)^+. $$

The arbitrage-free price is $C = e^{-rT}\,\mathbb{E}\bigl[(A_N - K)^+\bigr]$.
Because $A_N$ is a sum of correlated lognormals, $C$ has no elementary
closed form.

The auxiliary **geometric average**
$G_N = \bigl(\prod_i S(t_i)\bigr)^{1/N}$
is lognormal and therefore *does* admit a closed form (Kemna–Vorst 1990).
We use that price both as a benchmark in its own right and as the control
mean for variance reduction.

---

## 3. Methods implemented

All estimators take a common `AsianOptionParams` object and an `n_paths`
budget, and return `{price, std_err, runtime_s, n_paths, ...}`.

| Module | Function | What it does |
| --- | --- | --- |
| `analytic.py` | `geometric_asian_call_price` | Exact discrete Kemna–Vorst formula. |
| `analytic.py` | `levy_approx_call_price` | Levy (1992) lognormal moment-matching approximation. |
| `monte_carlo.py` | `standard_mc_price` | Plain MC. |
| `monte_carlo.py` | `antithetic_mc_price` | Antithetic variates. |
| `control_variate.py` | `control_variate_price` | Geometric-Asian control variate with sample-estimated `beta`. |
| `control_variate.py` | `antithetic_cv_price` | Antithetic + control variate combined. |
| `qmc.py` | `sobol_qmc_price` | Scrambled Sobol QMC, single replication. |
| `qmc.py` | `rqmc_sobol_price` | Randomised QMC, multiple scrambles, honest cross-replication SE. |
| `paths.py` | `build_paths` | Incremental **or** Brownian-bridge path construction. |

### Why Brownian bridge matters for QMC

The Sobol sequence is only *partially* equidistributed — its earliest
coordinates are much better-balanced than its last. The incremental
construction assigns equal variance to every coordinate, so the last
monitoring dates are simulated from the Sobol sequence's worst coordinates.
The Brownian-bridge construction places the terminal value on the first
coordinate and then fills in midpoints by bisection, concentrating the
integrand's variance on the best Sobol coordinates.

We implement it as a one-time $N \times N$ matrix multiplication:
`W = z @ B.T`, where the bridge matrix $B$ is built once and cached
(`brownian_bridge_matrix` in `paths.py`). The unit test
`test_brownian_bridge_matrix_reproduces_covariance` verifies
$B B^\top = \bigl(\min(t_i, t_j)\bigr)_{i,j}$ exactly.

### Fixing the discrete Kemna–Vorst formula

The scaffold shipped with an incorrect $\sigma_g$ for *discretely*
monitored geometric Asians. A 2M-path antithetic Monte Carlo disagreed with
the closed form by **19 standard errors** at `sigma = 0.2`. The correct
moments for equally-spaced monitoring are

$$ \sigma_g^2\,T \;=\; \sigma^2 T\,\frac{(N+1)(2N+1)}{6 N^2}, \qquad b_g \;=\; (r - \tfrac{1}{2}\sigma^2)\,\frac{N+1}{2N} \;+\; \tfrac{1}{2}\sigma_g^2. $$

After the fix, the closed form agrees with antithetic Monte Carlo to within
`0.9 × SE` across a 12-scenario validation grid (see §4).

---

## 4. Validation

The validation pipeline lives in `experiments/run_validation.py` and writes
two tables to `results/tables/`.

### 4.1 Geometric Asian: closed form vs. Monte Carlo

`geometric_validation.csv`: Kemna–Vorst closed form versus an antithetic MC
estimator at 200k pairs. The maximum $|z|$-score across a 12-scenario
`(sigma, K)` grid is **0.86** — i.e. every closed-form price is within one
standard error of its MC estimate.

| sigma | K | closed form | MC price | |z|-score |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 |  90 | 11.9127 | 11.9136 | 0.74 |
| 0.10 | 100 |  3.6355 |  3.6400 | 0.85 |
| 0.20 | 100 |  5.6411 |  5.6516 | 0.86 |
| 0.30 | 100 |  7.6225 |  7.6394 | 0.86 |
| 0.50 | 100 | 11.3296 | 11.3590 | 0.82 |

### 4.2 Arithmetic Asian: method error vs. a high-precision reference

`arithmetic_validation.csv`: for the same grid we compute a high-precision
reference (RQMC bridge + antithetic-CV, ~2M paths each, reference SE ≤
3 × 10⁻⁴), then report each method's error at a fixed 100k-path budget.

Representative errors at `S0 = 100`, `K = 100`, `T = 1`, `N = 50`:

| Method | sigma = 0.10 | sigma = 0.20 | sigma = 0.30 | sigma = 0.50 |
| --- | ---: | ---: | ---: | ---: |
| Levy approximation | +0.0059 | +0.0192 | +0.0460 | +0.1664 |
| Standard MC | +0.0046 | +0.0114 | +0.0200 | +0.0401 |
| Antithetic | +0.0048 | +0.0099 | +0.0161 | +0.0311 |
| Control variate | +0.0001 | +0.0002 | +0.0001 | +0.0006 |
| Antithetic + CV | −0.0002 | −0.0008 | −0.0015 | −0.0035 |
| Sobol (bridge) | −0.0001 | −0.0000 | +0.0002 | +0.0010 |
| RQMC (bridge) | +0.0002 | +0.0005 | +0.0008 | +0.0014 |

**Key observations.** (1) The Levy approximation is excellent up to `sigma ≈
0.3` (<~50 bps), but deteriorates rapidly beyond that (>200 bps at
`sigma = 0.5`) — consistent with Levy's own published tables. (2) CV, A+CV,
Sobol-bridge and RQMC-bridge are all within a handful of basis points of
the reference across every scenario, regardless of moneyness or volatility.

---

## 5. Robustness

`experiments/run_robustness.py` produces three tables.

### 5.1 Random parameter sweep (`robustness_random.csv`)

200 scenarios with `sigma ∈ [0.05, 0.80]`, `T ∈ [0.25, 3.0]`, moneyness
`K/S0 ∈ [0.7, 1.3]`, `r ∈ [0.0, 0.10]`, `N ∈ {12, 26, 52, 100, 250}`:

| Method | NaNs | Out-of-bounds | Mean runtime | Mean SE |
| --- | ---: | ---: | ---: | ---: |
| Standard MC | 0 | 0 | 21 ms | 0.142 |
| Antithetic | 0 | 0 | 17 ms | 0.113 |
| Control variate | 0 | 0 | 27 ms | 0.017 |
| Antithetic + CV | 0 | 0 | 22 ms | 0.017 |
| Sobol bridge | 0 | 0 | 49 ms | 0.142 *(proxy SE)* |
| RQMC bridge | 0 | 0 | 59 ms | **0.007** |

Zero NaNs, zero out-of-bounds prices, zero crashes. The RQMC bridge SE is
honest (cross-scramble); the single-run Sobol "SE" is the classical MC
formula applied to QMC samples, which is why it looks similar to plain MC
despite a much tighter empirical RMSE.

### 5.2 Monotonicity (`robustness_monotonicity.csv`)

Strike grid $K \in [70, 130]$ (31 points) and volatility grid
$\sigma \in [0.05, 0.80]$ (31 points). For every method checked (CV, A+CV,
RQMC) the call price is **weakly decreasing in K** and **weakly increasing
in σ** — no violations — which matches arbitrage-free theory.

### 5.3 Edge cases (`robustness_edge_cases.csv`)

* `sigma → 0`: all Monte Carlo / QMC estimators converge to the
  deterministic intrinsic value (2.46511). The geometric closed-form
  correctly converges to a *different* deterministic value (2.45495),
  reflecting Jensen's inequality between the arithmetic and geometric
  averages.
* Deep OTM (`K ∈ {200, 400}`): every estimator returns exactly `0.0`.
* Dense monitoring (`N = 500`): all estimators agree to within 4 bp.

---

## 6. Efficiency study

`experiments/run_efficiency.py` sweeps the path budget across five orders
of magnitude (2k → 524k) and reports RMSE vs. paths *and* RMSE vs. CPU
time. Each grid point is averaged over 12 independent repetitions with
disjoint seeds. Full results in `results/tables/efficiency.csv`; the
figure `results/figures/efficiency_rmse_vs_time.png` is reproduced in §1.

Key take-aways:

* **All Monte Carlo methods show the expected $n^{-1/2}$ convergence**; the
  control-variate and antithetic+CV curves sit a full order of magnitude
  below plain MC in parallel.
* **Sobol bridge achieves super-$n^{-1/2}$ convergence**: going from 2k to
  524k paths (256× more work) reduces RMSE by a factor of ≈ 37 — faster
  than the $\sqrt{256} = 16$ a plain MC would deliver.
* **Incremental Sobol is only ~3× better than plain MC**; the bridge
  construction is what unlocks QMC's variance advantage on this payoff.

---

## 7. Installation & usage

```bash
git clone <repo-url>
cd asian_option_project_scaffold
python3 -m pip install -r requirements.txt
python3 -m pip install -e .   # editable install of the package

# Quick sanity check
python3 experiments/run_benchmarks.py

# Full experiment pipeline (~2 minutes total on a laptop)
python3 experiments/run_validation.py
python3 experiments/run_robustness.py
python3 experiments/run_efficiency.py

# Test suite
PYTHONPATH=src python3 -m pytest tests/ -v
```

Minimal usage example:

```python
from asian_option_pricer import (
    AsianOptionParams,
    antithetic_cv_price,
    rqmc_sobol_price,
)

params = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.30, T=1.0, N=50)

print(antithetic_cv_price(params, n_paths=100_000))
# {'price': 8.0726, 'std_err': 0.0005, 'runtime_s': 0.04, 'n_paths': 100000, 'beta': 1.03}

print(rqmc_sobol_price(params, n_paths=131_072, n_replications=16,
                       path_method="brownian_bridge"))
# {'price': 8.0727, 'std_err': 0.0002, 'runtime_s': 0.12, 'n_paths': 131072, ...}
```

---

## 8. Repository layout

```
asian_option_project_scaffold/
├─ src/asian_option_pricer/          # core package
│  ├─ analytic.py                    # Kemna-Vorst + Levy (fixed discrete formula)
│  ├─ control_variate.py             # CV and antithetic+CV estimators
│  ├─ monte_carlo.py                 # plain + antithetic MC
│  ├─ paths.py                       # incremental + Brownian-bridge construction
│  ├─ qmc.py                         # Sobol QMC + randomised QMC
│  ├─ models.py, utils.py
│  └─ benchmarks.py                  # benchmark_suite helper
├─ experiments/
│  ├─ run_benchmarks.py              # smoke test at canonical parameters
│  ├─ run_validation.py              # §4 tables
│  ├─ run_robustness.py              # §5 tables
│  └─ run_efficiency.py              # §6 table + figure
├─ tests/                            # 47 unit tests — all passing
├─ results/
│  ├─ tables/                        # CSVs produced by the experiments
│  └─ figures/                       # PNGs produced by the experiments
├─ pyproject.toml
├─ requirements.txt
└─ README.md                         # this report
```

---

## 9. Test coverage

`pytest` reports **47 passing tests** covering:

* **Analytic prices** — closed-form geometric Asian matches antithetic MC
  to < 3 SE; reduces to Black–Scholes at `N = 1`; correct zero-volatility
  limit; Levy approximation matches MC to < 1% at `sigma = 0.2`.
* **Path construction** — Brownian-bridge matrix reproduces the
  Brownian-motion covariance exactly for arbitrary `N`; terminal value sits
  on the first Sobol coordinate; both constructions match theoretical mean
  and variance empirically.
* **Estimators** — every estimator has the expected return keys, is
  deterministic given a seed, rejects invalid budgets, produces the
  expected variance-reduction ordering (antithetic < MC, CV < antithetic,
  RQMC ≤ CV), and deep-ITM / deep-OTM prices lie in the correct range.
* **Monotonicity** — every variance-reduced estimator is weakly decreasing
  in `K` and weakly increasing in `sigma` on dense grids.

---

## 10. References

1. Kemna, A. G. Z., & Vorst, A. C. F. (1990). *A pricing method for options
   based on average asset values.* Journal of Banking and Finance, 14(1),
   113–129.
2. Levy, E. (1992). *Pricing European average rate currency options.*
   Journal of International Money and Finance, 11(5), 474–491.
3. Turnbull, S. M., & Wakeman, L. M. (1991). *A quick algorithm for
   pricing European average options.* Journal of Financial and Quantitative
   Analysis, 26(3), 377–389.
4. Caflisch, R. E., Morokoff, W. J., & Owen, A. B. (1997). *Valuation of
   mortgage-backed securities using Brownian bridges to reduce effective
   dimension.* Journal of Computational Finance, 1(1), 27–46.
5. Owen, A. B. (1997). *Scrambled net variance for integrals of smooth
   functions.* Annals of Statistics, 25(4), 1541–1562.
6. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering.*
   Springer. Chapters 3 (path simulation) and 5 (QMC).

---

## 11. Reproducibility

Every experiment uses a fixed seed (`SEED = 20240420` by default).
`numpy.random.default_rng` plus `scipy.stats.qmc.Sobol(..., seed=...)` are
deterministic given those seeds, so all tables and the efficiency figure
reproduce bit-for-bit on re-run. The full `run_validation` + `run_robustness`
+ `run_efficiency` pipeline completes in **under two minutes** on a 2020
MacBook Pro (M1, single-threaded NumPy).
