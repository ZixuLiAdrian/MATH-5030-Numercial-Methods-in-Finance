# asian-option-pricer

**Fast, accurate pricing of arithmetic Asian options under GBM — with variance reduction.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZixuLiAdrian/MATH5030-Numerical-Methods-in-Finance/blob/main/notebooks/demo.ipynb)

*MATH 5030 — Numerical Methods in Finance — Columbia University, Spring 2026*

---

Pricing an arithmetic Asian option requires simulation because the average of lognormal prices has no closed-form distribution. This package implements two analytic benchmarks and six Monte Carlo / Quasi-Monte Carlo estimators, and shows that combining a **geometric-Asian control variate** with **Brownian-bridge Sobol QMC** reduces pricing error by up to **82×** versus plain Monte Carlo at the same computational cost.

---

## Installation

```bash
pip install asian-option-pricer
```

Or from source:

```bash
git clone https://github.com/ZixuLiAdrian/MATH5030-Numerical-Methods-in-Finance
cd MATH5030-Numerical-Methods-in-Finance
pip install -e ".[test]"
```

---

## Quick start

```python
from asian_option_pricer import AsianOptionParams, antithetic_cv_price, rqmc_sobol_price

params = AsianOptionParams(S0=100, K=100, r=0.05, sigma=0.30, T=1.0, N=50)

# Antithetic + control variate (~25x variance reduction)
print(antithetic_cv_price(params, n_paths=100_000))
# {'price': 8.0726, 'std_err': 0.0005, 'runtime_s': 0.04, 'n_paths': 100000, 'beta': 1.03, 'var_reduction': 586.0}

# Randomised QMC with Brownian bridge (~82x RMSE reduction)
print(rqmc_sobol_price(params, n_paths=131_072, n_replications=16, path_method="brownian_bridge"))
# {'price': 8.0727, 'std_err': 0.0002, 'runtime_s': 0.12, 'n_paths': 131072, ...}
```

---

## API reference

### `AsianOptionParams`

Frozen dataclass holding all option parameters. Validated automatically by every pricing function.

| Parameter | Type | Description |
| --- | --- | --- |
| `S0` | `float` | Initial stock price |
| `K` | `float` | Strike price (> 0) |
| `r` | `float` | Risk-free rate |
| `sigma` | `float` | Volatility (> 0) |
| `T` | `float` | Time to maturity in years (> 0) |
| `N` | `int` | Number of monitoring dates (≥ 1) |
| `option_type` | `str` | `"call"` (default) or `"put"` |

### Analytic benchmarks

| Function | Returns | Description |
| --- | --- | --- |
| `geometric_asian_call_price(params)` | `float` | Exact Kemna–Vorst (1990) closed form for the discrete geometric Asian call. |
| `levy_approx_call_price(params)` | `float` | Levy (1992) lognormal moment-matching approximation. Accurate for `sigma ≲ 0.3`. |

### MC / QMC estimators

All estimators return a `dict` with at minimum `{price, std_err, runtime_s, n_paths}`.

| Function | Extra keys | Description |
| --- | --- | --- |
| `standard_mc_price(params, n_paths, seed, path_method)` | — | Plain Monte Carlo. |
| `antithetic_mc_price(params, n_paths, seed, path_method)` | — | Antithetic variates; pairs each draw `z` with `−z`. |
| `control_variate_price(params, n_paths, seed, path_method)` | `beta`, `var_reduction` | Geometric-Asian control variate with sample-estimated `beta`. |
| `antithetic_cv_price(params, n_paths, seed, path_method)` | `beta`, `var_reduction` | Antithetic + control variate combined. |
| `sobol_qmc_price(params, n_paths, seed, path_method)` | — | Single scrambled Sobol replication. `std_err` is a proxy only. |
| `rqmc_sobol_price(params, n_paths, seed, n_replications, path_method)` | `n_replications` | Multiple scrambles; reports an honest cross-replication SE. |

**`path_method`** — `"incremental"` (default) or `"brownian_bridge"`. The bridge construction concentrates path variance on the best-equidistributed Sobol coordinates and is recommended for QMC estimators.

---

## License

MIT — see [LICENSE](LICENSE). Authors: Chan-Peng Chen, Jack Jia, Zixu Li, Benjamin Tu, Tianrui Wang, Kaifan Ye.

---

## Results at a glance

Reference: `S0 = 100`, `K = 100`, `r = 5%`, `sigma = 30%`, `T = 1 year`, `N = 50`.
RMSE measured across 12 independent replications; see `experiments/run_efficiency.py`.

| Method | RMSE @ 524k paths | Gain over plain MC |
| --- | ---: | ---: |
| Standard Monte Carlo | 1.9 × 10⁻² | 1.0 × |
| Antithetic variates | 8.2 × 10⁻³ | ≈ 2.6 × |
| Sobol QMC — incremental | 1.8 × 10⁻³ | ≈ 10 × |
| Control variate (geometric Asian) | 7.3 × 10⁻⁴ | ≈ 26 × |
| Antithetic + Control variate | 7.8 × 10⁻⁴ | ≈ 24 × |
| Randomised QMC — Brownian bridge | 3.5 × 10⁻⁴ | ≈ 54 × |
| **Sobol QMC — Brownian bridge** | **2.3 × 10⁻⁴** | **≈ 82 ×** |

![Efficiency: RMSE vs paths and vs CPU time](results/figures/efficiency_rmse_vs_time.png)

---

## Mathematical setup

Under the risk-neutral measure the underlying follows geometric Brownian motion

$$ \frac{dS_t}{S_t} = r\,dt + \sigma\,dW_t, \qquad S_0 \text{ given}. $$

For equally-spaced monitoring dates $t_i = i\,\Delta t,\; i = 1, \dots, N,\;\Delta t = T/N$ the **arithmetic average** and **Asian call payoff** are

$$ A_N \;=\; \frac{1}{N}\sum_{i=1}^N S(t_i), \qquad \text{payoff} \;=\; \bigl(A_N - K\bigr)^+. $$

The arbitrage-free price is $C = e^{-rT}\,\mathbb{E}\bigl[(A_N - K)^+\bigr]$. Because $A_N$ is a sum of correlated lognormals, $C$ has no elementary closed form.

The auxiliary **geometric average** $G_N = \bigl(\prod_i S(t_i)\bigr)^{1/N}$ is lognormal and admits a closed form (Kemna–Vorst 1990). We use that price as a benchmark and as the control mean for variance reduction.

---

## Validation

### Geometric Asian: closed form vs. Monte Carlo

Kemna–Vorst closed form versus antithetic MC at 200k pairs. Maximum $|z|$-score across a 12-scenario `(sigma, K)` grid is **0.86** — every closed-form price within one standard error of its MC estimate.

| sigma | K | closed form | MC price | \|z\|-score |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 |  90 | 11.9127 | 11.9136 | 0.74 |
| 0.10 | 100 |  3.6355 |  3.6400 | 0.85 |
| 0.20 | 100 |  5.6411 |  5.6516 | 0.86 |
| 0.30 | 100 |  7.6225 |  7.6394 | 0.86 |
| 0.50 | 100 | 11.3296 | 11.3590 | 0.82 |

### Arithmetic Asian: method error vs. high-precision reference

Errors at `S0 = 100`, `K = 100`, `T = 1`, `N = 50` (reference SE ≤ 3 × 10⁻⁴):

| Method | sigma = 0.10 | sigma = 0.20 | sigma = 0.30 | sigma = 0.50 |
| --- | ---: | ---: | ---: | ---: |
| Levy approximation | +0.0059 | +0.0192 | +0.0460 | +0.1664 |
| Standard MC | +0.0046 | +0.0114 | +0.0200 | +0.0401 |
| Antithetic | +0.0048 | +0.0099 | +0.0161 | +0.0311 |
| Control variate | +0.0001 | +0.0002 | +0.0001 | +0.0006 |
| Antithetic + CV | −0.0002 | −0.0008 | −0.0015 | −0.0035 |
| Sobol (bridge) | −0.0001 | −0.0000 | +0.0002 | +0.0010 |
| RQMC (bridge) | +0.0002 | +0.0005 | +0.0008 | +0.0014 |

### Continuous-limit cross-validation (PyFENG BsmAsianJsu)

As $N \to \infty$ discrete prices converge to the continuous arithmetic Asian price. Cross-validated against PyFENG's `BsmAsianJsu` (Johnson's SU approximation):

| N | Our discrete price | PyFENG JSU | Gap |
| ---: | ---: | ---: | ---: |
|   52 | 5.8541 | 5.7630 | +0.0912 |
|  250 | 5.7824 | 5.7630 | +0.0194 |
|  500 | 5.7719 | 5.7630 | +0.0090 |
| 1000 | 5.7679 | 5.7630 | +0.0049 |

*Gap shrinks at the expected $O(1/N)$ rate.*

---

## Robustness

### Random parameter sweep

200 scenarios across `sigma ∈ [0.05, 0.80]`, `T ∈ [0.25, 3.0]`, moneyness `K/S0 ∈ [0.7, 1.3]`, `r ∈ [0.0, 0.10]`, `N ∈ {12, 26, 52, 100, 250}`:

| Method | NaNs | Out-of-bounds | Mean SE |
| --- | ---: | ---: | ---: |
| Standard MC | 0 | ≤ 1 | 0.142 |
| Antithetic | 0 | ≤ 1 | 0.113 |
| Control variate | 0 | ≤ 1 | 0.017 |
| Antithetic + CV | 0 | ≤ 1 | 0.017 |
| Sobol bridge | 0 | ≤ 1 | 0.142 *(proxy SE)* |
| RQMC bridge | 0 | ≤ 1 | **0.007** |

Zero NaNs, zero crashes. The rare out-of-bounds count (≤ 1 per method) reflects Monte Carlo noise on a tight geometric-Asian lower bound — not a pricing error.

### Monotonicity

Strike grid $K \in [70, 130]$ and volatility grid $\sigma \in [0.05, 0.80]$ (31 points each). Every variance-reduced method is **weakly decreasing in K** and **weakly increasing in σ** — no violations.

### Edge cases

- `sigma → 0`: all MC/QMC estimators converge to the deterministic intrinsic value (2.46511). The geometric closed-form correctly gives a different value (2.45495) by Jensen's inequality.
- Deep OTM (`K ∈ {200, 400}`): every estimator returns exactly `0.0`.
- Dense monitoring (`N = 500`): variance-reduced estimators (CV, A+CV, Sobol bridge, RQMC) agree within 2 bp of each other.

---

## Repository layout

```
MATH5030-Numerical-Methods-in-Finance/
├─ src/asian_option_pricer/          # core package
│  ├─ analytic.py                    # Kemna-Vorst + Levy
│  ├─ control_variate.py             # CV and antithetic+CV estimators
│  ├─ monte_carlo.py                 # plain + antithetic MC
│  ├─ paths.py                       # incremental + Brownian-bridge construction
│  ├─ qmc.py                         # Sobol QMC + randomised QMC
│  ├─ models.py, utils.py
│  └─ benchmarks.py
├─ experiments/
│  ├─ run_benchmarks.py              # smoke test
│  ├─ run_validation.py              # validation tables
│  ├─ run_robustness.py              # robustness tables
│  └─ run_efficiency.py              # efficiency table + figure
├─ tests/                            # 47 unit tests
├─ notebooks/demo.ipynb              # interactive walkthrough
├─ results/
│  ├─ tables/                        # CSVs from experiments
│  └─ figures/                       # PNGs from experiments
├─ pyproject.toml
├─ LICENSE
└─ README.md
```

---

## References

1. Kemna, A. G. Z., & Vorst, A. C. F. (1990). *A pricing method for options based on average asset values.* Journal of Banking and Finance, 14(1), 113–129.
2. Levy, E. (1992). *Pricing European average rate currency options.* Journal of International Money and Finance, 11(5), 474–491.
3. Turnbull, S. M., & Wakeman, L. M. (1991). *A quick algorithm for pricing European average options.* Journal of Financial and Quantitative Analysis, 26(3), 377–389.
4. Caflisch, R. E., Morokoff, W. J., & Owen, A. B. (1997). *Valuation of mortgage-backed securities using Brownian bridges to reduce effective dimension.* Journal of Computational Finance, 1(1), 27–46.
5. Owen, A. B. (1997). *Scrambled net variance for integrals of smooth functions.* Annals of Statistics, 25(4), 1541–1562.
6. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering.* Springer.
7. Choi, J., & Kwok, Y. K. (2022). *Moments of the continuous arithmetic Asian option under GBM.* SIAM Journal on Financial Mathematics. (Implemented as `BsmAsianJsu` in [PyFENG](https://github.com/PyFE/PyFENG).)
