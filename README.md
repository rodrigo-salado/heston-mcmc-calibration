# Bayesian Calibration of the Heston Model

## Adaptive MCMC for Option Market Inference

---

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     BAYESIAN CALIBRATION — HESTON MODEL                      ║
║     Adaptive Metropolis-Hastings via Fourier Pricing         ║
║                                                              ║
║     Python Implementation                                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

**MCMC Sampling • Fourier Option Pricing • Bayesian Inference • Posterior Diagnostics**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-blue.svg)](https://scipy.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-1792_lines-green.svg)]()

</div>

---

## Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [Why This Project Exists](#why-this-project-exists)
3. [What You Will Learn](#what-you-will-learn)
4. [Mathematical Background](#mathematical-background)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [How To Use](#how-to-use)
8. [Understanding The Output](#understanding-the-output)
9. [The Key Graph: Posterior Predictive Smile](#the-key-graph-posterior-predictive-smile)
10. [Results & Findings](#results--findings)
11. [Test Suite](#test-suite)
12. [Skills Demonstrated](#skills-demonstrated)
13. [References & Further Reading](#references--further-reading)
14. [License](#license)
15. [Author](#author)

---

## What Is This Project?

This repository implements a **complete Bayesian calibration pipeline** for the Heston stochastic volatility model, using Markov Chain Monte Carlo (MCMC) to recover the full posterior distribution of model parameters from observed option prices. It does three main things:

### 1. Prices Options Under Heston via Fourier Inversion

**The Problem**: Calibrating any model to market data requires a fast, accurate pricing engine. For the Heston model, closed-form prices exist via Fourier inversion of the characteristic function, but numerical instabilities (branch-cut discontinuities) make the implementation non-trivial.

**The Solution**: The Gil-Peláez inversion formula with the **Little Heston Trap** (Albrecher et al., 2007) formulation, which eliminates the branch-cut discontinuities present in the original Heston (1993) approach:

```
C(K) = S₀ P₁ - K e^{-rT} P₂

where:
    φ(u) = exp( iuS₀ + iurT + (κθ/ξ²)[(b-d)T - 2log((1 - ge^{-dT})/(1-g))]
                + v₀((b-d)/ξ²)(1 - e^{-dT})/(1 - ge^{-dT}) )

    b = κ - ρξiu,   d = √(b² + ξ²(u² + iu)),   g = (b-d)/(b+d)
```

Vectorised across strikes: 9 options × 512 Fourier grid points ≈ 4600 function evaluations per likelihood call.

### 2. Runs Adaptive Metropolis-Hastings MCMC Calibration

**The Challenge**: Standard maximum-likelihood calibration gives a point estimate but no uncertainty. Bayesian calibration recovers the full posterior distribution p(θ | data), quantifying exactly how well the market data constrains each parameter.

**The Algorithm**: Adaptive Metropolis-Hastings with Roberts-Gelman-Gilks optimal scaling:

```
Calibration pipeline (two-phase adaptive MCMC):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1 (Pilot chain, 3000 iter):
    → Diagonal proposal with hand-tuned scales
    → Discard burn-in, estimate posterior covariance from samples
    → Scale by 2.38²/d (Roberts-Gelman-Gilks optimal factor)

Phase 2 (Production chain, 8000 iter):
    → Burn-in: 2000 iterations with tuned multivariate Gaussian proposal
    → Sampling: 8000 iterations → posterior samples {θᵢ}
    → Target acceptance rate: ~23.4% (optimal for 5D Gaussian target)
```

The result is a set of posterior samples for all five Heston parameters (κ, θ, ξ, ρ, v₀), along with credible intervals, marginal densities, and trace plots.

### 3. Validates Calibration via Posterior Predictive Checks

After MCMC, the posterior is evaluated by generating option prices at every posterior sample and comparing the resulting **implied volatility smile** to the market:

```
For each posterior sample θᵢ:
    → Compute Heston prices for all strikes
    → Invert Black-Scholes formula (Brent's method) → implied vols
    → Overlay all smiles → posterior predictive band

True smile:  ──────────────────   (market IVs)
Posterior:   ░░░░░░░░░░░░░░░░░░░░░ (uncertainty band)
```

If the true smile falls inside the predictive band, calibration has succeeded.

---

## Why This Project Exists

### Technical Motivation

Standard Heston calibration (e.g., Levenberg-Marquardt minimisation of pricing error) gives a single parameter vector but no sense of how much uncertainty remains after calibration. This matters when:

- Parameters are weakly identified (e.g., ρ and ξ are correlated)
- Market data is sparse or noisy
- Downstream decisions depend on parameter uncertainty (e.g., hedging, risk management)

Bayesian calibration provides a principled answer: the posterior p(θ | data) is itself the complete summary of parameter uncertainty given the observed data.

### Educational Purpose

This project teaches:

- **Fourier option pricing:** Implementing the characteristic function approach with numerical stability
- **Bayesian inference:** Connecting priors, likelihoods, and posteriors in a concrete financial setting
- **MCMC design:** Adaptive Metropolis-Hastings, proposal tuning, convergence diagnostics
- **Model validation:** Posterior predictive checks, R-hat statistics, effective sample size
- **Software engineering:** Modular Python, dataclasses, type hints, 1792-line test suite

### Historical Context

Option model calibration became critical after the **1987 crash**, which produced the now-ubiquitous **volatility smile** that Black-Scholes (1973) cannot explain. Heston (1993) introduced stochastic volatility to match this smile analytically. But calibrating Heston's five parameters from option prices is an ill-posed inverse problem: different parameter combinations can produce nearly identical prices, leaving substantial parameter uncertainty unresolved.

The Bayesian approach, rooted in Bayes (1763) and made computationally feasible by Metropolis et al. (1953) and Hastings (1970), transforms this into a well-posed inference problem. This project implements the complete pipeline from characteristic function to posterior credible intervals.

---

## What You Will Learn

### Core Concepts

| Concept | How This Project Teaches It |
|---------|----------------------------|
| **Fourier Option Pricing** | Full implementation of Gil-Peláez inversion with Simpson quadrature and Little Heston Trap |
| **Bayes' Theorem** | Explicit construction of prior × likelihood → unnormalised posterior |
| **Metropolis-Hastings** | Random-walk proposal, log-acceptance ratio, accept/reject step |
| **Adaptive MCMC** | Two-phase strategy: pilot chain → covariance estimation → tuned production chain |
| **RGG Optimal Scaling** | Proposal covariance scaled by 2.38²/d to target 23.4% acceptance |
| **Posterior Predictive Check** | Generating synthetic data from posterior and comparing to observed |
| **Gelman-Rubin R-hat** | Convergence diagnostic comparing within- and between-chain variance |
| **MCMC ESS** | Effective sample size via autocorrelation, distinct from particle filter ESS |
| **Implied Volatility** | Root-finding (Brent's method) to invert Black-Scholes for IVs |
| **Black-Scholes as Limit** | Verifying Heston → Black-Scholes when ξ → 0 and v₀ = θ = σ² |

### Practical Skills

| Skill | Application Beyond This Project |
|-------|-------------------------------|
| **Fourier Pricing** | Any model with a known characteristic function (Bates, Variance Gamma, NIG) |
| **Bayesian Calibration** | Any inverse problem: risk model fitting, interest rate curves, credit spreads |
| **Adaptive MCMC** | Bayesian neural network training, epidemiological parameter estimation |
| **Posterior Predictive Checks** | Model validation in any probabilistic setting |
| **Convergence Diagnostics** | Essential for any MCMC-based inference |
| **Modular Python Design** | Clean separation of pricing / inference / types / visualisation |

---

## Mathematical Background

### The Heston Model (Risk-Neutral Measure)

Under the risk-neutral measure Q used for option pricing:

```
dS_t = r S_t dt + √v_t S_t dW_t¹
dv_t = κ(θ - v_t) dt + ξ √v_t dW_t²
⟨dW_t¹, dW_t²⟩ = ρ dt
```

| Parameter | Symbol | Prior | Typical Value |
|-----------|--------|-------|---------------|
| Mean-reversion speed | κ | Log-normal, μ=log(2), σ=0.5 | 1–5 |
| Long-run variance | θ | Normal, μ=0.04, σ=0.02 | 0.04 (20% vol) |
| Vol-of-vol | ξ | Normal, μ=0.30, σ=0.15 | 0.3 |
| Leverage correlation | ρ | Uniform on (−0.999, 0.999) | −0.7 |
| Initial variance | v₀ | Normal, μ=0.04, σ=0.02 | 0.04 |

### Fourier Pricing: Little Heston Trap

The characteristic function φ(u) = 𝔼[e^{iu log Sₜ}] has the closed form given in the module docstring, implemented using the Albrecher et al. (2007) "Little Heston Trap" formulation. The key fix is using `g = (b-d)/(b+d)` rather than the original sign, which eliminates 2πi log-branch jumps:

```
Gil-Peláez inversion:
    C(K) = S₀ P₁ - K e^{-rT} P₂

    Pⱼ = ½ + (1/π) ∫₀^∞ Re[ e^{-iu log K} φⱼ(u) / (iu) ] du

    φ₂(u) = φ(u),     φ₁(u) = φ(u - i) / φ(-i)

Quadrature: Simpson's rule on [1e-8, 200] with 512 points
Complexity: O(N_strikes × N_grid) per pricing call
```

### Bayesian Calibration

By Bayes' theorem:

```
p(θ | data) ∝ p(data | θ) × p(θ)

Likelihood (Gaussian observation errors):
    ℓ(θ) = -½ ∑ᵢ (Pᵢᵐᵃʳᵏᵉᵗ - Pᵢᵐᵒᵈᵉˡ(θ))² / σ²

Log-posterior:
    log p(θ | data) = ℓ(θ) + log p(θ) + constant
```

### Adaptive Metropolis-Hastings

The sampler uses a two-phase strategy:

```
Metropolis-Hastings acceptance step:
    θ_proposed = θ_current + chol(Σ) · z,   z ~ N(0, I)
    
    α = min(1, exp(log p(θ_proposed) - log p(θ_current)))
    
    Accept θ_proposed with probability α; otherwise keep θ_current.

Phase 1 — Pilot chain:
    Σ₀ = diag(σ₁², ..., σ₅²)    [hand-tuned scales]
    After pilot: Σ̂ = Cov(pilot samples after burn-in)

Phase 2 — Production chain:
    Σ = (2.38²/5) × Σ̂            [Roberts-Gelman-Gilks scaling]
    Target acceptance: 23.4%     [optimal for d-dim Gaussian target]
```

### Convergence Diagnostics

**Gelman-Rubin R-hat** (requires multiple chains):

```
B = between-chain variance,   W = within-chain variance
var̂ = (1 - 1/n)W + (1/n)B
R̂ = √(var̂ / W)

R̂ < 1.1  →  convergence acceptable
R̂ < 1.01 →  convergence good
```

**MCMC Effective Sample Size**:

```
ESS = n / (1 + 2 ∑_{k≥1} ρₖ)

where ρₖ is autocorrelation at lag k (truncated at first negative value)
```

---

## Project Structure

```
heston-mcmc-calibration/
│
├── README.md                          # You are here
├── requirements.txt                   # Dependencies
│
├── notebooks/
│   ├── 01_synthetic_data.ipynb        # Generate synthetic market data from known params
│   ├── 02_calibration.ipynb           # Run MCMC calibration interactively
│   └── 03_analysis.ipynb              # Posterior analysis, diagnostics, smile plots
│
├── scripts/
│   ├── run_calibration.py             # CLI entry point: load data → calibrate → save
│   └── generate_report.py             # Generate diagnostic report from saved results
│
├── src/
│   ├── __init__.py
│   ├── pricing.py                     # Fourier pricing engine (char. function + Gil-Peláez)
│   ├── inference.py                   # Priors, likelihood, posterior, MCMC samplers
│   ├── types.py                       # Data structures: HestonParams, MarketData, CalibrationResult
│   ├── utils.py                       # Shared utilities
│   └── visualization.py              # Trace plots, pair plots, posterior predictive smiles
│
└── tests/
    ├── conftest.py                    # Pytest fixtures (params, market data, samplers)
    ├── test_pricing.py                # Fourier pricing tests (655 lines)
    └── test_inference.py              # MCMC and inference tests (835 lines)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `pricing.py` | Characteristic function (Little Heston Trap), Gil-Peláez inversion, Simpson quadrature, Black-Scholes, implied vol (Brent), BS limit test |
| `inference.py` | `LogPrior`, `LogLikelihood`, `LogPosterior`, `MetropolisHastingsSampler`, `AdaptiveMetropolisHastings`, R-hat, MCMC ESS |
| `types.py` | `HestonParams` (frozen dataclass), `MarketData`, `CalibrationResult` (with `summary()`, `save()`, `load()`), `PriorConfig`, `MCMCConfig`, `FourierConfig` |
| `visualization.py` | Trace plots, marginal posterior densities, pair plots with KDE, posterior predictive smile |
| `run_calibration.py` | CLI script: argparse, data loading, calibration, timestamped result saving |

---

## Installation & Setup

**Prerequisites:** Python 3.9 or higher

**Step 1: Clone the repository**

```bash
git clone https://github.com/rodrigo-salado/heston-mcmc-calibration.git
cd heston-mcmc-calibration
```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

**Core dependencies:**

- `numpy` (≥1.24) — Vectorised Fourier integration, MCMC arrays, statistics
- `scipy` (≥1.10) — Statistical distributions for priors; Brent's method for implied vol
- `matplotlib` (≥3.7) — All visualisations (trace plots, pair plots, smile plots)
- `seaborn` (≥0.12) — Enhanced pair plots with KDE, correlation styling
- `pandas` (≥2.0) — Calibration result tables and CSV export

**Optional (for comparison studies):**

```bash
pip install pymc arviz          # PyMC NUTS sampler + ArviZ diagnostics
pip install pytest-benchmark    # Performance regression tests
```

**Step 3: Verify installation**

```bash
python -c "from src.pricing import heston_call_prices; print('✓ Ready')"
```

**No C extensions or system libraries required.** All core functionality uses NumPy, SciPy, and the Python standard library.

---

## How To Use

### Run the Complete Calibration (CLI)

```bash
# Default: synthetic data, 3000 pilot + 8000 production iterations
python scripts/run_calibration.py

# Custom data file
python scripts/run_calibration.py --data data/market/spx_options.npz

# Custom MCMC settings
python scripts/run_calibration.py --pilot-iter 5000 --prod-iter 12000 --seed 123

# All options
python scripts/run_calibration.py --help
```

Expected output:

```
======================================================================
HESTON MODEL BAYESIAN CALIBRATION
======================================================================

Market data: 9 options
  Spot: 100.0
  Rate: 0.0300
  Tenor: 1.0 years
  Strikes: [80, ..., 120]
Sigma noise: 0.050

Running pilot chain: 3000 iterations
  Iteration 300/3000  | Accept: 31.3% | Elapsed: 12.1s | ETA: 108.9s
  ...
  Pilot acceptance rate: 28.4%

Estimated posterior std (from pilot):
   kappa: 0.4821
   theta: 0.0063
      xi: 0.0512
     rho: 0.0894
      v0: 0.0109

Burn-in: 2000 iterations
  Sampling acceptance: 22.1%

Sampling: 8000 iterations
  Sampling acceptance: 23.8%

Calibration complete!

======================================================================
HESTON MODEL CALIBRATION RESULTS
======================================================================

True parameters: {'kappa': 2.0, 'theta': 0.04, 'xi': 0.3, 'rho': -0.7, 'v0': 0.04}

Posterior summary (n_samples=8000):
Acceptance rate: 23.8%
Runtime: 428.3 seconds

----------------------------------------------------------------------
   Param       Mean        Std       2.5%     97.5%
----------------------------------------------------------------------
   kappa     2.2684     0.6100     1.1847     3.6041
   theta     0.0348     0.0085     0.0192     0.0513
      xi     0.3218     0.0640     0.1994     0.4477
     rho    -0.7044     0.1161    -0.9643    -0.5330
      v0     0.0476     0.0134     0.0231     0.0773
----------------------------------------------------------------------
```

### Use as a Library

```python
import numpy as np
from src.types import MarketData, HestonParams, MCMCConfig
from src.inference import calibrate_heston

# Define observed market data
market = MarketData(
    strikes=np.array([80, 85, 90, 95, 100, 105, 110, 115, 120]),
    prices=np.array([23.69, 18.92, 14.45, 10.52, 7.21, 4.58, 2.71, 1.48, 0.74]),
    spot=100.0,
    rate=0.03,
    tenor=1.0,
)

# Optionally customise MCMC settings
config = MCMCConfig(
    pilot_n_iter=5000,
    prod_n_samples=10000,
    random_seed=42,
)

# Run Bayesian calibration
result = calibrate_heston(market, mcmc_config=config, sigma_noise=0.05)

# Inspect results
result.summary()
kappa_ci = result.get_credible_interval("kappa", cred_mass=0.95)
print(f"κ 95% CI: [{kappa_ci[0]:.3f}, {kappa_ci[1]:.3f}]")

# Save and reload
result.save("results/calibration.npz")
```

### Jupyter Notebooks

Three annotated notebooks walk through the full workflow:

| Notebook | Contents |
|----------|----------|
| `01_synthetic_data.ipynb` | Generate synthetic option prices from known Heston params, add observation noise |
| `02_calibration.ipynb` | Run adaptive MCMC interactively, inspect pilot chain and proposal tuning |
| `03_analysis.ipynb` | Trace plots, R-hat diagnostics, pair plots, posterior predictive smile |

### Run the Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing   # With coverage

# Performance benchmarks (requires pytest-benchmark)
pytest tests/test_inference.py -k "benchmark" --benchmark-only
pytest tests/test_inference.py --benchmark-compare   # Compare to previous run
```

---

## Understanding The Output

### Trace Plots

One panel per parameter showing the MCMC chain over iterations. A well-mixed chain looks like "white noise" around a stable mean — no drifts, no stuck periods.

```
   κ
  3 |  ╭╮  ╭╮╭╮    ╭╮     ╭──╮   Good: chain explores freely
  2 |──╯╰──╯╰╯╰────╯╰─────╯  ╰──
  1 |
     0       2000       4000       8000 (iterations)
```

### Marginal Posterior Densities

One panel per parameter showing the histogram and KDE of posterior samples. The 95% credible interval is shaded.

```
 p(κ | data)
     ^
  0.8|      ██████
  0.4|    ████████████      ← true κ = 2.0 falls inside
  0.0+──1───2───3───4──> κ
      └──95% CI──────┘
```

### Pair Plot

A 5×5 grid of scatter plots showing pairwise posterior correlations. Diagonal panels show marginal KDE. This reveals which parameters are jointly identified and which trade off against each other.

```
Key finding: ρ and ξ are negatively correlated in the posterior — a
larger vol-of-vol can partially compensate for a less negative leverage.
```

### Convergence Diagnostics

```
Parameter   R-hat   ESS
────────────────────────
kappa       1.003   612
theta       1.001   843
xi          1.002   724
rho         1.001   891
v0          1.004   578

All R-hat < 1.1: convergence achieved ✓
```

---

## The Key Graph: Posterior Predictive Smile

This is **the most important output** of the calibration. It validates that the posterior parameter distribution is consistent with the observed market data.

### What It Shows

```
   Implied Volatility
       ^
  35% |
      |  ░░░░░░░░░░░░░░░░░░░░░░░░░░░  ← 95% posterior predictive band
  25% |░░░████████████████████████████░
      |░░░████████████████████████████░  ← posterior mean smile
  20% |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
      |  × × × × × × × × ×           ← observed market IVs
  15% |
      +──80──90──100──110──120──> Strike
```

### How To Interpret

- **Crosses (×):** The implied volatilities extracted from observed market option prices via Brent inversion of Black-Scholes. These are the data the model was calibrated to.
- **Solid line:** The implied volatility smile produced using the posterior mean of each parameter. This is the "best single estimate."
- **Shaded band:** The 95% posterior predictive interval — the range of smiles generated by drawing randomly from the full posterior. This represents honest uncertainty about the model's fit.
- **Calibration success:** If all market IVs fall inside the predictive band, the posterior is consistent with the data. A narrow band with market points inside indicates precise, well-identified calibration.

### Why This Matters

The posterior predictive check is the Bayesian answer to "does the model fit?" Unlike a chi-squared test or RMSE, it accounts for parameter uncertainty: even if individual parameter estimates are imprecise, the predicted smiles may still be tight. Conversely, a wide predictive band warns that the market data do not identify the parameters precisely — a signal that the calibration problem is ill-conditioned, regardless of the algorithm used.

---

## Results & Findings

### Calibration Performance (Synthetic Data, True Params Known)

| Parameter | True | Posterior Mean | 95% CI | True Inside? |
|-----------|------|---------------|--------|--------------|
| κ (kappa) | 2.00 | 2.27 | [1.18, 3.60] | ✓ |
| θ (theta) | 0.04 | 0.035 | [0.019, 0.051] | ✓ |
| ξ (xi) | 0.30 | 0.322 | [0.199, 0.448] | ✓ |
| ρ (rho) | −0.70 | −0.704 | [−0.964, −0.533] | ✓ |
| v₀ (v0) | 0.04 | 0.048 | [0.023, 0.077] | ✓ |

All five true parameters fall inside their 95% credible intervals — the posterior is correctly calibrated.

### Key Findings

| Metric | Value | Notes |
|--------|-------|-------|
| **Total MCMC iterations** | 13000 (3000 pilot + 10000 production) | Standard configuration |
| **Posterior samples retained** | 8000 | After 2000 production burn-in |
| **Target acceptance rate** | 23.4% | Roberts-Gelman-Gilks optimal for 5D |
| **Observed acceptance rate** | ~23–25% | Well-tuned adaptive proposal |
| **Fourier grid points** | 512 | Simpson quadrature on [1e-8, 200] |
| **Pricing calls per iteration** | ~1 (likelihood) | Vectorised across all strikes |
| **BS limit accuracy** | < 1e-6 absolute | Heston → Black-Scholes as ξ → 0 ✓ |
| **R-hat (all params)** | < 1.01 | Good convergence |
| **Runtime** | ~400–500 seconds | Standard config on modern hardware |

### Parameter Identifiability

A key finding from the posterior is that **κ is the least identified parameter**: its 95% credible interval spans roughly [1.2, 3.6] — a ratio of 3:1 — even with 9 noise-free option prices. This is not a bug in the algorithm; it reflects the fact that European option prices are relatively insensitive to the mean-reversion speed compared to θ, ξ, and ρ. The pair plot shows the κ-θ posterior correlation is strongly positive (fast reversion with low long-run vol is nearly observationally equivalent to slow reversion with higher long-run vol over short maturities).

### Computational Complexity

| Operation | Cost | Notes |
|-----------|------|-------|
| Single option price | O(N_grid) | 512 characteristic function evaluations |
| Full likelihood (9 strikes) | O(9 × N_grid) | Vectorised across strikes |
| One MCMC iteration | ~2 likelihood calls | Propose + evaluate (rejected or accepted) |
| Full calibration (13000 iter) | ~26000 pricing calls | Pilot + production chains |

---

## Test Suite

The test suite covers 1792 lines across 3 files:

| Test File | Lines | Key Tests |
|-----------|-------|-----------|
| `test_pricing.py` | 655 | BS limit (ξ→0), put-call parity, surface shape, moneyness ordering, no-arbitrage bounds, Simpson vs trapezoidal, implied vol round-trip, input validation |
| `test_inference.py` | 835 | Prior density values, likelihood finiteness, posterior evaluation, MCMC acceptance rate bounds, proposal covariance shape, R-hat calculation, MCMC ESS, CalibrationResult save/load |
| `conftest.py` | 302 | Shared fixtures: `HestonParams`, `MarketData`, `LogPrior`, `LogLikelihood`, `LogPosterior`, `MetropolisHastingsSampler`, synthetic price arrays |

Key test highlights:

```bash
# Verify Heston → Black-Scholes in the zero vol-of-vol limit
test_bs_limit_convergence          # |Heston - BS| < 1e-5 when ξ = 1e-6

# Verify no-arbitrage bounds hold across all strikes
test_call_prices_no_arbitrage      # C(K) ∈ [max(S-Ke^{-rT}, 0), S] for all K

# Verify implied vol round-trip
test_implied_vol_roundtrip         # IV(BS(σ)) ≈ σ to 8 decimal places

# Verify MCMC acceptance rate is in expected range
test_acceptance_rate               # 0.05 < acceptance < 0.60
```

Run with:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing

# Optional: benchmark pricing and likelihood evaluation
pytest tests/test_inference.py -k "benchmark" --benchmark-only
```

---

## Skills Demonstrated

### Technical Skills

| Category | Skills | Location |
|----------|--------|----------|
| **Fourier Methods** | Characteristic function pricing, Gil-Peláez inversion, Little Heston Trap, Simpson quadrature | `pricing.py` |
| **Bayesian Inference** | Prior design, Gaussian likelihood, Bayes' theorem, unnormalised log-posterior | `inference.py` |
| **MCMC** | Metropolis-Hastings, Cholesky proposals, adaptive covariance, Roberts-Gelman-Gilks scaling | `inference.py` |
| **Convergence Diagnostics** | Gelman-Rubin R-hat, MCMC ESS, autocorrelation, trace plot inspection | `inference.py` |
| **Numerical Stability** | Branch-cut handling (Little Heston Trap), log-space arithmetic, input validation | `pricing.py` |
| **Root-Finding** | Brent's method for implied volatility inversion, no-arbitrage bound checks | `pricing.py` |
| **Python Engineering** | Frozen dataclasses, type hints, `mypy --strict`, modular architecture, `argparse` CLI | All modules |
| **Scientific Visualisation** | Trace plots, marginal KDE, pair grids (seaborn), posterior predictive smile | `visualization.py` |
| **Testing** | Pytest fixtures, parametrised tests, 1792-line suite, benchmark tests | `tests/` |

### Software Engineering Practices

| Practice | How Demonstrated |
|----------|-----------------|
| **Clean Architecture** | Strict separation: pricing / inference / types / utils / visualisation |
| **Strong Typing** | Frozen dataclasses (`HestonParams`), type aliases (`ParamsArray`, `SamplesArray`), `mypy --strict` |
| **Comprehensive Docs** | Every class, method, and formula documented with theory, derivations, and examples |
| **Numerical Rigour** | Little Heston Trap for branch-cut stability, log-space likelihood, Simpson quadrature |
| **Reproducibility** | Seeded `np.random.default_rng`, `random_seed` in `MCMCConfig`, timestamped result files |
| **Persistence** | `CalibrationResult.save()` / `.load()` via compressed NumPy, optional PyMC comparison |
| **CLI Interface** | `run_calibration.py` with `argparse`, timestamped outputs, structured results directory |

---

## References & Further Reading

### Foundational Papers

- **Heston, S. L. (1993).** *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.* Review of Financial Studies, 6(2), 327-343.
  - The model itself. Derives the characteristic function and its semi-closed-form inversion.

- **Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).** *The Little Heston Trap.* Wilmott Magazine, 2007(1), 83-92.
  - Identifies and fixes the branch-cut discontinuity in the original Heston characteristic function. Essential for numerically stable pricing.

- **Gil-Peláez, J. (1951).** *Note on the Inversion Theorem.* Biometrika, 38(3/4), 481-482.
  - The Fourier inversion formula used to recover call prices from the characteristic function.

- **Roberts, G. O., Gelman, A., & Gilks, W. R. (1997).** *Weak convergence and optimal scaling of random walk Metropolis algorithms.* The Annals of Applied Probability, 7(1), 110-120.
  - Proves that the optimal acceptance rate for a d-dimensional Gaussian target is 23.4%, and that the optimal scaling is 2.38²/d.

- **Gelman, A., & Rubin, D. B. (1992).** *Inference from iterative simulation using multiple sequences.* Statistical Science, 7(4), 457-472.
  - Introduces the R-hat convergence diagnostic implemented in `compute_r_hat()`.

### Option Pricing Methods Compared

| Method | Closed Form | Handles Smile | Speed | Notes |
|--------|------------|---------------|-------|-------|
| **Black-Scholes** | Yes | No | Instant | Flat vol assumption |
| **Heston (Fourier)** | Semi | Yes | ~ms | Used here (Gil-Peláez + Little Heston Trap) |
| **Monte Carlo** | No | Yes | ~seconds | Flexible but slow for calibration |
| **PDE (ADI)** | No | Yes | ~ms | Alternative to Fourier; harder to implement |
| **Carr-Madan (FFT)** | Semi | Yes | ~ms | Alternative Fourier method; faster for many strikes |

### MCMC Algorithms Compared

| Algorithm | Gradient? | Tuning | Best For |
|-----------|----------|--------|---------|
| **Random-walk MH** | No | Manual/adaptive | Low-dim, expensive likelihoods |
| **Adaptive MH (used here)** | No | Automatic (pilot chain) | Low-dim, noisy likelihood |
| **NUTS (PyMC)** | Yes | Automatic | Smooth posteriors, moderate dim |
| **SMC (particle filter)** | No | N particles | Sequential / online inference |
| **Hamiltonian MC** | Yes | Manual | High-dim smooth posteriors |

*Adaptive Metropolis-Hastings is the right choice when (a) the likelihood is expensive (Fourier pricing per iteration) and (b) gradients are unavailable or unreliable (non-smooth posterior boundaries from prior boxes).*

### Extending This Project

Natural next steps:

- **Multi-maturity calibration:** Stack `MarketData` objects for multiple expiries and calibrate jointly against a full implied volatility surface.
- **Real market data:** Load SPX option chains from Bloomberg or CBOE and run the calibration on live data.
- **Bates model:** Add a jump component (Poisson-Gaussian) to the Heston dynamics and extend the characteristic function accordingly.
- **Gradient-based MCMC:** Implement NUTS via PyMC for comparison (already listed as an optional dependency).
- **SMC² calibration:** Combine the particle filter from `particle-filter-heston` with an outer SMC loop for joint state and parameter inference in a single unified framework.

### Online Resources

- [Heston characteristic function derivation (Rouah, 2013)](https://frouah.com/finance%20notes/Heston%20Model.pdf) — Detailed derivation of both the original and Little Heston Trap formulations
- [Fourier methods for option pricing (Cont & Tankov, 2004)](https://www.routledge.com/Financial-Modelling-with-Jump-Processes/Cont-Tankov/p/book/9781584884132) — Comprehensive treatment of characteristic functions in finance
- [Bayesian Data Analysis, 3rd ed. (Gelman et al., 2013)](http://www.stat.columbia.edu/~gelman/book/) — The standard reference for Bayesian inference, including MCMC diagnostics

---

## License

```
MIT License

Copyright (c) 2026 Rodrigo Antonio Salado Ferrero

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Author

**Rodrigo Antonio Salado Ferrero**

- GitHub: [github.com/rodrigo-salado](https://github.com/rodrigo-salado)
- Repository: [github.com/rodrigo-salado/heston-mcmc-calibration](https://github.com/rodrigo-salado/heston-mcmc-calibration)

---

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     If this project was helpful, please consider:            ║
║                                                              ║
║     ⭐ Starring the repository                               ║
║     🔱 Forking for your own experiments                      ║
║     📝 Opening issues with suggestions or bugs               ║
║                                                              ║
║     Built with Fourier transforms and filter coffee ☕        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

</div>