#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heston Model Bayesian Calibration - Main Package
=================================================

This package provides a complete framework for Bayesian calibration of the
Heston stochastic volatility model using European option prices.

The calibration recovers the five Heston parameters:
    - κ (kappa): Mean reversion speed
    - θ (theta): Long-run variance
    - ξ (xi): Volatility of volatility
    - ρ (rho): Correlation between asset and variance
    - v₀ (v0): Initial variance

with full posterior distributions via Metropolis-Hastings MCMC.

Main components:
    - pricing: Fast Heston pricing via characteristic function and Fourier inversion
    - inference: MCMC sampling with adaptive proposals
    - visualization: Comprehensive plotting for posterior analysis

Example:
    >>> from heston_calibration import calibrate_heston, MarketData
    >>> import numpy as np
    >>> 
    >>> # Prepare market data
    >>> market_data = MarketData(
    ...     strikes=np.array([80, 90, 100, 110, 120]),
    ...     prices=np.array([23.69, 15.75, 9.24, 4.58, 1.84]),
    ...     spot=100.0,
    ...     rate=0.03,
    ...     tenor=1.0
    ... )
    >>> 
    >>> # Run calibration
    >>> result = calibrate_heston(market_data)
    >>> result.summary()

Author: Rodrigo Antonio Salado Ferrero
Repository: github.com/rodrigo-salado/heston-mcmc-calibration
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Rodrigo Antonio Salado Ferrero"
__license__ = "MIT"
__copyright__ = f"Copyright (c) 2024 {__author__}"
__email__ = "rodrigo.salado@example.com"
__status__ = "Production/Stable"

# Import main public API
from src.types import (
    HestonParams,
    MarketData,
    CalibrationResult,
    PriorConfig,
    MCMCConfig,
    FourierConfig,
    CalibrationConfig,
)

from src.pricing import (
    heston_char_func,
    heston_char_func_vectorized,
    heston_call_prices,
    implied_volatility,
    black_scholes_call,
)

from src.inference import (
    LogPrior,
    LogLikelihood,
    LogPosterior,
    MetropolisHastingsSampler,
    AdaptiveMetropolisHastings,
    calibrate_heston,
)

from src.visualization import (
    plot_trace,
    plot_posterior_density,
    plot_pair_grid,
    plot_posterior_predictive,
    plot_calibration_summary,
)

# Define what gets imported with "from src import *"
__all__ = [
    # Types
    "HestonParams",
    "MarketData",
    "CalibrationResult",
    "PriorConfig",
    "MCMCConfig",
    "FourierConfig",
    "CalibrationConfig",
    # Pricing
    "heston_char_func",
    "heston_char_func_vectorized",
    "heston_call_prices",
    "implied_volatility",
    "black_scholes_call",
    # Inference
    "LogPrior",
    "LogLikelihood",
    "LogPosterior",
    "MetropolisHastingsSampler",
    "AdaptiveMetropolisHastings",
    "calibrate_heston",
    # Visualization
    "plot_trace",
    "plot_posterior_density",
    "plot_pair_grid",
    "plot_posterior_predictive",
    "plot_calibration_summary",
]

# Package metadata for setuptools
PACKAGE_INFO = {
    "name": "heston-calibration",
    "version": __version__,
    "description": "Bayesian calibration of Heston model from option prices",
    "long_description": __doc__,
    "author": __author__,
    "author_email": __email__,
    "license": __license__,
    "url": "https://github.com/rodrigo-salado/heston-mcmc-calibration",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial",
    ],
    "python_requires": ">=3.9",
    "install_requires": [
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    "extras_require": {
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.4.0",
            "ruff>=0.0.280",
            "jupyterlab>=4.0.0",
        ],
        "pymc": [
            "pymc>=5.0.0",
            "arviz>=0.15.0",
        ],
    },
}