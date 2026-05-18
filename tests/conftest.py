#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest Configuration and Fixtures
==================================

This module provides shared fixtures and configuration for all tests.

Fixtures:
    - heston_true_params: Ground truth parameters for synthetic data
    - market_data: Market data fixture with synthetic option prices
    - rng: Reproducible random number generator
    - tolerance: Numerical tolerance for floating point comparisons

Usage:
    pytest tests/ -v --cov=src --cov-report=html

Author: Rodrigo Antonio Salado Ferrero
Repository: github.com/rodrigo-salado/heston-mcmc-calibration
License: MIT
"""

import pytest
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple
from pathlib import Path
import sys

# Add project root to sys.path so `src` imports work when running pytest.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from src
from src.types import HestonParams, MarketData, PriorConfig, MCMCConfig, FourierConfig


# =============================================================================
# Constants
# =============================================================================

# Numerical tolerance for floating point comparisons
TOLERANCE = 1e-8

# Common test parameters
S0_TEST = 100.0
R_TEST = 0.03
T_TEST = 1.0
STRIKES_TEST = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=np.float64)
SIGMA_NOISE_TEST = 0.05


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def heston_true_params() -> HestonParams:
    """
    Ground truth Heston parameters for synthetic data generation.
    
    These parameters represent typical equity index values:
        - kappa: 2.0 (mean reversion speed)
        - theta: 0.04 (long-run variance, 20% vol)
        - xi: 0.3 (volatility of volatility)
        - rho: -0.7 (leverage effect - negative correlation)
        - v0: 0.04 (initial variance)
    
    Returns:
        HestonParams object with ground truth values
    """
    return HestonParams(
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        v0=0.04,
    )


@pytest.fixture
def market_data(heston_true_params: HestonParams) -> MarketData:
    """
    Synthetic market data generated from true Heston parameters.
    
    This fixture generates option prices using the Heston model with
    the ground truth parameters and adds Gaussian noise to simulate
    market microstructure effects.
    
    Args:
        heston_true_params: Ground truth parameters
    
    Returns:
        MarketData object with synthetic option prices
    """
    from src.pricing import heston_call_prices
    
    # Compute true model prices
    true_prices = heston_call_prices(
        S0=S0_TEST,
        strikes=STRIKES_TEST,
        r=R_TEST,
        T=T_TEST,
        kappa=heston_true_params.kappa,
        theta=heston_true_params.theta,
        xi=heston_true_params.xi,
        rho=heston_true_params.rho,
        v0=heston_true_params.v0,
    )
    
    # Add Gaussian noise
    rng = np.random.default_rng(42)
    market_prices = true_prices + SIGMA_NOISE_TEST * rng.standard_normal(len(STRIKES_TEST))
    
    return MarketData(
        strikes=STRIKES_TEST,
        prices=market_prices,
        spot=S0_TEST,
        rate=R_TEST,
        tenor=T_TEST,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """
    Reproducible random number generator for tests.
    
    Returns:
        NumPy random Generator with fixed seed (42)
    """
    return np.random.default_rng(42)


@pytest.fixture
def tolerance() -> float:
    """
    Numerical tolerance for floating point comparisons.
    
    Returns:
        Tolerance value (1e-8)
    """
    return TOLERANCE


@pytest.fixture
def prior_config() -> PriorConfig:
    """
    Default prior configuration for testing.
    
    Returns:
        PriorConfig with default values
    """
    return PriorConfig()


@pytest.fixture
def mcmc_config() -> MCMCConfig:
    """
    Reduced MCMC configuration for fast testing.
    
    Uses smaller iteration counts to keep tests fast.
    
    Returns:
        MCMCConfig with reduced iteration counts
    """
    config = MCMCConfig()
    # Reduce iterations for faster tests
    config.pilot_n_iter = 500
    config.pilot_n_burn = 200
    config.prod_n_burn = 500
    config.prod_n_samples = 1000
    return config


@pytest.fixture
def fourier_config() -> FourierConfig:
    """
    Default Fourier configuration for testing.
    
    Returns:
        FourierConfig with default values
    """
    return FourierConfig()


@pytest.fixture
def at_the_money_strike() -> float:
    """
    At-the-money strike price (equal to spot).
    
    Returns:
        Strike price equal to spot (100.0)
    """
    return S0_TEST


@pytest.fixture
def out_of_the_money_strike() -> float:
    """
    Out-of-the-money strike price.
    
    Returns:
        Strike price above spot (120.0)
    """
    return 120.0


@pytest.fixture
def in_the_money_strike() -> float:
    """
    In-the-money strike price.
    
    Returns:
        Strike price below spot (80.0)
    """
    return 80.0


# =============================================================================
# Utility Functions for Tests
# =============================================================================

def assert_params_close(
    params1: HestonParams,
    params2: HestonParams,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Assert that two HestonParams objects are close.
    
    Args:
        params1: First parameter set
        params2: Second parameter set
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Raises:
        AssertionError: If parameters differ beyond tolerance
    """
    assert abs(params1.kappa - params2.kappa) < atol + rtol * abs(params2.kappa)
    assert abs(params1.theta - params2.theta) < atol + rtol * abs(params2.theta)
    assert abs(params1.xi - params2.xi) < atol + rtol * abs(params2.xi)
    assert abs(params1.rho - params2.rho) < atol + rtol * abs(params2.rho)
    assert abs(params1.v0 - params2.v0) < atol + rtol * abs(params2.v0)


def assert_prices_close(
    prices1: NDArray[np.float64],
    prices2: NDArray[np.float64],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Assert that two price arrays are close.
    
    Args:
        prices1: First price array
        prices2: Second price array
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Raises:
        AssertionError: If prices differ beyond tolerance
    """
    assert np.allclose(prices1, prices2, rtol=rtol, atol=atol)


def generate_synthetic_prices(
    params: HestonParams,
    strikes: NDArray[np.float64],
    S0: float = S0_TEST,
    r: float = R_TEST,
    T: float = T_TEST,
) -> NDArray[np.float64]:
    """
    Generate synthetic option prices for given parameters.
    
    Args:
        params: Heston parameters
        strikes: Strike prices
        S0: Spot price
        r: Risk-free rate
        T: Time to maturity
    
    Returns:
        Array of option prices
    """
    from src.pricing import heston_call_prices
    
    return heston_call_prices(
        S0=S0,
        strikes=strikes,
        r=r,
        T=T,
        kappa=params.kappa,
        theta=params.theta,
        xi=params.xi,
        rho=params.rho,
        v0=params.v0,
    )