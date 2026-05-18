#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions for Heston Calibration
=========================================

This module provides helper functions for:
    - Random number generation and seeding
    - Numerical stability utilities
    - Timing decorators
    - Data validation
    - Logging configuration
    - File I/O helpers
    - Parameter transformations

These utilities are used throughout the calibration pipeline to ensure
reproducibility, numerical stability, and clean code organisation.

Author: Rodrigo Antonio Salado Ferrero
Repository: github.com/rodrigo-salado/heston-mcmc-calibration
License: MIT
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, Any, Callable, Tuple, Union
import json
import yaml
import logging
import time
import os
from functools import wraps
from datetime import datetime
import warnings


# =============================================================================
# Random Number Generation
# =============================================================================

def set_global_seed(seed: int = 42) -> np.random.Generator:
    """
    Set global random seed for reproducibility.
    
    Sets seeds for:
        - NumPy random generator
        - Python's random module
        - (Optionally) PyMC and TensorFlow if available
    
    Args:
        seed: Random seed (default: 42)
    
    Returns:
        NumPy random Generator object
    
    Example:
        >>> rng = set_global_seed(42)
        >>> samples = rng.normal(0, 1, 100)
    
    Note:
        Using a fixed seed ensures identical results across runs,
        essential for debugging and validation.
    """
    # NumPy random generator
    rng = np.random.default_rng(seed)
    
    # Python's random module
    import random
    random.seed(seed)
    
    # Optional: PyMC
    try:
        import pymc as pm
        pm.set_random_seed(seed)
    except ImportError:
        pass
    
    # Optional: TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    return rng


# =============================================================================
# Timing Utilities
# =============================================================================

def timer(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function with timing
    
    Example:
        >>> @timer
        ... def expensive_function():
        ...     time.sleep(1)
        >>> expensive_function()
        [INFO] expensive_function completed in 1.00 seconds
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        logger = logging.getLogger(__name__)
        logger.info(f"{func.__name__} completed in {elapsed:.2f} seconds")
        
        return result
    return wrapper


class Timer:
    """
    Context manager for timing code blocks.
    
    Example:
        >>> with Timer("Heavy computation"):
        ...     result = expensive_function()
        Heavy computation took 1.23 seconds
    """
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialise timer.
        
        Args:
            name: Name of the operation being timed
            verbose: Print timing if True
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"{self.name} took {self.elapsed:.2f} seconds")
    
    def get_elapsed(self) -> float:
        """Return elapsed time in seconds."""
        return self.elapsed


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the calibration pipeline.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Path to log file (optional)
        format_str: Custom log format string
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logging(level="DEBUG", log_file="calibration.log")
        >>> logger.info("Starting calibration...")
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[]
    )
    
    # Create logger
    logger = logging.getLogger("heston_calibration")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Numerical Stability Utilities
# =============================================================================

def safe_log(x: NDArray[np.float64], eps: float = 1e-100) -> NDArray[np.float64]:
    """
    Compute log(x) safely, avoiding log(0) by clipping.
    
    Args:
        x: Input array
        eps: Minimum value (default: 1e-100)
    
    Returns:
        log(max(x, eps))
    
    Example:
        >>> x = np.array([0, 1e-50, 1e-10, 1.0])
        >>> safe_log(x)
        array([-230.26, -115.13, -23.03, 0.])
    """
    x_clipped = np.maximum(x, eps)
    return np.log(x_clipped)


def safe_exp(x: NDArray[np.float64], max_exp: float = 700.0) -> NDArray[np.float64]:
    """
    Compute exp(x) safely, clipping to avoid overflow.
    
    Args:
        x: Input array
        max_exp: Maximum exponent to prevent overflow (default: 700)
    
    Returns:
        exp(min(x, max_exp))
    
    Note:
        exp(709) is ~8e307, near double precision limit.
        We use 700 as a conservative bound.
    """
    x_clipped = np.minimum(x, max_exp)
    return np.exp(x_clipped)


def log_sum_exp(log_probs: NDArray[np.float64]) -> float:
    """
    Compute log(sum(exp(log_probs))) in a numerically stable way.
    
    This function prevents underflow/overflow when summing probabilities
    in log space using the log-sum-exp trick:
        log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    
    Args:
        log_probs: Array of log probabilities
    
    Returns:
        log of the sum of probabilities
    
    Example:
        >>> log_probs = np.array([-1000, -1001, -999])
        >>> log_sum_exp(log_probs)
        -998.999...
    """
    max_log = np.max(log_probs)
    return max_log + np.log(np.sum(np.exp(log_probs - max_log)))


def softmax(x: NDArray[np.float64], temperature: float = 1.0) -> NDArray[np.float64]:
    """
    Compute softmax with temperature parameter.
    
    Softmax is used for weight normalisation:
        p_i = exp(x_i / τ) / sum(exp(x_j / τ))
    
    Args:
        x: Input array
        temperature: Temperature parameter (τ > 0). Higher τ → more uniform.
    
    Returns:
        Softmax probabilities
    
    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> softmax(x, temperature=1.0)
        array([0.09, 0.24, 0.67])
    """
    x_scaled = x / temperature
    exp_x = np.exp(x_scaled - np.max(x_scaled))
    return exp_x / np.sum(exp_x)


# =============================================================================
# Parameter Transformations
# =============================================================================

def transform_to_unbounded(params: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform bounded parameters to unbounded space for optimisation.
    
    Uses logit transformation for parameters in (0,1) and log for positive.
    
    Transformations:
        κ, θ, ξ, v₀ → log(parameter) (positive → ℝ)
        ρ → logit((ρ + 1)/2) ([-1, 1] → ℝ)
    
    Args:
        params: Array of shape (5,) = [κ, θ, ξ, ρ, v₀] in original space
    
    Returns:
        Array of shape (5,) in unbounded space
    
    Example:
        >>> params = np.array([2.0, 0.04, 0.3, -0.7, 0.04])
        >>> transformed = transform_to_unbounded(params)
        >>> print(transformed)
        [0.6931, -3.2189, -1.2040, -1.7346, -3.2189]
    """
    kappa, theta, xi, rho, v0 = params
    
    # Positive parameters -> log space
    kappa_unb = np.log(kappa)
    theta_unb = np.log(theta)
    xi_unb = np.log(xi)
    v0_unb = np.log(v0)
    
    # Rho: [-1, 1] -> logit space
    rho_norm = (rho + 1) / 2  # Map to [0, 1]
    rho_unb = np.log(rho_norm / (1 - rho_norm))  # Logit
    
    return np.array([kappa_unb, theta_unb, xi_unb, rho_unb, v0_unb])


def transform_from_unbounded(params_unb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform parameters from unbounded back to original space.
    
    Inverse of transform_to_unbounded.
    
    Args:
        params_unb: Array of shape (5,) in unbounded space
    
    Returns:
        Array of shape (5,) in original parameter space
    
    Example:
        >>> params_unb = np.array([0.6931, -3.2189, -1.2040, -1.7346, -3.2189])
        >>> params = transform_from_unbounded(params_unb)
        >>> print(params)
        [2.0, 0.04, 0.3, -0.7, 0.04]
    """
    kappa_unb, theta_unb, xi_unb, rho_unb, v0_unb = params_unb
    
    # Log space -> positive
    kappa = np.exp(kappa_unb)
    theta = np.exp(theta_unb)
    xi = np.exp(xi_unb)
    v0 = np.exp(v0_unb)
    
    # Logit space -> [-1, 1]
    rho_norm = 1 / (1 + np.exp(-rho_unb))  # Sigmoid
    rho = 2 * rho_norm - 1  # Map back to [-1, 1]
    
    return np.array([kappa, theta, xi, rho, v0])


# =============================================================================
# Data Validation
# =============================================================================

def validate_price_surface(
    strikes: NDArray[np.float64],
    prices: NDArray[np.float64],
    S0: float,
    r: float,
    T: float,
) -> Tuple[bool, str]:
    """
    Validate option price surface for no-arbitrage conditions.
    
    Checks:
        1. Monotonicity: Call prices decrease with strike
        2. Convexity: Call prices are convex in strike
        3. No arbitrage bounds: intrinsic value ≤ price ≤ spot
        4. Put-call parity (if puts available)
    
    Args:
        strikes: Strike prices
        prices: Call option prices
        S0: Spot price
        r: Risk-free rate
        T: Time to maturity
    
    Returns:
        Tuple of (is_valid, message)
    
    Example:
        >>> is_valid, msg = validate_price_surface(strikes, prices, 100.0, 0.03, 1.0)
        >>> if not is_valid:
        ...     print(f"Validation failed: {msg}")
    """
    # Check monotonicity
    for i in range(len(prices) - 1):
        if prices[i] < prices[i + 1]:
            return False, f"Prices not monotonic: price[{i}]={prices[i]:.4f} < price[{i+1}]={prices[i+1]:.4f}"
    
    # Check convexity (second differences non-negative).
    # For call prices convex in strike K, the discrete second difference
    #   p[i-1] - 2*p[i] + p[i+1] >= 0
    # is equivalent to backward_diff >= forward_diff. The surface is therefore
    # *non-convex* (and arbitrageable) only when forward_diff exceeds
    # backward_diff. A small tolerance absorbs numerical integration noise.
    for i in range(1, len(prices) - 1):
        forward_diff = prices[i] - prices[i + 1]
        backward_diff = prices[i - 1] - prices[i]
        if forward_diff > backward_diff + 1e-6:
            return False, f"Prices not convex at strike {strikes[i]}"
    
    # Check no-arbitrage bounds
    for K, price in zip(strikes, prices):
        intrinsic = max(S0 - K * np.exp(-r * T), 0)
        if price < intrinsic - 1e-6:
            return False, f"Price {price:.4f} below intrinsic {intrinsic:.4f} at K={K}"
        if price > S0 + 1e-6:
            return False, f"Price {price:.4f} above spot {S0} at K={K}"
    
    return True, "Price surface passes all validation checks"


# =============================================================================
# File I/O Utilities
# =============================================================================

def save_config(config: Dict[str, Any], filepath: str, format: str = "yaml") -> None:
    """
    Save configuration to file in YAML or JSON format.
    
    Args:
        config: Configuration dictionary
        filepath: Output file path
        format: Format ('yaml' or 'json')
    
    Example:
        >>> config = {"seed": 42, "n_iter": 10000}
        >>> save_config(config, "config.yaml", format="yaml")
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == "yaml":
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif format == "json":
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        filepath: Path to config file
    
    Returns:
        Configuration dictionary
    
    Example:
        >>> config = load_config("config.yaml")
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    elif ext == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def ensure_directory(path: str) -> None:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
    
    Example:
        >>> ensure_directory("results/figures/trace_plots")
    """
    os.makedirs(path, exist_ok=True)


# =============================================================================
# Progress Bar
# =============================================================================

class ProgressBar:
    """
    Simple progress bar for long-running operations.
    
    Example:
        >>> pb = ProgressBar(total=100, desc="Processing")
        >>> for i in range(100):
        ...     pb.update(1)
        ...     time.sleep(0.01)
        Processing: 100%|████████████| 100/100 [00:01<00:00, 89.2it/s]
    """
    
    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        width: int = 50,
        verbose: bool = True,
    ):
        """
        Initialise progress bar.
        
        Args:
            total: Total number of iterations
            desc: Description text
            width: Bar width in characters
            verbose: Display progress if True
        """
        self.total = total
        self.desc = desc
        self.width = width
        self.verbose = verbose
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1) -> None:
        """
        Update progress by n steps.
        
        Args:
            n: Number of steps completed
        """
        if not self.verbose:
            return
        
        self.current += n
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
        else:
            eta = 0
        
        # Format time strings
        def format_time(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                return f"{seconds / 60:.1f}m"
            else:
                return f"{seconds / 3600:.1f}h"
        
        elapsed_str = format_time(elapsed)
        eta_str = format_time(eta)
        
        print(f"\r{self.desc}: {percent:3.0%}|{bar}| {self.current}/{self.total} "
              f"[{elapsed_str}<{eta_str}, {self.current / elapsed:.1f}it/s]",
              end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def close(self) -> None:
        """Close progress bar (called automatically when complete)."""
        if self.verbose and self.current < self.total:
            self.update(self.total - self.current)


# =============================================================================
# Parameter Summary
# =============================================================================

def format_parameter_summary(
    samples: NDArray[np.float64],
    param_names: Tuple[str, ...] = ("kappa", "theta", "xi", "rho", "v0"),
    true_params: Optional[Dict[str, float]] = None,
) -> str:
    """
    Format parameter summary as formatted string.
    
    Args:
        samples: MCMC samples (n_samples, n_params)
        param_names: Parameter names
        true_params: Optional ground truth values
    
    Returns:
        Formatted summary string
    
    Example:
        >>> summary = format_parameter_summary(samples, true_params=true_dict)
        >>> print(summary)
    """
    lines = []
    lines.append("=" * 80)
    lines.append("HESTON PARAMETER SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    if true_params:
        lines.append(f"{'Param':>8} {'True':>10} {'Mean':>10} {'Std':>10} {'2.5%':>10} {'97.5%':>10}")
    else:
        lines.append(f"{'Param':>8} {'Mean':>10} {'Std':>10} {'2.5%':>10} {'97.5%':>10}")
    
    lines.append("-" * 80)
    
    for i, name in enumerate(param_names):
        chain = samples[:, i]
        mean = np.mean(chain)
        std = np.std(chain)
        q2_5 = np.quantile(chain, 0.025)
        q97_5 = np.quantile(chain, 0.975)
        
        if true_params and name in true_params:
            true_val = true_params[name]
            lines.append(f"{name:>8} {true_val:10.4f} {mean:10.4f} {std:10.4f} {q2_5:10.4f} {q97_5:10.4f}")
        else:
            lines.append(f"{name:>8} {mean:10.4f} {std:10.4f} {q2_5:10.4f} {q97_5:10.4f}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Random number generation
    "set_global_seed",
    # Timing utilities
    "timer",
    "Timer",
    # Logging
    "setup_logging",
    # Numerical stability
    "safe_log",
    "safe_exp",
    "log_sum_exp",
    "softmax",
    # Parameter transformations
    "transform_to_unbounded",
    "transform_from_unbounded",
    # Data validation
    "validate_price_surface",
    # File I/O
    "save_config",
    "load_config",
    "ensure_directory",
    # Progress bar
    "ProgressBar",
    # Formatting
    "format_parameter_summary",
]