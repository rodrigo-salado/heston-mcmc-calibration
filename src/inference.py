#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Inference for Heston Model
====================================

This module implements the Bayesian inference framework for calibrating
the Heston model to option prices using Markov Chain Monte Carlo (MCMC).

COMPONENTS:
    1. Prior Distributions: Mildly informative priors with domain constraints
    2. Likelihood Function: Gaussian observation noise on option prices
    3. Posterior: Unnormalised log-posterior = log prior + log likelihood
    4. MCMC Sampler: Random-walk Metropolis-Hastings with adaptive proposal
    
THEORY:
    Bayes' theorem for calibration:
    
        p(θ | data) ∝ p(data | θ) p(θ)
    
    where:
        θ = (κ, θ, ξ, ρ, v₀) are the Heston parameters
        p(θ) are the prior distributions
        p(data | θ) is the likelihood
    
    Likelihood (Gaussian observation errors):
        
        p(P_market | θ) = ∏_{i=1}^N (1/√(2πσ²)) exp[-(P_i^market - P_i^model(θ))²/(2σ²)]
    
    Log-likelihood (ignoring constants):
        
        ℓ(θ) = -½ ∑ (P_i^market - P_i^model(θ))² / σ²

ALGORITHM - Adaptive Metropolis-Hastings:
    
    1. Run a pilot chain with diagonal proposal covariance
    2. Estimate posterior covariance from pilot samples
    3. Scale covariance by 2.38²/d (Roberts-Gelman-Gilks optimal scaling)
    4. Run production chain with tuned multivariate Gaussian proposal
    
    Proposal: θ_proposed = θ_current + chol(Σ) * z,  z ~ N(0, I)
    
    Acceptance probability:
        
        α = min(1, exp(log p(θ_proposed) - log p(θ_current)))
    
    Optimal acceptance rate for 5D Gaussian target: ~0.234

REFERENCE:
    Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). "Weak convergence
    and optimal scaling of random walk Metropolis algorithms".
    The Annals of Applied Probability, 7(1), 110-120.

Author: Rodrigo Antonio Salado Ferrero
Repository: github.com/rodrigo-salado/heston-mcmc-calibration
License: MIT
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Callable, Dict, Any
from scipy.stats import norm, uniform, lognorm
import time
import warnings

# Import from local modules
from src.types import (
    HestonParams,
    MarketData,
    CalibrationResult,
    PriorConfig,
    MCMCConfig,
    FourierConfig,
    CalibrationConfig,
)
from src.pricing import heston_call_prices


# =============================================================================
# Prior Distributions
# =============================================================================

class LogPrior:
    """
    Log-prior density for Heston parameters.
    
    This class implements mildly informative priors that provide
    regularisation while allowing the data to dominate. Each parameter
    has domain constraints and a specified distribution.
    
    Prior choices:
        κ (kappa): Log-normal with μ=log(2), σ=0.5
                   Mean reversion is positive and typically O(1)
        
        θ (theta): Truncated Normal with μ=0.04, σ=0.02
                   Long-run variance ≈ 4% (≈20% vol) for equities
        
        ξ (xi): Truncated Normal with μ=0.30, σ=0.15
                Typical vol-of-vol range for equity indices
        
        ρ (rho): Uniform on (-0.999, 0.999)
                 Allows negative correlation (leverage effect)
        
        v₀ (v0): Truncated Normal with μ=0.04, σ=0.02
                 Initial variance close to long-run mean
    
    Attributes:
        config: PriorConfig object with distribution parameters
        param_names: List of parameter names ['kappa', 'theta', 'xi', 'rho', 'v0']
    
    Example:
        >>> prior = LogPrior()
        >>> params = np.array([2.0, 0.04, 0.3, -0.7, 0.04])
        >>> lp = prior(params)
        >>> print(f"{lp:.4f}")
        -0.1234
    """
    
    def __init__(self, config: Optional[PriorConfig] = None):
        """
        Initialise log-prior with optional custom configuration.
        
        Args:
            config: PriorConfig object (uses defaults if None)
        """
        self.config = config if config is not None else PriorConfig()
        self.param_names = ["kappa", "theta", "xi", "rho", "v0"]
        
        # Pre-compute log-prior constants for efficiency
        self._cache = {}
    
    def _log_prior_kappa(self, kappa: float) -> float:
        """
        Log-prior density for κ (mean reversion speed).
        
        Uses log-normal distribution to ensure positivity.
        
        Args:
            kappa: Mean reversion speed (> 0)
        
        Returns:
            Log-prior density (or -inf if out of bounds)
        """
        # Hard box constraint
        if kappa <= self.config.kappa_bounds[0] or kappa > self.config.kappa_bounds[1]:
            return -np.inf
        
        # Log-normal prior (natural for positive parameters)
        if self.config.kappa_dist == "lognormal":
            log_kappa = np.log(kappa)
            mu_log = self.config.kappa_mu_log
            sigma_log = self.config.kappa_sigma_log
            return lognorm.logpdf(kappa, s=sigma_log, scale=np.exp(mu_log))
        
        # Fallback: uniform on log scale
        return 0.0
    
    def _log_prior_theta(self, theta: float) -> float:
        """
        Log-prior density for θ (long-run variance).
        
        Uses truncated normal distribution bounded away from zero.
        
        Args:
            theta: Long-run variance (> 0)
        
        Returns:
            Log-prior density (or -inf if out of bounds)
        """
        # Hard box constraint
        if theta <= self.config.theta_bounds[0] or theta > self.config.theta_bounds[1]:
            return -np.inf
        
        # Normal prior (truncated by bounds)
        if self.config.theta_dist == "normal":
            # Manual truncation (bounds already checked)
            mu = self.config.theta_mu
            sigma = self.config.theta_sigma
            return norm.logpdf(theta, loc=mu, scale=sigma)
        
        # Fallback: uniform
        return 0.0
    
    def _log_prior_xi(self, xi: float) -> float:
        """
        Log-prior density for ξ (volatility of volatility).
        
        Uses truncated normal distribution.
        
        Args:
            xi: Volatility of volatility (> 0)
        
        Returns:
            Log-prior density (or -inf if out of bounds)
        """
        # Hard box constraint
        if xi <= self.config.xi_bounds[0] or xi > self.config.xi_bounds[1]:
            return -np.inf
        
        # Normal prior
        if self.config.xi_dist == "normal":
            mu = self.config.xi_mu
            sigma = self.config.xi_sigma
            return norm.logpdf(xi, loc=mu, scale=sigma)
        
        return 0.0
    
    def _log_prior_rho(self, rho: float) -> float:
        """
        Log-prior density for ρ (correlation).
        
        Uses uniform distribution on (-0.999, 0.999) to avoid boundary issues.
        
        Args:
            rho: Correlation coefficient ∈ (-1, 1)
        
        Returns:
            Log-prior density (0 if in bounds, -inf otherwise)
        """
        # Hard bounds with small margin for numerical stability
        if rho <= self.config.rho_bounds[0] or rho >= self.config.rho_bounds[1]:
            return -np.inf
        
        # Uniform prior (constant density within bounds)
        # The constant cancels in Metropolis ratio
        return 0.0
    
    def _log_prior_v0(self, v0: float) -> float:
        """
        Log-prior density for v₀ (initial variance).
        
        Uses truncated normal distribution.
        
        Args:
            v0: Initial variance (> 0)
        
        Returns:
            Log-prior density (or -inf if out of bounds)
        """
        # Hard box constraint
        if v0 <= self.config.v0_bounds[0] or v0 > self.config.v0_bounds[1]:
            return -np.inf
        
        # Normal prior
        if self.config.v0_dist == "normal":
            mu = self.config.v0_mu
            sigma = self.config.v0_sigma
            return norm.logpdf(v0, loc=mu, scale=sigma)
        
        return 0.0
    
    def __call__(self, params: NDArray[np.float64]) -> float:
        """
        Compute total log-prior density for all parameters.
        
        Args:
            params: Array of shape (5,) = [κ, θ, ξ, ρ, v₀]
        
        Returns:
            Log-prior density (sum of individual logs)
        
        Raises:
            ValueError: If params shape is not (5,)
        """
        if len(params) != 5:
            raise ValueError(f"Expected 5 parameters, got {len(params)}")
        
        kappa, theta, xi, rho, v0 = params
        
        # Sum individual log-priors
        lp = (
            self._log_prior_kappa(kappa)
            + self._log_prior_theta(theta)
            + self._log_prior_xi(xi)
            + self._log_prior_rho(rho)
            + self._log_prior_v0(v0)
        )
        
        return lp


# =============================================================================
# Likelihood Function
# =============================================================================

class LogLikelihood:
    """
    Log-likelihood for observed option prices given Heston parameters.
    
    Assumes Gaussian observation errors:
        P_i^market = P_i^model(θ) + ε_i,  ε_i ~ N(0, σ²)
    
    The likelihood is:
        L(θ) = ∏_{i=1}^N N(P_i^market | P_i^model(θ), σ²)
    
    Log-likelihood (ignoring constants):
        ℓ(θ) = -½ ∑ (P_i^market - P_i^model(θ))² / σ²
    
    Attributes:
        market_data: MarketData object with observed prices
        sigma_noise: Standard deviation of observation noise
        fourier_config: Configuration for Fourier integration
    
    Example:
        >>> market = MarketData(strikes=prices, spot=100.0, ...)
        >>> likelihood = LogLikelihood(market, sigma_noise=0.05)
        >>> params = np.array([2.0, 0.04, 0.3, -0.7, 0.04])
        >>> ll = likelihood(params)
        >>> print(f"{ll:.2f}")
        -123.45
    """
    
    def __init__(
        self,
        market_data: MarketData,
        sigma_noise: float,
        fourier_config: Optional[FourierConfig] = None,
    ):
        """
        Initialise log-likelihood.
        
        Args:
            market_data: Market data (strikes, prices, S0, r, T)
            sigma_noise: Standard deviation of observation noise
            fourier_config: Configuration for Fourier pricing
        """
        self.market_data = market_data
        self.sigma_noise = sigma_noise
        self.fourier_config = fourier_config if fourier_config else FourierConfig()
        
        # Pre-compute constants
        self._inv_sigma2 = 1.0 / (sigma_noise ** 2)
        self._log_const = -0.5 * len(market_data.prices) * np.log(2 * np.pi * sigma_noise ** 2)
    
    def _compute_model_prices(
        self,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        v0: float,
    ) -> NDArray[np.float64]:
        """
        Compute model prices for all strikes.
        
        Wrapper around heston_call_prices with pre-configured parameters.
        
        Args:
            kappa, theta, xi, rho, v0: Heston parameters
        
        Returns:
            Array of model prices for each strike
        """
        return heston_call_prices(
            S0=self.market_data.spot,
            strikes=self.market_data.strikes,
            r=self.market_data.rate,
            T=self.market_data.tenor,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            v0=v0,
            u_max=self.fourier_config.u_max,
            n_grid=self.fourier_config.n_grid,
            method=self.fourier_config.integration_method,
        )
    
    def __call__(self, params: NDArray[np.float64]) -> float:
        """
        Compute log-likelihood for given parameters.
        
        Args:
            params: Array of shape (5,) = [κ, θ, ξ, ρ, v₀]
        
        Returns:
            Log-likelihood (or -inf if pricing fails)
        """
        kappa, theta, xi, rho, v0 = params
        
        # Attempt to price options
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                model_prices = self._compute_model_prices(kappa, theta, xi, rho, v0)
        except Exception:
            # Pricing failed (e.g., numerical instability)
            return -np.inf
        
        # Check for invalid prices
        if not np.all(np.isfinite(model_prices)):
            return -np.inf
        
        # Compute squared errors
        residuals = self.market_data.prices - model_prices
        squared_errors = residuals ** 2
        
        # Gaussian log-likelihood
        log_likelihood = -0.5 * np.sum(squared_errors * self._inv_sigma2)
        
        return log_likelihood


# =============================================================================
# Posterior Distribution
# =============================================================================

class LogPosterior:
    """
    Unnormalised log-posterior density: log p(θ | data).
    
    By Bayes' theorem:
        log p(θ | data) = log p(data | θ) + log p(θ) + constant
    
    The constant cancels in Metropolis-Hastings acceptance ratio.
    
    Attributes:
        log_prior: LogPrior instance
        log_likelihood: LogLikelihood instance
    
    Example:
        >>> posterior = LogPosterior(prior, likelihood)
        >>> params = np.array([2.0, 0.04, 0.3, -0.7, 0.04])
        >>> lp = posterior(params)
        >>> print(f"{lp:.4f}")
        -123.57
    """
    
    def __init__(self, log_prior: LogPrior, log_likelihood: LogLikelihood):
        """
        Initialise log-posterior.
        
        Args:
            log_prior: LogPrior instance
            log_likelihood: LogLikelihood instance
        """
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
    
    def __call__(self, params: NDArray[np.float64]) -> float:
        """
        Compute unnormalised log-posterior.
        
        Args:
            params: Array of shape (5,) = [κ, θ, ξ, ρ, v₀]
        
        Returns:
            Log-posterior = log_prior + log_likelihood
        """
        lp = self.log_prior(params)
        
        # If prior is invalid, skip likelihood evaluation
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(params)
        
        # If likelihood is invalid, return -inf
        if not np.isfinite(ll):
            return -np.inf
        
        return lp + ll


# =============================================================================
# Metropolis-Hastings Sampler
# =============================================================================

class MetropolisHastingsSampler:
    """
    Random-walk Metropolis-Hastings MCMC sampler.
    
    Proposes new states using a multivariate Gaussian distribution:
        θ_proposed = θ_current + chol(Σ) · z,  z ~ N(0, I)
    
    Acceptance probability:
        α = min(1, exp(log p(θ_proposed) - log p(θ_current)))
    
    Attributes:
        log_posterior: Callable that returns log posterior density
        proposal_cov: Proposal covariance matrix Σ
        rng: NumPy random number generator
    
    Example:
        >>> sampler = MetropolisHastingsSampler(posterior, proposal_cov)
        >>> samples, log_ps, acceptance = sampler.run(theta0, n_iter=10000)
    """
    
    def __init__(
        self,
        log_posterior: Callable[[NDArray[np.float64]], float],
        proposal_cov: NDArray[np.float64],
        random_seed: Optional[int] = None,
    ):
        """
        Initialise MCMC sampler.
        
        Args:
            log_posterior: Function that returns log posterior density
            proposal_cov: Proposal covariance matrix (5x5)
            random_seed: Random seed for reproducibility
        """
        self.log_posterior = log_posterior
        self.proposal_cov = proposal_cov
        self.rng = np.random.default_rng(random_seed)
        
        # Pre-compute Cholesky factor for efficient sampling
        # Add small diagonal regularisation for numerical stability
        n = proposal_cov.shape[0]
        reg_cov = proposal_cov + 1e-12 * np.eye(n)
        self._chol = np.linalg.cholesky(reg_cov)
    
    def _propose(self, theta_current: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Generate proposal from multivariate Gaussian.
        
        Args:
            theta_current: Current parameter vector (shape 5,)
        
        Returns:
            Proposed parameter vector
        """
        z = self.rng.standard_normal(len(theta_current))
        return theta_current + self._chol @ z
    
    def run(
        self,
        theta0: NDArray[np.float64],
        n_iter: int,
        verbose: bool = True,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """
        Run MCMC chain.
        
        Args:
            theta0: Initial parameter vector (shape 5,)
            n_iter: Number of MCMC iterations
            verbose: Print progress if True
        
        Returns:
            Tuple of (samples, log_posterior_values, acceptance_rate)
            - samples: Array of shape (n_iter, 5)
            - log_posterior_values: Array of shape (n_iter,)
            - acceptance_rate: Float between 0 and 1
        
        Example:
            >>> sampler = MetropolisHastingsSampler(posterior, cov)
            >>> samples, log_ps, acc = sampler.run(theta0, 10000)
            >>> print(f"Acceptance rate: {acc:.1%}")
        """
        n_params = len(theta0)
        samples = np.zeros((n_iter, n_params))
        log_ps = np.zeros(n_iter)
        accepts = 0
        
        # Initial state
        theta_current = np.array(theta0, dtype=np.float64)
        lp_current = self.log_posterior(theta_current)
        
        # Progress tracking
        start_time = time.time()
        last_print = start_time
        
        for i in range(n_iter):
            # Propose new state
            theta_proposed = self._propose(theta_current)
            lp_proposed = self.log_posterior(theta_proposed)
            
            # Metropolis acceptance step
            log_alpha = lp_proposed - lp_current
            if np.log(self.rng.uniform()) < log_alpha:
                theta_current = theta_proposed
                lp_current = lp_proposed
                accepts += 1
            
            # Store state
            samples[i] = theta_current
            log_ps[i] = lp_current
            
            # Progress reporting
            if verbose and (i + 1) % (n_iter // 10) == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (n_iter - i - 1)
                print(f"  Iteration {i+1}/{n_iter} | "
                      f"Accept: {accepts/(i+1):.1%} | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"ETA: {eta:.1f}s")
        
        acceptance_rate = accepts / n_iter
        
        return samples, log_ps, acceptance_rate


# =============================================================================
# Adaptive Metropolis-Hastings
# =============================================================================

class AdaptiveMetropolisHastings:
    """
    Adaptive Metropolis-Hastings with covariance estimation from pilot chain.
    
    Strategy:
        1. Run a pilot chain with diagonal proposal to explore the posterior
        2. Estimate posterior covariance from pilot samples
        3. Scale covariance by 2.38²/d (Roberts-Gelman-Gilks optimal)
        4. Run production chain with tuned proposal
    
    This approach automatically adapts to parameter correlations and scales.
    
    Attributes:
        log_posterior: Callable that returns log posterior density
        rng: NumPy random number generator
        config: MCMCConfig with tuning parameters
    
    Example:
        >>> adaptive = AdaptiveMetropolisHastings(posterior, config)
        >>> result = adaptive.calibrate()
        >>> result.summary()
    
    Reference:
        Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). "Weak convergence
        and optimal scaling of random walk Metropolis algorithms".
        The Annals of Applied Probability, 7(1), 110-120.
    """
    
    def __init__(
        self,
        log_posterior: Callable[[NDArray[np.float64]], float],
        config: MCMCConfig,
    ):
        """
        Initialise adaptive MCMC sampler.
        
        Args:
            log_posterior: Function that returns log posterior density
            config: MCMCConfig with sampling parameters
        """
        self.log_posterior = log_posterior
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
    
    def _run_pilot_chain(
        self,
        verbose: bool = True,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Run pilot chain to estimate posterior covariance.
        
        Args:
            verbose: Print progress if True
        
        Returns:
            Tuple of (samples, log_posterior_values)
        """
        # Initial proposal: diagonal with pilot_proposal_scale
        proposal_std = self.config.pilot_proposal_scale * np.array(
            [0.15, 0.005, 0.025, 0.05, 0.005]  # Scales for each parameter
        )
        proposal_cov = np.diag(proposal_std ** 2)
        
        # Create sampler
        sampler = MetropolisHastingsSampler(
            self.log_posterior,
            proposal_cov,
            random_seed=self.config.random_seed,
        )
        
        # Run pilot chain
        if verbose:
            print(f"Running pilot chain: {self.config.pilot_n_iter} iterations")
        samples, log_ps, acceptance = sampler.run(
            np.array(self.config.pilot_initial),
            self.config.pilot_n_iter,
            verbose=verbose,
        )
        if verbose:
            print(f"  Pilot acceptance rate: {acceptance:.1%}")
        
        return samples, log_ps
    
    def _estimate_covariance(
        self,
        pilot_samples: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Estimate posterior covariance from pilot samples.
        
        Args:
            pilot_samples: Pilot chain samples (n_iter, 5)
        
        Returns:
            Estimated covariance matrix (5x5)
        """
        # Discard burn-in
        burned = pilot_samples[self.config.pilot_n_burn:]
        
        # Empirical covariance
        emp_cov = np.cov(burned, rowvar=False)
        
        # Apply Roberts-Gelman-Gilks optimal scaling
        tuned_cov = self.config.adaptation_scale * emp_cov
        
        return tuned_cov
    
    def _run_production_chain(
        self,
        theta0: NDArray[np.float64],
        proposal_cov: NDArray[np.float64],
        verbose: bool = True,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """
        Run production chain with tuned proposal.
        
        Args:
            theta0: Initial parameter vector
            proposal_cov: Tuned proposal covariance
            verbose: Print progress if True
        
        Returns:
            Tuple of (samples, log_posterior_values, acceptance_rate)
        """
        # Burn-in phase
        if verbose:
            print(f"Burn-in: {self.config.prod_n_burn} iterations")
        sampler_burn = MetropolisHastingsSampler(
            self.log_posterior,
            proposal_cov,
            random_seed=self.config.random_seed,
        )
        burn_samples, _, burn_acc = sampler_burn.run(
            theta0,
            self.config.prod_n_burn,
            verbose=verbose,
        )
        if verbose:
            print(f"  Burn-in acceptance: {burn_acc:.1%}")
        
        # Sampling phase
        if verbose:
            print(f"Sampling: {self.config.prod_n_samples} iterations")
        sampler_prod = MetropolisHastingsSampler(
            self.log_posterior,
            proposal_cov,
            random_seed=self.config.random_seed + 1,
        )
        samples, log_ps, acc = sampler_prod.run(
            burn_samples[-1],
            self.config.prod_n_samples,
            verbose=verbose,
        )
        if verbose:
            print(f"  Sampling acceptance: {acc:.1%}")
        
        return samples, log_ps, acc
    
    def calibrate(
        self,
        true_params: Optional[HestonParams] = None,
        verbose: bool = True,
    ) -> CalibrationResult:
        """
        Run full adaptive MCMC calibration.
        
        Args:
            true_params: Optional ground truth for synthetic data
            verbose: Print progress if True
        
        Returns:
            CalibrationResult with posterior samples and metadata
        """
        start_time = time.time()
        
        # Step 1: Pilot chain
        pilot_samples, _ = self._run_pilot_chain(verbose=verbose)
        
        # Step 2: Estimate covariance
        proposal_cov = self._estimate_covariance(pilot_samples)
        
        # Print estimated standard deviations
        if verbose:
            stds = np.sqrt(np.diag(proposal_cov) / self.config.adaptation_scale)
            print("\nEstimated posterior std (from pilot):")
            names = ["kappa", "theta", "xi", "rho", "v0"]
            for name, std in zip(names, stds):
                print(f"  {name:>6}: {std:.4f}")
            print()
        
        # Step 3: Production chain
        theta0 = pilot_samples[-1]  # Last pilot sample as starting point
        samples, log_ps, acceptance = self._run_production_chain(
            theta0, proposal_cov, verbose=verbose
        )
        
        # Compute runtime
        runtime = time.time() - start_time
        
        # Create result object
        result = CalibrationResult(
            samples=samples,
            log_posterior=log_ps,
            acceptance_rate=acceptance,
            true_params=true_params,
            runtime_seconds=runtime,
            metadata={
                "config": {
                    "pilot_n_iter": self.config.pilot_n_iter,
                    "pilot_n_burn": self.config.pilot_n_burn,
                    "prod_n_burn": self.config.prod_n_burn,
                    "prod_n_samples": self.config.prod_n_samples,
                    "random_seed": self.config.random_seed,
                }
            },
        )
        
        return result


# =============================================================================
# Main Calibration Function
# =============================================================================

def calibrate_heston(
    market_data: MarketData,
    prior_config: Optional[PriorConfig] = None,
    mcmc_config: Optional[MCMCConfig] = None,
    fourier_config: Optional[FourierConfig] = None,
    sigma_noise: float = 0.05,
    true_params: Optional[HestonParams] = None,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Main function for Bayesian calibration of the Heston model.
    
    This function orchestrates the entire calibration process:
        1. Set up prior and likelihood
        2. Initialise adaptive MCMC sampler
        3. Run calibration
        4. Return results
    
    Args:
        market_data: MarketData with observed option prices
        prior_config: Prior configuration (uses defaults if None)
        mcmc_config: MCMC configuration (uses defaults if None)
        fourier_config: Fourier configuration (uses defaults if None)
        sigma_noise: Standard deviation of observation noise
        true_params: Optional ground truth (for synthetic data validation)
        verbose: Print progress if True
    
    Returns:
        CalibrationResult with posterior samples and statistics
    
    Example:
        >>> from src.types import MarketData
        >>> import numpy as np
        >>> 
        >>> market = MarketData(
        ...     strikes=np.array([80, 90, 100, 110, 120]),
        ...     prices=np.array([23.69, 15.75, 9.24, 4.58, 1.84]),
        ...     spot=100.0,
        ...     rate=0.03,
        ...     tenor=1.0
        ... )
        >>> 
        >>> result = calibrate_heston(market)
        >>> result.summary()
    """
    if verbose:
        print("\n" + "=" * 70)
        print("HESTON MODEL BAYESIAN CALIBRATION")
        print("=" * 70)
        print(f"\nMarket data: {market_data.n_options} options")
        print(f"  Spot: {market_data.spot}")
        print(f"  Rate: {market_data.rate:.4f}")
        print(f"  Tenor: {market_data.tenor} years")
        print(f"  Strikes: [{market_data.strikes[0]:.0f}, ..., {market_data.strikes[-1]:.0f}]")
        print(f"Sigma noise: {sigma_noise:.3f}")
        print()
    
    # Set up prior and likelihood
    log_prior = LogPrior(prior_config)
    log_likelihood = LogLikelihood(market_data, sigma_noise, fourier_config)
    log_posterior = LogPosterior(log_prior, log_likelihood)
    
    # Set up adaptive MCMC
    mcmc_config = mcmc_config or MCMCConfig()
    adaptive_mcmc = AdaptiveMetropolisHastings(log_posterior, mcmc_config)
    
    # Run calibration
    result = adaptive_mcmc.calibrate(true_params=true_params, verbose=verbose)
    
    if verbose:
        print("\nCalibration complete!")
        result.summary()
    
    return result


# =============================================================================
# Diagnostic Functions
# =============================================================================

def compute_r_hat(
    chains: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute Gelman-Rubin R-hat convergence statistic.
    
    R-hat compares within-chain variance to between-chain variance.
    Values close to 1.0 indicate convergence (typically < 1.1).
    
    Formula:
        B = between-chain variance
        W = within-chain variance
        var_hat = (1 - 1/n)W + (1/n)B
        R_hat = sqrt(var_hat / W)
    
    Args:
        chains: Array of shape (n_chains, n_samples, n_params)
    
    Returns:
        R-hat for each parameter (shape n_params,)
    
    Reference:
        Gelman, A., & Rubin, D. B. (1992). "Inference from iterative
        simulation using multiple sequences". Statistical Science, 7(4), 457-472.
    """
    n_chains, n_samples, n_params = chains.shape
    
    # Within-chain variance
    chain_means = np.mean(chains, axis=1)  # (n_chains, n_params)
    chain_vars = np.var(chains, axis=1, ddof=1)  # (n_chains, n_params)
    W = np.mean(chain_vars, axis=0)  # (n_params,)
    
    # Between-chain variance
    grand_mean = np.mean(chain_means, axis=0)  # (n_params,)
    B = n_samples * np.var(chain_means, axis=0, ddof=1)  # (n_params,)
    
    # Pooled variance
    var_hat = (1 - 1/n_samples) * W + (1/n_samples) * B
    
    # R-hat
    r_hat = np.sqrt(var_hat / W)
    
    return r_hat


def compute_ess(
    samples: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute effective sample size (ESS) for MCMC chain.
    
    ESS estimates the number of independent samples in a correlated chain.
    
    Formula:
        ESS = n / (1 + 2 ∑_{k=1}^∞ ρ_k)
    
    where ρ_k is the autocorrelation at lag k.
    
    Args:
        samples: Array of shape (n_samples, n_params)
    
    Returns:
        ESS for each parameter (shape n_params,)
    
    Reference:
        Kass, R. E., Carlin, B. P., Gelman, A., & Neal, R. M. (1998).
        "Markov chain Monte Carlo in practice: a roundtable".
        The American Statistician, 52(2), 93-100.
    """
    n_samples, n_params = samples.shape
    
    ess = np.zeros(n_params)
    
    for j in range(n_params):
        chain = samples[:, j]
        
        # Compute autocorrelation
        max_lag = min(n_samples // 4, 100)
        autocorr = np.correlate(chain - np.mean(chain), chain - np.mean(chain), mode='full')
        autocorr = autocorr[autocorr.size // 2:] / autocorr[autocorr.size // 2]
        
        # Compute ESS (truncate at first negative autocorrelation)
        sum_rho = 0
        for k in range(1, max_lag):
            if autocorr[k] <= 0:
                break
            sum_rho += autocorr[k]
        
        ess[j] = n_samples / (1 + 2 * sum_rho)
    
    return ess


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Prior and likelihood
    "LogPrior",
    "LogLikelihood",
    "LogPosterior",
    # MCMC samplers
    "MetropolisHastingsSampler",
    "AdaptiveMetropolisHastings",
    # Main calibration
    "calibrate_heston",
    # Diagnostics
    "compute_r_hat",
    "compute_ess",
]