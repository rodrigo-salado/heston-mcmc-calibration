#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type Definitions for Heston Calibration
========================================

This module defines all data structures and type hints used throughout
the Heston calibration package. Using dataclasses ensures immutability
where appropriate and provides clear documentation of the data schema.

The type hints enable static type checking with mypy and improve IDE
autocompletion and error detection.

THEORY:
    The Heston model dynamics under the risk-neutral measure Q are:
    
        dS_t = r S_t dt + sqrt(v_t) S_t dW_t^1
        dv_t = κ (θ - v_t) dt + ξ sqrt(v_t) dW_t^2
        dW_t^1 dW_t^2 = ρ dt
    
    where:
        S_t: Asset price
        v_t: Instantaneous variance
        r: Risk-free rate
        κ (kappa): Mean reversion speed (positive)
        θ (theta): Long-run variance (positive)
        ξ (xi): Volatility of volatility (positive)
        ρ (rho): Correlation (-1 to 1)
    
    The Feller condition 2κθ > ξ² ensures v_t remains strictly positive.

REFERENCES:
    Heston, S. L. (1993). "A Closed-Form Solution for Options with
        Stochastic Volatility with Applications to Bond and Currency
        Options". Review of Financial Studies, 6(2), 327-343.
    
    Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).
        "The Little Heston Trap". Wilmott Magazine, 2007(1), 83-92.

Author: Rodrigo Antonio Salado Ferrero
Repository: github.com/rodrigo-salado/heston-mcmc-calibration
License: MIT
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Type Aliases
# =============================================================================

# 1-dimensional array of floats
Array1D = NDArray[np.float64]

# 2-dimensional array of floats
Array2D = NDArray[np.float64]

# Heston parameter array: shape (5,) = [κ, θ, ξ, ρ, v₀]
ParamsArray = NDArray[np.float64]

# Strike price array: shape (n_strikes,)
StrikesArray = NDArray[np.float64]

# Option price array: shape (n_strikes,)
PricesArray = NDArray[np.float64]

# MCMC samples: shape (n_samples, 5)
SamplesArray = NDArray[np.float64]

# Log-posterior array: shape (n_samples,)
LogPosteriorArray = NDArray[np.float64]


# =============================================================================
# Core Data Classes
# =============================================================================

@dataclass(frozen=True)
class HestonParams:
    """
    Heston model parameters (risk-neutral measure).
    
    This immutable dataclass encapsulates the five parameters that
    fully define the Heston stochastic volatility model under the
    risk-neutral measure Q.
    
    Attributes:
        kappa: Mean reversion speed (positive). Controls how quickly
               variance reverts to its long-run mean θ.
        theta: Long-run variance (positive). The level to which v_t
               reverts in the long term.
        xi: Volatility of volatility (positive). Controls the
            variability of the variance process.
        rho: Correlation between asset and variance processes.
             Range: [-1, 1]. Negative ρ produces the leverage effect
             (volatility rises when prices fall).
        v0: Initial variance (positive). Starting point of the
            variance process at time 0.
    
    Example:
        >>> params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
        >>> params.to_array()
        array([2. , 0.04, 0.3 , -0.7, 0.04])
    
    Notes:
        - The Feller condition: 2 * κ * θ > ξ² ensures v_t > 0 almost surely
        - All parameters are assumed positive except ρ which can be negative
        - Parameters are defined under the risk-neutral measure Q
    """
    
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    
    def __post_init__(self) -> None:
        """
        Validate parameter constraints after initialization.
        
        Raises:
            ValueError: If any parameter violates its domain constraints.
        """
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.xi <= 0:
            raise ValueError(f"xi must be positive, got {self.xi}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.v0 <= 0:
            raise ValueError(f"v0 must be positive, got {self.v0}")
    
    def to_array(self) -> ParamsArray:
        """
        Convert Heston parameters to a NumPy array.
        
        Returns:
            Array of shape (5,) with order [κ, θ, ξ, ρ, v₀]
        
        Example:
            >>> params = HestonParams(2.0, 0.04, 0.3, -0.7, 0.04)
            >>> params.to_array()
            array([2. , 0.04, 0.3 , -0.7, 0.04])
        """
        return np.array([self.kappa, self.theta, self.xi, self.rho, self.v0])
    
    @classmethod
    def from_array(cls, arr: ParamsArray) -> "HestonParams":
        """
        Create HestonParams from a NumPy array.
        
        Args:
            arr: Array of shape (5,) with order [κ, θ, ξ, ρ, v₀]
        
        Returns:
            HestonParams instance with values from the array
        
        Raises:
            ValueError: If array shape is not (5,)
        
        Example:
            >>> arr = np.array([2.0, 0.04, 0.3, -0.7, 0.04])
            >>> params = HestonParams.from_array(arr)
        """
        if arr.shape != (5,):
            raise ValueError(f"Expected shape (5,), got {arr.shape}")
        return cls(
            kappa=arr[0],
            theta=arr[1],
            xi=arr[2],
            rho=arr[3],
            v0=arr[4],
        )
    
    def as_dict(self) -> Dict[str, float]:
        """
        Return parameters as a dictionary.
        
        Returns:
            Dictionary with parameter names as keys
        
        Example:
            >>> params = HestonParams(2.0, 0.04, 0.3, -0.7, 0.04)
            >>> params.as_dict()
            {'kappa': 2.0, 'theta': 0.04, 'xi': 0.3, 'rho': -0.7, 'v0': 0.04}
        """
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
            "v0": self.v0,
        }
    
    def check_feller(self) -> bool:
        """
        Check if the Feller condition is satisfied.
        
        The Feller condition 2κθ > ξ² ensures that the variance process
        v_t remains strictly positive almost surely. If violated, v_t can
        hit zero, though the process remains well-defined.
        
        Returns:
            True if 2κθ > ξ², False otherwise
        
        Example:
            >>> params = HestonParams(2.0, 0.04, 0.3, -0.7, 0.04)
            >>> params.check_feller()
            True  # 2*2.0*0.04 = 0.16 > 0.09
        """
        return 2 * self.kappa * self.theta > self.xi ** 2


@dataclass
class MarketData:
    """
    Market data container for option prices.
    
    This class holds all market-observable data needed for calibration:
    strike prices, corresponding option prices, and market environment
    parameters (spot, rate, tenor).
    
    Attributes:
        strikes: Array of strike prices (positive values)
        prices: Array of observed option prices (non-negative)
        spot: Current underlying asset price (positive)
        rate: Risk-free interest rate (can be zero or negative)
        tenor: Time to expiry in years (positive)
        dividend_yield: Continuous dividend yield (default: 0.0)
    
    Example:
        >>> data = MarketData(
        ...     strikes=np.array([80, 90, 100, 110, 120]),
        ...     prices=np.array([23.69, 15.75, 9.24, 4.58, 1.84]),
        ...     spot=100.0,
        ...     rate=0.03,
        ...     tenor=1.0
        ... )
    
    Notes:
        - Prices are assumed to be European call options
        - All options share the same expiry (tenor)
        - For multi-maturity calibration, use multiple MarketData instances
    """
    
    strikes: StrikesArray
    prices: PricesArray
    spot: float
    rate: float
    tenor: float
    dividend_yield: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate market data after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate market data consistency.
        
        Checks performed:
            1. Strikes and prices have same length
            2. All strikes are positive
            3. All prices are non-negative
            4. Spot price is positive
            5. Tenor is positive
        
        Raises:
            ValueError: If any validation check fails
        """
        if len(self.strikes) != len(self.prices):
            raise ValueError(
                f"Strikes and prices must have same length. "
                f"Got {len(self.strikes)} strikes and {len(self.prices)} prices."
            )
        
        if np.any(self.strikes <= 0):
            raise ValueError(f"All strikes must be positive. Found: {self.strikes}")
        
        if np.any(self.prices < 0):
            raise ValueError(f"All prices must be non-negative. Found: {self.prices}")
        
        if self.spot <= 0:
            raise ValueError(f"Spot price must be positive. Got {self.spot}")
        
        if self.tenor <= 0:
            raise ValueError(f"Tenor must be positive. Got {self.tenor}")
        
        # Note: rate can be negative (in some economic environments)
        # and dividend_yield can be zero or positive
    
    @property
    def n_options(self) -> int:
        """Number of option quotes in the dataset."""
        return len(self.strikes)
    
    @property
    def moneyness(self) -> Array1D:
        """
        Calculate moneyness = strike / spot.
        
        Returns:
            Array of moneyness values (dimensionless)
        
        Example:
            >>> data = MarketData(strikes=[80, 100, 120], spot=100.0, ...)
            >>> data.moneyness
            array([0.8, 1.0, 1.2])
        """
        return self.strikes / self.spot
    
    @property
    def log_moneyness(self) -> Array1D:
        """
        Calculate log-moneyness = log(strike / spot).
        
        Returns:
            Array of log-moneyness values (dimensionless)
        
        Notes:
            Used in Fourier inversion methods which operate on log-strike.
        """
        return np.log(self.strikes / self.spot)


@dataclass
class CalibrationResult:
    """
    Results of Bayesian calibration.
    
    This class stores the complete output of an MCMC calibration run,
    including posterior samples, acceptance statistics, and metadata.
    
    Attributes:
        samples: MCMC samples of shape (n_samples, 5) - [κ, θ, ξ, ρ, v₀]
        log_posterior: Log posterior values for each sample
        acceptance_rate: Fraction of proposals accepted (0 to 1)
        true_params: Optional ground truth parameters (for synthetic data)
        param_names: Parameter names (default: ["kappa", "theta", "xi", "rho", "v0"])
        runtime_seconds: Total runtime in seconds
        metadata: Additional run information (seed, config, etc.)
    
    Example:
        >>> result = CalibrationResult(
        ...     samples=samples_array,
        ...     log_posterior=log_ps,
        ...     acceptance_rate=0.234,
        ...     true_params=HestonParams(2.0, 0.04, 0.3, -0.7, 0.04),
        ...     runtime_seconds=6.2
        ... )
        >>> result.summary()
    """
    
    samples: SamplesArray
    log_posterior: LogPosteriorArray
    acceptance_rate: float
    true_params: Optional[HestonParams] = None
    param_names: Tuple[str, ...] = ("kappa", "theta", "xi", "rho", "v0")
    runtime_seconds: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate result data."""
        if self.samples.shape[1] != 5:
            raise ValueError(
                f"Samples must have 5 parameters, got shape {self.samples.shape}"
            )
        if len(self.param_names) != 5:
            raise ValueError(
                f"param_names must have 5 elements, got {len(self.param_names)}"
            )
        if not 0 <= self.acceptance_rate <= 1:
            raise ValueError(
                f"Acceptance rate must be in [0, 1], got {self.acceptance_rate}"
            )
    
    def summary(self) -> None:
        """
        Print summary statistics of the posterior distribution.
        
        Output includes:
            - True parameters (if synthetic data)
            - Number of samples and acceptance rate
            - Mean, standard deviation, and 95% credible interval for each parameter
        
        Example output:
            ====================================================================
            HESTON MODEL CALIBRATION RESULTS
            ====================================================================
            
            True parameters: {'kappa': 2.0, 'theta': 0.04, 'xi': 0.3, ...}
            
            Posterior summary (n_samples=8000):
            Acceptance rate: 17.5%
            Runtime: 6.02 seconds
            
            --------------------------------------------------------------------
            Param          Mean        Std       2.5%     97.5%
            --------------------------------------------------------------------
              kappa       2.268     0.6100      1.185     3.604
              theta       0.035     0.0085      0.019     0.051
                 xi       0.322     0.0640      0.199     0.448
                rho      -0.704     0.1161     -0.964    -0.533
                 v0       0.048     0.0134      0.023     0.077
            --------------------------------------------------------------------
        """
        print("\n" + "=" * 70)
        print("HESTON MODEL CALIBRATION RESULTS")
        print("=" * 70)
        
        if self.true_params:
            print(f"\nTrue parameters: {self.true_params.as_dict()}")
        
        print(f"\nPosterior summary (n_samples={len(self.samples)}):")
        print(f"Acceptance rate: {self.acceptance_rate:.1%}")
        print(f"Runtime: {self.runtime_seconds:.2f} seconds")
        
        print("\n" + "-" * 70)
        print(f"{'Param':>8} {'Mean':>10} {'Std':>10} {'2.5%':>10} {'97.5%':>10}")
        print("-" * 70)
        
        for i, name in enumerate(self.param_names):
            chain = self.samples[:, i]
            mean = np.mean(chain)
            std = np.std(chain)
            q2_5 = np.quantile(chain, 0.025)
            q97_5 = np.quantile(chain, 0.975)
            print(f"{name:>8} {mean:10.4f} {std:10.4f} {q2_5:10.4f} {q97_5:10.4f}")
        
        print("-" * 70)
    
    def save(self, filepath: str) -> None:
        """
        Save calibration results to a compressed NumPy file.
        
        Args:
            filepath: Path to save the results (should end with .npz)
        
        Example:
            >>> result.save("results/my_calibration.npz")
        
        Notes:
            Saved file can be reloaded with CalibrationResult.load()
        """
        data = {
            "samples": self.samples,
            "log_posterior": self.log_posterior,
            "acceptance_rate": self.acceptance_rate,
            "runtime_seconds": self.runtime_seconds,
            "param_names": self.param_names,
        }
        if self.true_params:
            data["true_params"] = self.true_params.to_array()
        if self.metadata:
            data["metadata"] = str(self.metadata)
        
        np.savez_compressed(filepath, **data)
    
    @classmethod
    def load(cls, filepath: str) -> "CalibrationResult":
        """
        Load calibration results from a saved NPZ file.
        
        Args:
            filepath: Path to the saved NPZ file
        
        Returns:
            CalibrationResult instance with loaded data
        
        Example:
            >>> result = CalibrationResult.load("results/my_calibration.npz")
            >>> result.summary()
        """
        data = np.load(filepath, allow_pickle=True)
        
        # Extract true parameters if present
        true_params = None
        if "true_params" in data:
            true_params = HestonParams.from_array(data["true_params"])
        
        # Extract metadata if present
        metadata = None
        if "metadata" in data and data["metadata"].size > 0:
            metadata = data["metadata"].item()
        
        return cls(
            samples=data["samples"],
            log_posterior=data["log_posterior"],
            acceptance_rate=float(data["acceptance_rate"]),
            true_params=true_params,
            param_names=tuple(data["param_names"]),
            runtime_seconds=float(data["runtime_seconds"]),
            metadata=metadata,
        )
    
    def get_credible_interval(
        self, 
        param_name: str, 
        cred_mass: float = 0.95
    ) -> Tuple[float, float]:
        """
        Get credible interval for a specific parameter.
        
        Args:
            param_name: Parameter name ('kappa', 'theta', 'xi', 'rho', 'v0')
            cred_mass: Credible mass (e.g., 0.95 for 95% interval)
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        
        Raises:
            ValueError: If param_name is invalid or cred_mass not in (0, 1)
        
        Example:
            >>> kappa_interval = result.get_credible_interval('kappa', 0.95)
            >>> print(f"κ ∈ [{kappa_interval[0]:.3f}, {kappa_interval[1]:.3f}]")
        """
        if param_name not in self.param_names:
            raise ValueError(
                f"Invalid parameter name. Choose from {self.param_names}"
            )
        if not 0 < cred_mass < 1:
            raise ValueError(f"cred_mass must be in (0, 1), got {cred_mass}")
        
        idx = self.param_names.index(param_name)
        chain = self.samples[:, idx]
        alpha = (1 - cred_mass) / 2
        lower = np.quantile(chain, alpha)
        upper = np.quantile(chain, 1 - alpha)
        
        return (lower, upper)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class PriorConfig:
    """
    Configuration for prior distributions of Heston parameters.
    
    Each parameter has a specified prior distribution with its own
    parameters and domain bounds.
    
    Attributes:
        kappa_dist: Distribution type ('lognormal', 'normal', 'uniform')
        kappa_mu_log: Log-mean for lognormal prior on κ
        kappa_sigma_log: Log-standard deviation for lognormal prior on κ
        kappa_bounds: (lower, upper) bounds for κ
        
        theta_dist: Distribution type for θ
        theta_mu: Mean for normal prior on θ
        theta_sigma: Standard deviation for normal prior on θ
        theta_bounds: (lower, upper) bounds for θ
        
        xi_dist: Distribution type for ξ
        xi_mu: Mean for normal prior on ξ
        xi_sigma: Standard deviation for normal prior on ξ
        xi_bounds: (lower, upper) bounds for ξ
        
        rho_dist: Distribution type for ρ (typically 'uniform')
        rho_bounds: (lower, upper) bounds for ρ
        
        v0_dist: Distribution type for v₀
        v0_mu: Mean for normal prior on v₀
        v0_sigma: Standard deviation for normal prior on v₀
        v0_bounds: (lower, upper) bounds for v₀
    
    Default values are chosen to be mildly informative, providing enough
    regularisation while allowing the data to dominate.
    
    Reference:
        The prior choices follow recommendations in:
        Bakshi, G., Cao, C., & Chen, Z. (1997). "Empirical Performance of
        Alternative Option Pricing Models". Journal of Finance, 52(5), 2003-2049.
    """
    
    # Kappa prior (mean reversion speed)
    kappa_dist: str = "lognormal"
    kappa_mu_log: float = 0.6931  # log(2)
    kappa_sigma_log: float = 0.5
    kappa_bounds: Tuple[float, float] = (0.01, 20.0)
    
    # Theta prior (long-run variance)
    theta_dist: str = "normal"
    theta_mu: float = 0.04  # 20% annual volatility squared
    theta_sigma: float = 0.02
    theta_bounds: Tuple[float, float] = (0.001, 1.0)
    
    # Xi prior (volatility of volatility)
    xi_dist: str = "normal"
    xi_mu: float = 0.30
    xi_sigma: float = 0.15
    xi_bounds: Tuple[float, float] = (0.001, 2.0)
    
    # Rho prior (correlation)
    rho_dist: str = "uniform"
    rho_bounds: Tuple[float, float] = (-0.999, 0.999)
    
    # V0 prior (initial variance)
    v0_dist: str = "normal"
    v0_mu: float = 0.04
    v0_sigma: float = 0.02
    v0_bounds: Tuple[float, float] = (0.001, 1.0)


@dataclass
class MCMCConfig:
    """
    Configuration for MCMC sampling.
    
    Controls the pilot chain (for covariance estimation) and production
    chain (for posterior sampling) with Roberts-Gelman-Gilks optimal scaling.
    
    Attributes:
        pilot_n_iter: Number of pilot chain iterations
        pilot_n_burn: Burn-in iterations to discard from pilot
        pilot_proposal_scale: Initial proposal standard deviation scale
        pilot_initial: Initial parameter vector for pilot chain
        
        prod_n_burn: Production chain burn-in iterations
        prod_n_samples: Production chain sampling iterations
        
        target_acceptance: Target acceptance rate (0.234 optimal for 5D)
        adaptation_scale: RGG scaling factor 2.38² / d
        
        random_seed: Random seed for reproducibility
    
    Reference:
        Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). "Weak convergence
        and optimal scaling of random walk Metropolis algorithms".
        The Annals of Applied Probability, 7(1), 110-120.
    """
    
    # Pilot chain
    pilot_n_iter: int = 10000
    pilot_n_burn: int = 1000
    pilot_proposal_scale: float = 0.25
    pilot_initial: Tuple[float, ...] = (1.5, 0.05, 0.4, -0.5, 0.05)
    
    # Production chain
    prod_n_burn: int = 2000
    prod_n_samples: int = 50000
    
    # Proposal tuning
    target_acceptance: float = 0.234  # Optimal for high dimensions
    adaptation_scale: float = (2.38 ** 2) / 5.0  # RGG scaling
    
    # Random seed
    random_seed: int = 42


@dataclass
class FourierConfig:
    """
    Configuration for Fourier inversion integration.
    
    Controls the numerical integration parameters for the Gil-Peláez
    inversion formula used to compute option prices from the characteristic
    function.
    
    Attributes:
        u_max: Upper truncation limit for Fourier integral
        n_grid: Number of grid points for Simpson quadrature (must be even)
        integration_method: Integration method ('simpson' or 'trapezoidal')
    
    Notes:
        - u_max should be large enough to capture the integrand's decay
        - n_grid of 512 is typically sufficient for Heston
        - Simpson's rule converges faster than trapezoidal
    """
    
    u_max: float = 200.0
    n_grid: int = 512
    integration_method: str = "simpson"  # "simpson" or "trapezoidal"
    
    def __post_init__(self) -> None:
        """Validate Fourier configuration."""
        if self.u_max <= 0:
            raise ValueError(f"u_max must be positive, got {self.u_max}")
        if self.n_grid <= 0:
            raise ValueError(f"n_grid must be positive, got {self.n_grid}")
        if self.integration_method not in ["simpson", "trapezoidal"]:
            raise ValueError(
                f"integration_method must be 'simpson' or 'trapezoidal', "
                f"got {self.integration_method}"
            )
        # Ensure n_grid is even for Simpson's rule
        if self.integration_method == "simpson" and self.n_grid % 2 != 0:
            self.n_grid += 1  # Make even


@dataclass
class CalibrationConfig:
    """
    Complete calibration configuration.
    
    This class bundles all configuration components into a single object
    for passing to the calibration function.
    
    Attributes:
        market_data: Market data (strikes, prices, spot, rate, tenor)
        prior: Prior configuration (default: PriorConfig())
        mcmc: MCMC configuration (default: MCMCConfig())
        fourier: Fourier integration configuration (default: FourierConfig())
        sigma_noise: Standard deviation of Gaussian observation noise
        verbose: Print progress messages if True
    
    Example:
        >>> config = CalibrationConfig(
        ...     market_data=market_data,
        ...     sigma_noise=0.05,
        ...     verbose=True
        ... )
        >>> result = calibrate_heston(config)
    """
    
    market_data: MarketData
    prior: PriorConfig = field(default_factory=PriorConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    fourier: FourierConfig = field(default_factory=FourierConfig)
    sigma_noise: float = 0.05
    verbose: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.sigma_noise <= 0:
            raise ValueError(f"sigma_noise must be positive, got {self.sigma_noise}")