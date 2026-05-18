#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module for Heston Calibration
============================================

This module provides comprehensive visualization tools for analysing
MCMC posterior samples from Heston model calibration.

VISUALIZATION TYPES:
    1. Trace Plots: Convergence diagnostics (chain evolution over time)
    2. Posterior Densities: Marginal distributions for each parameter
    3. Pair Plots: Parameter correlations with scatter matrices
    4. Posterior Predictive: Model fit validation with smile plots
    5. Summary Dashboard: Complete calibration overview

STYLE CONFIGURATION:
    Uses seaborn styling with color schemes designed for:
    - Print-quality figures (300 DPI)
    - Colourblind accessibility (viridis/colorblind palettes)
    - Professional publication standards

DIAGNOSTIC INTERPRETATION:
    - Trace plots should look like "fuzzy caterpillars" (no trends)
    - Posterior densities should be unimodal (if well-identified)
    - Pair plot correlations > 0.7 indicate identifiability issues
    - Predictive bands should contain observed prices

Author: Rodrigo Antonio Salado Ferrero
Repository: github.com/rodrigo-salado/heston-mcmc-calibration
License: MIT
"""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any
import warnings

# Import local modules
from src.types import HestonParams, CalibrationResult, MarketData
from src.pricing import heston_call_prices, implied_volatility


# =============================================================================
# Style Configuration
# =============================================================================

def set_style(
    style: str = "seaborn-v0_8-darkgrid",
    context: str = "notebook",
    font_scale: float = 1.0,
    dpi: int = 150,
) -> None:
    """
    Set global plotting style for all visualizations.
    
    Configures matplotlib and seaborn for professional-looking figures.
    
    Args:
        style: Seaborn style theme
        context: Plot context ('paper', 'notebook', 'talk', 'poster')
        font_scale: Font size scaling factor
        dpi: Default figure DPI
    
    Example:
        >>> set_style(style="seaborn-v0_8-whitegrid", context="talk", font_scale=1.2)
    """
    try:
        sns.set_theme(style=style, context=context, font_scale=font_scale)
    except Exception:
        # Fallback for older seaborn versions
        plt.style.use(style)
    
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 1.5


# =============================================================================
# Trace Plots (Convergence Diagnostics)
# =============================================================================

def plot_trace(
    samples: NDArray[np.float64],
    param_names: Tuple[str, ...] = ("kappa", "theta", "xi", "rho", "v0"),
    true_params: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot MCMC trace plots for convergence diagnostics.
    
    Trace plots show the evolution of each parameter over MCMC iterations.
    A well-mixed chain should look like a "fuzzy caterpillar" with no
    visible trends, drifts, or distinct phases.
    
    Convergence criteria:
        - Stationarity: Mean and variance stable over time
        - Mixing: Rapid oscillation around stable mean
        - No long-range correlations or stuck regions
    
    Args:
        samples: MCMC samples of shape (n_samples, n_params)
        param_names: Parameter names (default: Heston parameters)
        true_params: Optional ground truth values (for synthetic data)
        figsize: Figure size (width, height) in inches
        save_path: Path to save figure (optional)
        show: Display figure if True
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> plot_trace(result.samples, true_params=result.true_params.as_dict())
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
    
    if n_params == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        chain = samples[:, i]
        iterations = np.arange(len(chain))
        
        # Plot chain
        ax.plot(iterations, chain, color='steelblue', lw=0.5, alpha=0.7)
        
        # Add true value if provided
        if true_params and name in true_params:
            ax.axhline(
                true_params[name],
                color='darkred',
                ls='--',
                lw=1.5,
                label=f'True {name}={true_params[name]:.3f}'
            )
        
        # Add posterior mean line
        ax.axhline(
            np.mean(chain),
            color='black',
            ls=':',
            lw=1.0,
            alpha=0.7,
            label=f'Post. mean={np.mean(chain):.3f}'
        )
        
        ax.set_ylabel(name)
        ax.legend(loc='upper right', fontsize=9)
        
        # Add acceptance summary
        if i == 0:
            ax.set_title('MCMC Trace Plots - Convergence Diagnostics')
    
    axes[-1].set_xlabel('MCMC Iteration')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Posterior Density Plots
# =============================================================================

def plot_posterior_density(
    samples: NDArray[np.float64],
    param_names: Tuple[str, ...] = ("kappa", "theta", "xi", "rho", "v0"),
    true_params: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (15, 3),
    bins: int = 50,
    kde: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot marginal posterior densities for all parameters.
    
    Histograms with optional KDE overlay showing the marginal distribution
    of each parameter. Narrow, well-separated peaks indicate well-identified
    parameters. Wide, flat distributions indicate poor identifiability.
    
    Interpretation:
        - θ (theta): Long-run variance → Usually tight (well-identified)
        - v₀ (v0): Initial variance → Usually tight (well-identified)
        - ξ (xi): Vol-of-vol → Moderate uncertainty
        - ρ (rho): Correlation → Moderate uncertainty  
        - κ (kappa): Mean reversion → Often wide (poorly identified)
    
    Args:
        samples: MCMC samples (n_samples, n_params)
        param_names: Parameter names
        true_params: Optional ground truth values
        figsize: Figure size (width, height)
        bins: Number of histogram bins
        kde: Overlay KDE if True
        save_path: Path to save figure
        show: Display figure if True
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> plot_posterior_density(result.samples, true_params=result.true_params.as_dict())
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=figsize)
    
    if n_params == 1:
        axes = [axes]
    
    colors = sns.color_palette("viridis", n_params)
    
    for i, (ax, name, color) in enumerate(zip(axes, param_names, colors)):
        chain = samples[:, i]
        
        # Plot histogram
        ax.hist(chain, bins=bins, density=True, color=color, alpha=0.6, edgecolor='black', linewidth=0.5)
        
        # Overlay KDE
        if kde:
            sns.kdeplot(chain, ax=ax, color=color, linewidth=2, label='Posterior')
        
        # Add true value
        if true_params and name in true_params:
            ax.axvline(
                true_params[name],
                color='darkred',
                ls='--',
                lw=2,
                label=f'True: {true_params[name]:.3f}'
            )
        
        # Add posterior mean
        mean_val = np.mean(chain)
        ax.axvline(mean_val, color='black', ls=':', lw=1.5, alpha=0.7, label=f'Mean: {mean_val:.3f}')
        
        # Add 95% credible interval
        ci_lower = np.quantile(chain, 0.025)
        ci_upper = np.quantile(chain, 0.975)
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color=color, label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'{name.upper()} Posterior')
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    fig.suptitle('Marginal Posterior Densities', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Pair Plot (Correlation Matrix)
# =============================================================================

def plot_pair_grid(
    samples: NDArray[np.float64],
    param_names: Tuple[str, ...] = ("kappa", "theta", "xi", "rho", "v0"),
    true_params: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (12, 10),
    alpha: float = 0.3,
    s: float = 2.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create pair plot showing parameter correlations.
    
    Visualises the joint posterior distribution with:
        - Diagonal: Marginal histograms for each parameter
        - Lower triangle: 2D scatter plots
        - Upper triangle: Pearson correlation coefficients
    
    Strong correlations (|r| > 0.7) indicate identifiability issues:
        - κ (kappa) vs v₀: Often negatively correlated
        - κ (kappa) vs θ: Can be positively correlated
        - ρ (rho) vs ξ (xi): Sometimes correlated
    
    Args:
        samples: MCMC samples (n_samples, n_params)
        param_names: Parameter names
        true_params: Optional ground truth values
        figsize: Figure size (width, height)
        alpha: Scatter point transparency
        s: Scatter point size
        save_path: Path to save figure
        show: Display figure if True
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> plot_pair_grid(result.samples, true_params=result.true_params.as_dict())
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)
    
    # Sample thinning for scatter plots (improves performance)
    n_samples = len(samples)
    thin = max(1, n_samples // 2000)  # Plot at most 2000 points
    samples_thinned = samples[::thin]
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: Histogram
                ax.hist(samples[:, i], bins=30, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
                
                if true_params and param_names[i] in true_params:
                    ax.axvline(
                        true_params[param_names[i]],
                        color='darkred',
                        ls='--',
                        lw=1.5
                    )
                
                ax.set_yticks([])
                
            elif i > j:
                # Lower triangle: Scatter plot
                ax.scatter(
                    samples_thinned[:, j],
                    samples_thinned[:, i],
                    s=s,
                    alpha=alpha,
                    color='steelblue',
                    edgecolors='none'
                )
                
                # Add true parameter point if available
                if true_params:
                    ax.scatter(
                        true_params[param_names[j]],
                        true_params[param_names[i]],
                        s=50,
                        color='darkred',
                        marker='x',
                        linewidths=2,
                        zorder=5
                    )
                
                ax.set_xlim(np.quantile(samples[:, j], 0.001), np.quantile(samples[:, j], 0.999))
                ax.set_ylim(np.quantile(samples[:, i], 0.001), np.quantile(samples[:, i], 0.999))
                
            else:
                # Upper triangle: Correlation coefficient
                corr = np.corrcoef(samples[:, i], samples[:, j])[0, 1]
                
                # Colour based on correlation strength
                color = 'darkred' if abs(corr) > 0.7 else 'black'
                fontsize = 12 if abs(corr) > 0.5 else 10
                
                ax.text(
                    0.5, 0.5, f'{corr:+.2f}',
                    ha='center', va='center',
                    fontsize=fontsize, fontweight='bold',
                    color=color,
                    transform=ax.transAxes
                )
                
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Axis labels
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
            else:
                ax.set_xticklabels([])
            
            if j == 0:
                ax.set_ylabel(param_names[i])
            else:
                ax.set_yticklabels([])
    
    fig.suptitle('Pair Plot: Parameter Correlations', y=0.98, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Posterior Predictive Check
# =============================================================================

def plot_posterior_predictive(
    result: CalibrationResult,
    market_data: MarketData,
    n_samples: int = 200,
    cred_mass: float = 0.95,
    K_grid: Optional[NDArray[np.float64]] = None,
    n_grid: int = 100,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot posterior predictive check on the implied volatility smile.
    
    This is the most important visual validation: it shows whether the
    calibrated model can reproduce the observed option prices.
    
    The plot displays:
        - Observed market implied volatilities (points)
        - Posterior median model smile (solid line)
        - 95% posterior predictive band (shaded region)
    
    A well-calibrated model should have:
        - Observed points within the predictive band
        - Band width reflecting parameter uncertainty
        - No systematic bias (points scattered around median)
    
    Args:
        result: CalibrationResult from MCMC
        market_data: Market data used for calibration
        n_samples: Number of posterior samples to use
        cred_mass: Credible mass for predictive band (e.g., 0.95)
        K_grid: Custom strike grid (optional)
        n_grid: Number of grid points if K_grid not provided
        figsize: Figure size (width, height)
        save_path: Path to save figure
        show: Display figure if True
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> plot_posterior_predictive(result, market_data)
    """
    # Create strike grid if not provided
    if K_grid is None:
        K_min = market_data.strikes.min() * 0.95
        K_min = max(K_min, 0.01)  # Ensure positive strikes
        K_max = market_data.strikes.max() * 1.05
        K_grid = np.linspace(K_min, K_max, n_grid)
    
    # Randomly sample from posterior
    n_total = len(result.samples)
    indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
    
    # Compute model prices and implied vols for each sample
    model_iv = np.zeros((len(indices), len(K_grid)))
    
    for k, idx in enumerate(indices):
        params = result.samples[idx]
        kappa, theta, xi, rho, v0 = params
        
        # Compute model prices
        prices = heston_call_prices(
            S0=market_data.spot,
            strikes=K_grid,
            r=market_data.rate,
            T=market_data.tenor,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            v0=v0,
        )
        
        # Compute implied volatilities
        for j, (K, price) in enumerate(zip(K_grid, prices)):
            model_iv[k, j] = implied_volatility(
                price, market_data.spot, K, market_data.rate, market_data.tenor
            )
    
    # Compute predictive intervals
    iv_median = np.nanmedian(model_iv, axis=0)
    iv_lower = np.nanquantile(model_iv, (1 - cred_mass) / 2, axis=0)
    iv_upper = np.nanquantile(model_iv, (1 + cred_mass) / 2, axis=0)
    
    # Compute market implied volatilities
    market_iv = np.array([
        implied_volatility(p, market_data.spot, K, market_data.rate, market_data.tenor)
        for K, p in zip(market_data.strikes, market_data.prices)
    ])
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Predictive band
    ax.fill_between(
        K_grid, iv_lower, iv_upper,
        color='steelblue', alpha=0.25,
        label=f'{int(cred_mass * 100)}% Posterior Predictive Band'
    )
    
    # Posterior median
    ax.plot(
        K_grid, iv_median, '-',
        color='steelblue', lw=2,
        label='Posterior Median'
    )
    
    # Market observations
    ax.scatter(
        market_data.strikes, market_iv,
        color='darkred', s=80, zorder=5,
        label='Market Prices (observed)',
        edgecolors='white', linewidths=1.5
    )
    
    # Spot reference line
    ax.axvline(
        market_data.spot, color='gray',
        linestyle=':', alpha=0.6,
        label=f'Spot = {market_data.spot:.1f}'
    )
    
    ax.set_xlabel('Strike Price K')
    ax.set_ylabel('Implied Volatility')
    ax.set_title('Posterior Predictive Check - Volatility Smile')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Summary Dashboard
# =============================================================================

def plot_calibration_summary(
    result: CalibrationResult,
    market_data: MarketData,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create comprehensive calibration summary dashboard.
    
    Combines multiple plots into a single figure:
        - Trace plots (convergence)
        - Posterior densities (marginals)
        - Posterior predictive (model fit)
        - Parameter summary table
    
    This is the recommended single-plot summary for reports.
    
    Args:
        result: CalibrationResult from MCMC
        market_data: Market data used for calibration
        figsize: Figure size (width, height)
        save_path: Path to save figure
        show: Display figure if True
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> plot_calibration_summary(result, market_data)
    """
    # Create figure with GridSpec for custom layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Trace plots (top row, first two columns)
    ax_trace = []
    for i in range(2):
        ax = fig.add_subplot(gs[0, i])
        ax_trace.append(ax)
    
    # Posterior density (top row, third column)
    ax_density = fig.add_subplot(gs[0, 2])
    
    # Predictive check (middle row, all columns)
    ax_predictive = fig.add_subplot(gs[1, :])
    
    # Parameter summary (bottom row, all columns)
    ax_summary = fig.add_subplot(gs[2, :])
    
    # 1. Trace plots (first 2 parameters for brevity)
    param_names = result.param_names
    true_dict = result.true_params.as_dict() if result.true_params else None
    
    for i, (ax, name) in enumerate(zip(ax_trace, param_names[:2])):
        chain = result.samples[:, i]
        iterations = np.arange(len(chain))
        
        ax.plot(iterations, chain, color='steelblue', lw=0.5, alpha=0.7)
        
        if true_dict and name in true_dict:
            ax.axhline(true_dict[name], color='darkred', ls='--', lw=1.5, label=f'True: {true_dict[name]:.3f}')
        
        ax.axhline(np.mean(chain), color='black', ls=':', lw=1.0, alpha=0.7)
        ax.set_ylabel(name)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'{name.upper()} Trace')
    
    ax_trace[0].set_title('MCMC Convergence')
    ax_trace[-1].set_xlabel('Iteration')
    
    # 2. Posterior density (all parameters)
    colors = sns.color_palette("viridis", len(param_names))
    
    for i, (name, color) in enumerate(zip(param_names, colors)):
        chain = result.samples[:, i]
        sns.kdeplot(chain, ax=ax_density, color=color, linewidth=2, label=name.upper())
        
        if true_dict and name in true_dict:
            ax_density.axvline(true_dict[name], color=color, ls='--', lw=1.5, alpha=0.5)
    
    ax_density.set_xlabel('Parameter Value')
    ax_density.set_ylabel('Density')
    ax_density.set_title('Posterior Densities')
    ax_density.legend(loc='upper right', fontsize=9)
    
    # 3. Predictive check
    # Sample from posterior for predictive check
    n_pp_samples = min(100, len(result.samples))
    indices = np.random.choice(len(result.samples), n_pp_samples, replace=False)
    
    K_grid = np.linspace(market_data.strikes.min() * 0.95, market_data.strikes.max() * 1.05, 100)
    pp_iv = np.zeros((n_pp_samples, len(K_grid)))
    
    for k, idx in enumerate(indices):
        params = result.samples[idx]
        prices = heston_call_prices(
            S0=market_data.spot,
            strikes=K_grid,
            r=market_data.rate,
            T=market_data.tenor,
            kappa=params[0], theta=params[1], xi=params[2], rho=params[3], v0=params[4]
        )
        
        for j, (K, price) in enumerate(zip(K_grid, prices)):
            pp_iv[k, j] = implied_volatility(price, market_data.spot, K, market_data.rate, market_data.tenor)
    
    iv_median = np.nanmedian(pp_iv, axis=0)
    iv_lower = np.nanquantile(pp_iv, 0.025, axis=0)
    iv_upper = np.nanquantile(pp_iv, 0.975, axis=0)
    
    market_iv = np.array([
        implied_volatility(p, market_data.spot, K, market_data.rate, market_data.tenor)
        for K, p in zip(market_data.strikes, market_data.prices)
    ])
    
    ax_predictive.fill_between(K_grid, iv_lower, iv_upper, color='steelblue', alpha=0.25, label='95% Predictive Band')
    ax_predictive.plot(K_grid, iv_median, '-', color='steelblue', lw=2, label='Posterior Median')
    ax_predictive.scatter(market_data.strikes, market_iv, color='darkred', s=60, zorder=5, label='Market', edgecolors='white')
    ax_predictive.axvline(market_data.spot, color='gray', ls=':', alpha=0.6)
    ax_predictive.set_xlabel('Strike Price K')
    ax_predictive.set_ylabel('Implied Volatility')
    ax_predictive.set_title('Posterior Predictive Check')
    ax_predictive.legend(loc='best', fontsize=9)
    
    # 4. Parameter summary table
    ax_summary.axis('tight')
    ax_summary.axis('off')
    
    # Prepare table data
    table_data = []
    for i, name in enumerate(param_names):
        chain = result.samples[:, i]
        row = [
            name.upper(),
            f'{np.mean(chain):.4f}',
            f'{np.std(chain):.4f}',
            f'{np.quantile(chain, 0.025):.4f}',
            f'{np.quantile(chain, 0.975):.4f}',
        ]
        if true_dict and name in true_dict:
            row.insert(1, f'{true_dict[name]:.4f}')
        table_data.append(row)
    
    # Column headers
    if true_dict:
        columns = ['Parameter', 'True', 'Mean', 'Std', '2.5%', '97.5%']
    else:
        columns = ['Parameter', 'Mean', 'Std', '2.5%', '97.5%']
    
    table = ax_summary.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(len(table_data)):
        for j in range(len(columns)):
            if (i % 2) == 0:
                table[(i + 1, j)].set_facecolor('#f2f2f2')
    
    ax_summary.set_title('Parameter Summary', pad=20, fontweight='bold')
    
    # Overall title
    fig.suptitle(
        f'Heston Calibration Summary\n'
        f'Acceptance Rate: {result.acceptance_rate:.1%} | '
        f'Runtime: {result.runtime_seconds:.1f}s',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Additional Diagnostic Plots
# =============================================================================

def plot_autocorrelation(
    samples: NDArray[np.float64],
    param_names: Tuple[str, ...] = ("kappa", "theta", "xi", "rho", "v0"),
    max_lag: int = 50,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot autocorrelation function for MCMC samples.
    
    High autocorrelation indicates poor mixing; low autocorrelation
    indicates efficient sampling. The autocorrelation should decay
    quickly to zero (typically within 10-20 lags).
    
    Args:
        samples: MCMC samples (n_samples, n_params)
        param_names: Parameter names
        max_lag: Maximum lag to compute
        figsize: Figure size (width, height)
        save_path: Path to save figure
        show: Display figure if True
    
    Returns:
        matplotlib Figure object
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
    
    if n_params == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        chain = samples[:, i]
        chain_centered = chain - np.mean(chain)
        
        # Compute autocorrelation
        autocorr = np.correlate(chain_centered, chain_centered, mode='full')
        autocorr = autocorr[autocorr.size // 2:] / autocorr[autocorr.size // 2]
        
        lags = np.arange(min(max_lag, len(autocorr)))
        ax.bar(lags, autocorr[:len(lags)], width=0.8, color='steelblue', alpha=0.7)
        ax.axhline(0, color='black', lw=0.5)
        ax.axhline(0.1, color='red', ls='--', lw=1, alpha=0.5, label='Significance threshold')
        
        ax.set_ylabel(name)
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(f'{name.upper()} - Autocorrelation')
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Lag')
    fig.suptitle('Autocorrelation Function - MCMC Mixing Diagnostics', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_ess(
    samples: NDArray[np.float64],
    param_names: Tuple[str, ...] = ("kappa", "theta", "xi", "rho", "v0"),
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot effective sample size (ESS) for each parameter.
    
    ESS estimates the number of independent samples in a correlated chain.
    Higher ESS indicates better mixing and more reliable inference.
    
    Rule of thumb: ESS should be at least 100 for reliable quantile estimates.
    
    Args:
        samples: MCMC samples (n_samples, n_params)
        param_names: Parameter names
        figsize: Figure size (width, height)
        save_path: Path to save figure
        show: Display figure if True
    
    Returns:
        matplotlib Figure object
    """
    from src.inference import compute_ess
    
    ess_values = compute_ess(samples)
    n_samples = len(samples)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(param_names))
    bars = ax.bar(x, ess_values, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Colour bars based on ESS/n ratio
    for i, (bar, ess) in enumerate(zip(bars, ess_values)):
        ratio = ess / n_samples
        if ratio < 0.1:
            bar.set_color('darkred')
        elif ratio < 0.3:
            bar.set_color('orange')
        else:
            bar.set_color('steelblue')
    
    ax.axhline(n_samples, color='black', ls='--', lw=1.5, alpha=0.7, label=f'Total samples = {n_samples}')
    ax.axhline(100, color='red', ls=':', lw=1.5, alpha=0.7, label='Minimum threshold (100)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([name.upper() for name in param_names])
    ax.set_ylabel('Effective Sample Size (ESS)')
    ax.set_title('Effective Sample Size - MCMC Efficiency')
    ax.legend(loc='upper right')
    
    # Add value labels on bars
    for i, (bar, ess) in enumerate(zip(bars, ess_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                f'{ess:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Style
    "set_style",
    # Main plots
    "plot_trace",
    "plot_posterior_density",
    "plot_pair_grid",
    "plot_posterior_predictive",
    "plot_calibration_summary",
    # Diagnostic plots
    "plot_autocorrelation",
    "plot_ess",
]