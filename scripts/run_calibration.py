#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Calibration Script
=======================

This script runs the complete Bayesian calibration pipeline for the Heston model
from the command line. It loads market data, runs adaptive MCMC, saves results,
and optionally generates diagnostic plots.

The script can be used for:
    1. Synthetic data validation (using generated data from notebook 01)
    2. Real market data calibration (using custom data files)
    3. Batch calibration experiments
    4. Automated calibration pipelines

USAGE:
    python scripts/run_calibration.py [--data DATA_FILE] [--output OUTPUT_DIR]

OPTIONS:
    --data          Path to market data NPZ file (default: ../data/synthetic/synthetic_market_data.npz)
    --output        Output directory for results (default: ../results)
    --noise-std     Standard deviation of observation noise (default: 0.05)
    --pilot-iter    Pilot chain iterations (default: 3000)
    --prod-iter     Production chain iterations (default: 8000)
    --seed          Random seed for reproducibility (default: 42)
    --verbose       Print detailed progress (default: True)

EXAMPLES:
    # Run calibration with synthetic data
    python scripts/run_calibration.py

    # Run calibration with custom data
    python scripts/run_calibration.py --data ../data/market/spx_options.npz

    # Run with custom parameters
    python scripts/run_calibration.py --pilot-iter 5000 --prod-iter 10000 --seed 123

    # Silent mode (minimal output)
    python scripts/run_calibration.py --verbose False

INPUT DATA FORMAT:
    The NPZ file must contain the following keys:
        - strikes: Array of strike prices
        - observed_prices: Array of observed option prices
        - spot: Current underlying price
        - rate: Risk-free interest rate
        - tenor: Time to maturity in years

    Optional keys (for validation):
        - kappa, theta, xi, rho, v0: True parameters

OUTPUT STRUCTURE:
    results/
    ├── chains/
    │   ├── calibration_result_YYYYMMDD_HHMMSS.npz  # Timestamped result
    │   └── latest_calibration.npz                  # Symlink to latest
    └── figures/
        ├── trace_plots.png
        ├── posterior_densities.png
        ├── pair_plot.png
        └── posterior_predictive.png

REFERENCE:
    Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). "Weak convergence
    and optimal scaling of random walk Metropolis algorithms".
    The Annals of Applied Probability, 7(1), 110-120.

Author: Rodrigo Antonio Salado Ferrero
Repository: github.com/rodrigo-salado/heston-mcmc-calibration
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field, asdict
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.types import CalibrationResult, MarketData, HestonParams, CalibrationConfig
from src.pricing import heston_call_prices, implied_volatility
from src.inference import calibrate_heston
from src.visualization import (
    set_style, plot_trace, plot_posterior_density,
    plot_pair_grid, plot_posterior_predictive, plot_calibration_summary
)


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logger(verbose: bool = True) -> logging.Logger:
    """
    Configure logger for the calibration script.
    
    Args:
        verbose: Enable detailed logging if True
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("heston_calibration")
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler if not already present
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_market_data(
    file_path: Path,
    logger: Optional[logging.Logger] = None
) -> Tuple[MarketData, Optional[HestonParams]]:
    """
    Load market data from NPZ file.
    
    The file must contain:
        - strikes: Strike prices array
        - observed_prices: Option prices array
        - spot: Current underlying price
        - rate: Risk-free rate
        - tenor: Time to maturity
    
    Optional (for validation):
        - kappa, theta, xi, rho, v0: True Heston parameters
    
    Args:
        file_path: Path to NPZ file
        logger: Logger instance for progress reporting
    
    Returns:
        Tuple of (MarketData, optional HestonParams for validation)
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If required keys are missing
    """
    if logger:
        logger.info(f"Loading market data from: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with np.load(file_path, allow_pickle=True) as data:
        # Required keys
        required_keys = ['strikes', 'observed_prices', 'spot', 'rate', 'tenor']
        missing_keys = [k for k in required_keys if k not in data]
        
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        strikes = data['strikes']
        observed_prices = data['observed_prices']
        spot = float(data['spot'])
        rate = float(data['rate'])
        tenor = float(data['tenor'])
        
        if logger:
            logger.info(f"  Options: {len(strikes)}")
            logger.info(f"  Spot: {spot}")
            logger.info(f"  Rate: {rate:.4f}")
            logger.info(f"  Tenor: {tenor} years")
            logger.info(f"  Strike range: [{strikes.min():.0f}, {strikes.max():.0f}]")
        
        # Optional true parameters
        true_params = None
        true_keys = ['kappa', 'theta', 'xi', 'rho', 'v0']
        if all(k in data for k in true_keys):
            true_params = HestonParams(
                kappa=float(data['kappa']),
                theta=float(data['theta']),
                xi=float(data['xi']),
                rho=float(data['rho']),
                v0=float(data['v0']),
            )
            if logger:
                logger.info(f"  True parameters: {true_params.as_dict()}")
    
    market_data = MarketData(
        strikes=strikes,
        prices=observed_prices,
        spot=spot,
        rate=rate,
        tenor=tenor,
    )
    
    return market_data, true_params


def save_calibration_results(
    result: CalibrationResult,
    output_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Path]:
    """
    Save calibration results to disk.
    
    Args:
        result: CalibrationResult from MCMC
        output_dir: Output directory
        logger: Logger instance for progress reporting
    
    Returns:
        Dictionary with paths to saved files
    """
    output_dir = Path(output_dir)
    chains_dir = output_dir / "chains"
    figures_dir = output_dir / "figures"
    
    # Create directories
    chains_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Save calibration result with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = chains_dir / f"calibration_result_{timestamp}.npz"
    result.save(str(result_path))
    saved_paths['timestamped'] = result_path
    
    if logger:
        logger.info(f"Saved timestamped result: {result_path}")
    
    # Save as latest
    latest_path = chains_dir / "latest_calibration.npz"
    result.save(str(latest_path))
    saved_paths['latest'] = latest_path
    
    if logger:
        logger.info(f"Saved as latest: {latest_path}")
    
    # Save summary as JSON
    summary = {
        "timestamp": timestamp,
        "acceptance_rate": result.acceptance_rate,
        "n_samples": len(result.samples),
        "runtime_seconds": result.runtime_seconds,
        "param_names": list(result.param_names),
        "posterior_stats": {
            name: {
                "mean": float(np.mean(result.samples[:, i])),
                "std": float(np.std(result.samples[:, i])),
                "2.5%": float(np.quantile(result.samples[:, i], 0.025)),
                "50%": float(np.quantile(result.samples[:, i], 0.5)),
                "97.5%": float(np.quantile(result.samples[:, i], 0.975)),
            }
            for i, name in enumerate(result.param_names)
        },
        "true_params": result.true_params.as_dict() if result.true_params else None,
    }
    
    summary_path = chains_dir / f"calibration_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    saved_paths['summary'] = summary_path
    
    if logger:
        logger.info(f"Saved summary: {summary_path}")
    
    return saved_paths


def generate_diagnostic_plots(
    result: CalibrationResult,
    market_data: MarketData,
    output_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Path]:
    """
    Generate diagnostic plots from calibration results.
    
    Args:
        result: CalibrationResult from MCMC
        market_data: Market data used for calibration
        output_dir: Output directory for figures
        logger: Logger instance for progress reporting
    
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    true_dict = result.true_params.as_dict() if result.true_params else None
    
    plot_paths = {}
    
    # 1. Trace plots
    if logger:
        logger.info("Generating trace plots...")
    
    plot_path = figures_dir / "trace_plots.png"
    plot_trace(
        samples=result.samples,
        param_names=result.param_names,
        true_params=true_dict,
        save_path=str(plot_path),
        show=False,
    )
    plot_paths['trace'] = plot_path
    
    # 2. Posterior densities
    if logger:
        logger.info("Generating posterior density plots...")
    
    plot_path = figures_dir / "posterior_densities.png"
    plot_posterior_density(
        samples=result.samples,
        param_names=result.param_names,
        true_params=true_dict,
        save_path=str(plot_path),
        show=False,
    )
    plot_paths['densities'] = plot_path
    
    # 3. Pair plot
    if logger:
        logger.info("Generating pair plot...")
    
    plot_path = figures_dir / "pair_plot.png"
    plot_pair_grid(
        samples=result.samples,
        param_names=result.param_names,
        true_params=true_dict,
        save_path=str(plot_path),
        show=False,
    )
    plot_paths['pair'] = plot_path
    
    # 4. Posterior predictive check
    if logger:
        logger.info("Generating posterior predictive plot...")
    
    plot_path = figures_dir / "posterior_predictive.png"
    plot_posterior_predictive(
        result=result,
        market_data=market_data,
        n_samples=200,
        save_path=str(plot_path),
        show=False,
    )
    plot_paths['predictive'] = plot_path
    
    # 5. Summary dashboard
    if logger:
        logger.info("Generating summary dashboard...")
    
    plot_path = figures_dir / "calibration_summary.png"
    plot_calibration_summary(
        result=result,
        market_data=market_data,
        save_path=str(plot_path),
        show=False,
    )
    plot_paths['summary'] = plot_path
    
    if logger:
        logger.info(f"Saved {len(plot_paths)} plots to {figures_dir}")
    
    return plot_paths


# =============================================================================
# Main Calibration Function
# =============================================================================

def run_calibration(
    data_path: Path,
    output_dir: Path,
    noise_std: float = 0.05,
    pilot_n_iter: int = 3000,
    prod_n_samples: int = 8000,
    random_seed: int = 42,
    verbose: bool = True,
    generate_plots: bool = True,
) -> Tuple[CalibrationResult, Dict[str, Path], Dict[str, Path]]:
    """
    Run the complete calibration pipeline.
    
    This function orchestrates the entire calibration process:
        1. Load market data
        2. Run adaptive MCMC calibration
        3. Save results
        4. Generate diagnostic plots (optional)
    
    Args:
        data_path: Path to market data NPZ file
        output_dir: Output directory for results
        noise_std: Standard deviation of observation noise
        pilot_n_iter: Number of pilot chain iterations
        prod_n_samples: Number of production chain samples
        random_seed: Random seed for reproducibility
        verbose: Print detailed progress if True
        generate_plots: Generate diagnostic plots if True
    
    Returns:
        Tuple of (CalibrationResult, saved_paths, plot_paths)
    
    Example:
        >>> result, saved, plots = run_calibration(
        ...     data_path=Path("../data/synthetic/synthetic_market_data.npz"),
        ...     output_dir=Path("../results"),
        ...     verbose=True,
        ... )
    """
    # Setup logging
    logger = setup_logger(verbose)
    
    print("\n" + "=" * 70)
    print("HESTON MODEL BAYESIAN CALIBRATION")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load data
    print("\n" + "-" * 40)
    print("STEP 1: Loading Market Data")
    print("-" * 40)
    
    market_data, true_params = load_market_data(data_path, logger)
    
    # Step 2: Run calibration
    print("\n" + "-" * 40)
    print("STEP 2: Running MCMC Calibration")
    print("-" * 40)
    print(f"  Pilot chain iterations: {pilot_n_iter}")
    print(f"  Production samples: {prod_n_samples}")
    print(f"  Noise standard deviation: {noise_std}")
    print(f"  Random seed: {random_seed}")
    print()
    
    start_time = time.time()
    
    result = calibrate_heston(
        market_data=market_data,
        sigma_noise=noise_std,
        true_params=true_params,
        verbose=verbose,
    )
    
    runtime = time.time() - start_time
    result.runtime_seconds = runtime
    
    print(f"\n  Total runtime: {runtime:.2f} seconds")
    print(f"  Acceptance rate: {result.acceptance_rate:.1%}")
    
    # Step 3: Save results
    print("\n" + "-" * 40)
    print("STEP 3: Saving Results")
    print("-" * 40)
    
    saved_paths = save_calibration_results(result, output_dir, logger)
    
    # Step 4: Generate plots
    plot_paths = {}
    if generate_plots:
        print("\n" + "-" * 40)
        print("STEP 4: Generating Diagnostic Plots")
        print("-" * 40)
        
        plot_paths = generate_diagnostic_plots(result, market_data, output_dir, logger)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {runtime:.2f} seconds")
    
    print("\nResults saved to:")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")
    
    if plot_paths:
        print("\nPlots saved to:")
        for name, path in plot_paths.items():
            print(f"  {name}: {path}")
    
    print("\n" + "=" * 70)
    
    return result, saved_paths, plot_paths


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run Bayesian calibration for Heston model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run calibration with synthetic data
    python run_calibration.py

    # Run calibration with custom data
    python run_calibration.py --data ../data/market/spx_options.npz

    # Run with custom parameters
    python run_calibration.py --pilot-iter 5000 --prod-iter 10000 --seed 123

    # Silent mode (minimal output)
    python run_calibration.py --verbose False

    # Skip plot generation
    python run_calibration.py --no-plots
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="../data/synthetic/synthetic_market_data.npz",
        help="Path to market data NPZ file (default: ../data/synthetic/synthetic_market_data.npz)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="../results",
        help="Output directory for results (default: ../results)"
    )
    
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Standard deviation of observation noise (default: 0.05)"
    )
    
    parser.add_argument(
        "--pilot-iter",
        type=int,
        default=3000,
        help="Pilot chain iterations (default: 3000)"
    )
    
    parser.add_argument(
        "--prod-iter",
        type=int,
        default=8000,
        help="Production chain samples (default: 8000)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Print detailed progress (default: True)"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating diagnostic plots"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Convert paths
    data_path = Path(args.data)
    output_dir = Path(args.output)
    
    # Validate input file
    if not data_path.exists():
        print(f"\nError: Data file not found: {data_path}")
        print("\nPlease ensure the file exists or provide a valid path with --data")
        sys.exit(1)
    
    # Run calibration
    try:
        result, saved_paths, plot_paths = run_calibration(
            data_path=data_path,
            output_dir=output_dir,
            noise_std=args.noise_std,
            pilot_n_iter=args.pilot_iter,
            prod_n_samples=args.prod_iter,
            random_seed=args.seed,
            verbose=args.verbose,
            generate_plots=not args.no_plots,
        )
        
        # Final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        result.summary()
        
    except Exception as e:
        print(f"\nError during calibration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
