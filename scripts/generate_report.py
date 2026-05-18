#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Calibration Report Script
===================================

This script generates a comprehensive HTML/PDF report from the calibration results,
including all diagnostic plots and summary statistics. The report is designed
for both quick validation and formal documentation.

The report includes:
    1. Executive summary with key metrics
    2. Parameter summary tables (mean, std, credible intervals)
    3. Convergence diagnostics (R-hat, ESS, acceptance rate)
    4. Diagnostic plots (trace, density, pair plot, predictive check)
    5. Uncertainty quantification (prediction intervals)
    6. Feller condition analysis
    7. Recommendations and next steps

USAGE:
    python scripts/generate_report.py [--input CHAIN_FILE] [--output REPORT_DIR]

OPTIONS:
    --input     Path to calibration result NPZ file (default: latest)
    --output    Output directory for report (default: ../results/report)
    --format    Report format: html, pdf, or both (default: html)
    --no-plots  Skip generating new plots (use existing)

EXAMPLES:
    # Generate report from latest calibration
    python scripts/generate_report.py

    # Generate report from specific file
    python scripts/generate_report.py --input results/chains/calibration_result_20260101_120000.npz

    # Generate PDF report
    python scripts/generate_report.py --format pdf

    # Generate both HTML and PDF
    python scripts/generate_report.py --format both

OUTPUT STRUCTURE:
    results/report/
    ├── index.html          # Main HTML report
    ├── report.pdf          # PDF version (if requested)
    ├── figures/            # Generated plots
    │   ├── trace_plots.png
    │   ├── posterior_densities.png
    │   ├── pair_plot.png
    │   └── posterior_predictive.png
    └── data/               # JSON data files
        ├── parameters.json
        ├── convergence.json
        └── metadata.json

REFERENCE:
    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021).
    "Rank-normalization, folding, and localization: An improved R-hat for
    assessing convergence of MCMC". Bayesian Analysis, 16(2), 667-718.

Author: Rodrigo Antonio Salado Ferrero
Repository: github.com/rodrigo-salado/heston-mcmc-calibration
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import time
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field, asdict
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.types import CalibrationResult, MarketData, HestonParams
from src.pricing import heston_call_prices, implied_volatility
from src.inference import compute_r_hat, compute_ess
from src.visualization import (
    set_style, plot_trace, plot_posterior_density,
    plot_pair_grid, plot_posterior_predictive, plot_calibration_summary,
    plot_autocorrelation
)


# =============================================================================
# Data Classes for Report Data
# =============================================================================

@dataclass
class ParameterSummary:
    """
    Summary statistics for a single parameter.
    
    Attributes:
        name: Parameter name (kappa, theta, xi, rho, v0)
        true_value: Optional true value (for synthetic data)
        mean: Posterior mean
        std: Posterior standard deviation
        median: Posterior median
        ci_lower: Lower bound of 95% credible interval
        ci_upper: Upper bound of 95% credible interval
        ess: Effective sample size
        r_hat: Gelman-Rubin convergence statistic
    """
    name: str
    true_value: Optional[float]
    mean: float
    std: float
    median: float
    ci_lower: float
    ci_upper: float
    ess: float
    r_hat: float
    
    @property
    def ci_width(self) -> float:
        """Width of the 95% credible interval."""
        return self.ci_upper - self.ci_lower
    
    @property
    def cv(self) -> float:
        """Coefficient of variation (std/mean)."""
        return self.std / self.mean if self.mean != 0 else 0.0
    
    @property
    def is_recovered(self) -> Optional[bool]:
        """Check if true value is within credible interval."""
        if self.true_value is not None:
            return self.ci_lower <= self.true_value <= self.ci_upper
        return None
    
    @property
    def identifiability(self) -> str:
        """Classify parameter identifiability based on CV."""
        if self.cv < 0.3:
            return "Well-identified"
        elif self.cv < 0.6:
            return "Moderately identified"
        else:
            return "Poorly identified"


@dataclass
class ConvergenceSummary:
    """Convergence diagnostics summary."""
    acceptance_rate: float
    n_samples: int
    n_burn: int
    n_chains: int
    r_hat_min: float
    r_hat_max: float
    r_hat_all_ok: bool
    ess_min: float
    ess_all_sufficient: bool
    runtime_seconds: float
    
    @property
    def acceptance_status(self) -> str:
        """Classify acceptance rate."""
        if 0.15 <= self.acceptance_rate <= 0.35:
            return "Optimal"
        elif self.acceptance_rate < 0.1:
            return "Too low (increase proposal scale)"
        elif self.acceptance_rate > 0.5:
            return "Too high (decrease proposal scale)"
        else:
            return "Acceptable"


@dataclass
class MarketSummary:
    """Market data summary."""
    n_options: int
    spot: float
    rate: float
    tenor: float
    strikes_min: float
    strikes_max: float
    atm_price: Optional[float] = None
    atm_iv: Optional[float] = None


@dataclass
class FellerSummary:
    """Feller condition analysis."""
    satisfied_pct: float
    mean_value: float
    std_value: float
    true_value: Optional[float] = None


# =============================================================================
# HTML Report Generator
# =============================================================================

class ReportGenerator:
    """
    HTML report generator for calibration results.
    
    This class orchestrates the generation of a comprehensive HTML report
    including all diagnostic plots, summary tables, and analysis sections.
    
    Attributes:
        result: CalibrationResult from MCMC
        market_data: MarketData used for calibration
        output_dir: Output directory for report files
        figures_dir: Subdirectory for plot images
        data_dir: Subdirectory for JSON data files
        timestamp: Report generation timestamp
    """
    
    def __init__(
        self,
        result: CalibrationResult,
        market_data: Optional[MarketData] = None,
        output_dir: Path = Path("../results/report"),
    ):
        """
        Initialise report generator.
        
        Args:
            result: CalibrationResult from MCMC
            market_data: Market data used for calibration
            output_dir: Output directory for report files
        """
        self.result = result
        self.market_data = market_data
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.data_dir = self.output_dir / "data"
        self.timestamp = datetime.now()
        
        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        set_style()
    
    def generate_plots(self, force: bool = False) -> Dict[str, Path]:
        """
        Generate all diagnostic plots.
        
        Args:
            force: Regenerate plots even if they exist
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        plots = {}
        true_dict = self.result.true_params.as_dict() if self.result.true_params else None
        
        # 1. Trace plots
        plot_path = self.figures_dir / "trace_plots.png"
        if force or not plot_path.exists():
            fig = plot_trace(
                samples=self.result.samples,
                param_names=self.result.param_names,
                true_params=true_dict,
                save_path=str(plot_path),
                show=False,
            )
            plt.close(fig)
        plots["trace"] = plot_path
        
        # 2. Posterior densities
        plot_path = self.figures_dir / "posterior_densities.png"
        if force or not plot_path.exists():
            fig = plot_posterior_density(
                samples=self.result.samples,
                param_names=self.result.param_names,
                true_params=true_dict,
                save_path=str(plot_path),
                show=False,
            )
            plt.close(fig)
        plots["densities"] = plot_path
        
        # 3. Pair plot
        plot_path = self.figures_dir / "pair_plot.png"
        if force or not plot_path.exists():
            fig = plot_pair_grid(
                samples=self.result.samples,
                param_names=self.result.param_names,
                true_params=true_dict,
                save_path=str(plot_path),
                show=False,
            )
            plt.close(fig)
        plots["pair"] = plot_path
        
        # 4. Posterior predictive
        if self.market_data is not None:
            plot_path = self.figures_dir / "posterior_predictive.png"
            if force or not plot_path.exists():
                fig = plot_posterior_predictive(
                    result=self.result,
                    market_data=self.market_data,
                    n_samples=200,
                    save_path=str(plot_path),
                    show=False,
                )
                plt.close(fig)
            plots["predictive"] = plot_path
        
        # 5. Autocorrelation
        plot_path = self.figures_dir / "autocorrelation.png"
        if force or not plot_path.exists():
            fig = plot_autocorrelation(
                samples=self.result.samples,
                param_names=self.result.param_names,
                save_path=str(plot_path),
                show=False,
            )
            plt.close(fig)
        plots["autocorr"] = plot_path
        
        return plots
    
    def compute_parameter_summaries(self) -> List[ParameterSummary]:
        """
        Compute summary statistics for all parameters.
        
        Returns:
            List of ParameterSummary objects
        """
        # Split chains for R-hat computation
        def split_chains(samples: np.ndarray, n_chains: int = 4) -> np.ndarray:
            n_samples = len(samples)
            chain_length = n_samples // n_chains
            return np.array([
                samples[i*chain_length:(i+1)*chain_length]
                for i in range(n_chains)
            ])
        
        chains = split_chains(self.result.samples)
        r_hat = compute_r_hat(chains)
        ess = compute_ess(self.result.samples)
        
        summaries = []
        true_dict = self.result.true_params.as_dict() if self.result.true_params else None
        
        for i, name in enumerate(self.result.param_names):
            chain = self.result.samples[:, i]
            true_val = true_dict.get(name) if true_dict else None
            
            summary = ParameterSummary(
                name=name,
                true_value=true_val,
                mean=np.mean(chain),
                std=np.std(chain),
                median=np.median(chain),
                ci_lower=np.quantile(chain, 0.025),
                ci_upper=np.quantile(chain, 0.975),
                ess=ess[i],
                r_hat=r_hat[i],
            )
            summaries.append(summary)
        
        return summaries
    
    def compute_convergence_summary(self) -> ConvergenceSummary:
        """Compute convergence diagnostics summary."""
        # Split chains for R-hat
        def split_chains(samples: np.ndarray, n_chains: int = 4) -> np.ndarray:
            n_samples = len(samples)
            chain_length = n_samples // n_chains
            return np.array([
                samples[i*chain_length:(i+1)*chain_length]
                for i in range(n_chains)
            ])
        
        chains = split_chains(self.result.samples)
        r_hat = compute_r_hat(chains)
        ess = compute_ess(self.result.samples)
        
        return ConvergenceSummary(
            acceptance_rate=self.result.acceptance_rate,
            n_samples=len(self.result.samples),
            n_burn=2000,  # Default burn-in
            n_chains=4,
            r_hat_min=np.min(r_hat),
            r_hat_max=np.max(r_hat),
            r_hat_all_ok=np.all(r_hat < 1.1),
            ess_min=np.min(ess),
            ess_all_sufficient=np.all(ess > 100),
            runtime_seconds=self.result.runtime_seconds,
        )
    
    def compute_market_summary(self) -> Optional[MarketSummary]:
        """Compute market data summary."""
        if self.market_data is None:
            return None
        
        # Compute ATM price and implied vol
        atm_idx = np.argmin(np.abs(self.market_data.strikes - self.market_data.spot))
        atm_price = self.market_data.prices[atm_idx] if atm_idx < len(self.market_data.prices) else None
        atm_iv = None
        
        if atm_price is not None:
            atm_iv = implied_volatility(
                atm_price, self.market_data.spot, self.market_data.spot,
                self.market_data.rate, self.market_data.tenor
            )
        
        return MarketSummary(
            n_options=self.market_data.n_options,
            spot=self.market_data.spot,
            rate=self.market_data.rate,
            tenor=self.market_data.tenor,
            strikes_min=np.min(self.market_data.strikes),
            strikes_max=np.max(self.market_data.strikes),
            atm_price=atm_price,
            atm_iv=atm_iv,
        )
    
    def compute_feller_summary(self) -> FellerSummary:
        """Compute Feller condition analysis."""
        samples = self.result.samples
        feller_values = 2 * samples[:, 0] * samples[:, 1] - samples[:, 2]**2
        feller_satisfied = feller_values > 0
        
        true_value = None
        if self.result.true_params:
            true_value = (
                2 * self.result.true_params.kappa * self.result.true_params.theta
                - self.result.true_params.xi ** 2
            )
        
        return FellerSummary(
            satisfied_pct=np.mean(feller_satisfied) * 100,
            mean_value=np.mean(feller_values),
            std_value=np.std(feller_values),
            true_value=true_value,
        )
    
    def image_to_base64(self, image_path: Path) -> str:
        """
        Convert image to base64 for embedding in HTML.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def generate_html(
        self,
        plots: Dict[str, Path],
        param_summaries: List[ParameterSummary],
        conv_summary: ConvergenceSummary,
        market_summary: Optional[MarketSummary],
        feller_summary: FellerSummary,
    ) -> str:
        """
        Generate complete HTML report.
        
        Args:
            plots: Dictionary mapping plot names to paths
            param_summaries: List of parameter summaries
            conv_summary: Convergence summary
            market_summary: Market data summary
            feller_summary: Feller condition summary
        
        Returns:
            HTML string
        """
        # Convert plots to base64 for embedding
        plot_embedded = {}
        for name, path in plots.items():
            if path.exists():
                plot_embedded[name] = self.image_to_base64(path)
        
        # Build parameter table rows
        param_rows = []
        for p in param_summaries:
            true_str = f"{p.true_value:.4f}" if p.true_value is not None else "—"
            recovered_str = ""
            if p.is_recovered is True:
                recovered_str = '<span style="color: green;">✓</span>'
            elif p.is_recovered is False:
                recovered_str = '<span style="color: red;">✗</span>'
            else:
                recovered_str = "—"
            
            row = f"""
            <tr>
                <td><strong>{p.name}</strong></td>
                <td>{true_str}</td>
                <td>{p.mean:.4f}</td>
                <td>{p.std:.4f}</td>
                <td>{p.median:.4f}</td>
                <td>[{p.ci_lower:.4f}, {p.ci_upper:.4f}]</td>
                <td>{p.ci_width:.4f}</td>
                <td>{p.cv:.3f}</td>
                <td>{p.ess:.0f}</td>
                <td>{p.r_hat:.3f}</td>
                <td>{p.identifiability}</td>
                <td>{recovered_str}</td>
            </tr>
            """
            param_rows.append(row)
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heston Model Calibration Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 30px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h1 {{
            text-align: center;
            border-bottom: 3px solid #3498db;
        }}
        .summary-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-success {{
            background-color: #d4edda;
            color: #155724;
        }}
        .badge-warning {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .badge-danger {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #666;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Heston Model Bayesian Calibration Report</h1>
        <p style="text-align: center;">
            Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}<br>
            Author: Rodrigo Antonio Salado Ferrero
        </p>
        
        <div class="summary-box">
            <h3>📈 Executive Summary</h3>
            <p>
                This report presents the results of Bayesian calibration of the Heston stochastic 
                volatility model using Markov Chain Monte Carlo (MCMC) methods.
            </p>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{conv_summary.acceptance_rate:.1%}</div>
                    <div class="metric-label">Acceptance Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{conv_summary.n_samples:,}</div>
                    <div class="metric-label">Posterior Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{conv_summary.ess_min:.0f}</div>
                    <div class="metric-label">Min Effective Sample Size</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{conv_summary.runtime_seconds:.1f}s</div>
                    <div class="metric-label">Runtime</div>
                </div>
            </div>
        </div>
        
        <h2>📋 Parameter Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>True</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Median</th>
                    <th>95% CI</th>
                    <th>CI Width</th>
                    <th>CV</th>
                    <th>ESS</th>
                    <th>R-hat</th>
                    <th>Identifiability</th>
                    <th>Recovered</th>
                </tr>
            </thead>
            <tbody>
                {''.join(param_rows)}
            </tbody>
        </table>
        
        <h2>🔍 Convergence Diagnostics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{conv_summary.r_hat_min:.3f} - {conv_summary.r_hat_max:.3f}</div>
                <div class="metric-label">R-hat Range (<1.1 = converged)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{'✓' if conv_summary.r_hat_all_ok else '⚠'}</div>
                <div class="metric-label">All R-hat {'<' if conv_summary.r_hat_all_ok else '>'} 1.1</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{conv_summary.ess_min:.0f}</div>
                <div class="metric-label">Min ESS (>100 recommended)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{conv_summary.acceptance_status}</div>
                <div class="metric-label">Acceptance Rate Status</div>
            </div>
        </div>
        
        <h2>📊 Diagnostic Plots</h2>
        <div class="plot-container">
            <h3>Trace Plots</h3>
            <img src="data:image/png;base64,{plot_embedded.get('trace', '')}" alt="Trace Plots">
        </div>
        
        <div class="plot-container">
            <h3>Posterior Densities</h3>
            <img src="data:image/png;base64,{plot_embedded.get('densities', '')}" alt="Posterior Densities">
        </div>
        
        <div class="plot-container">
            <h3>Parameter Correlations (Pair Plot)</h3>
            <img src="data:image/png;base64,{plot_embedded.get('pair', '')}" alt="Pair Plot">
        </div>
        
        <div class="plot-container">
            <h3>Autocorrelation</h3>
            <img src="data:image/png;base64,{plot_embedded.get('autocorr', '')}" alt="Autocorrelation">
        </div>
        
        <div class="plot-container">
            <h3>Posterior Predictive Check</h3>
            <img src="data:image/png;base64,{plot_embedded.get('predictive', '')}" alt="Posterior Predictive">
        </div>
        
        <h2>📈 Feller Condition Analysis</h2>
        <div class="summary-box">
            <p>
                The Feller condition (2·κ·θ > ξ²) ensures that the variance process remains
                strictly positive almost surely.
            </p>
            <ul>
                <li><strong>Satisfied in:</strong> {feller_summary.satisfied_pct:.1f}% of posterior samples</li>
                <li><strong>Mean value:</strong> {feller_summary.mean_value:.4f}</li>
                <li><strong>Standard deviation:</strong> {feller_summary.std_value:.4f}</li>
                <li><strong>True value:</strong> {feller_summary.true_value:.4f if feller_summary.true_value else 'N/A'}</li>
            </ul>
        </div>
        
        <h2>💡 Recommendations</h2>
        <div class="summary-box">
            <ul>
                <li><strong>Multi-maturity calibration:</strong> Add options with multiple expiries to better identify κ (mean reversion).</li>
                <li><strong>Time series data:</strong> Incorporate underlying price history to constrain physical parameters.</li>
                <li><strong>Alternative priors:</strong> Consider more informative priors if parameter uncertainty remains high.</li>
                <li><strong>Parallel tempering:</strong> Implement for multi-modal posterior distributions.</li>
                <li><strong>Model extensions:</strong> Consider Bates (jumps) or rough volatility models for improved fit.</li>
            </ul>
        </div>
    </div>
    
    <div class="footer">
        <p>
            Generated by Heston Calibration Framework v1.0<br>
            <a href="https://github.com/rodrigo-salado/heston-mcmc-calibration">github.com/rodrigo-salado/heston-mcmc-calibration</a>
        </p>
    </div>
</body>
</html>
"""
        return html
    
    def save_json_data(
        self,
        param_summaries: List[ParameterSummary],
        conv_summary: ConvergenceSummary,
        market_summary: Optional[MarketSummary],
        feller_summary: FellerSummary,
    ) -> None:
        """Save summary data as JSON files."""
        # Parameters data
        params_data = [asdict(p) for p in param_summaries]
        with open(self.data_dir / "parameters.json", "w") as f:
            json.dump(params_data, f, indent=2, default=str)
        
        # Convergence data
        with open(self.data_dir / "convergence.json", "w") as f:
            json.dump(asdict(conv_summary), f, indent=2, default=str)
        
        # Market data
        if market_summary:
            with open(self.data_dir / "market.json", "w") as f:
                json.dump(asdict(market_summary), f, indent=2, default=str)
        
        # Feller data
        with open(self.data_dir / "feller.json", "w") as f:
            json.dump(asdict(feller_summary), f, indent=2, default=str)
        
        # Metadata
        metadata = {
            "timestamp": self.timestamp.isoformat(),
            "version": "1.0",
            "author": "Rodrigo Antonio Salado Ferrero",
            "repository": "github.com/rodrigo-salado/heston-mcmc-calibration",
        }
        with open(self.data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def generate(self, regenerate_plots: bool = False) -> Path:
        """
        Generate complete report.
        
        Args:
            regenerate_plots: Regenerate all plots
        
        Returns:
            Path to generated HTML report
        """
        print("=" * 60)
        print("GENERATING CALIBRATION REPORT")
        print("=" * 60)
        
        # Generate plots
        print("\n1. Generating diagnostic plots...")
        plots = self.generate_plots(force=regenerate_plots)
        print(f"   Generated {len(plots)} plots")
        
        # Compute summaries
        print("\n2. Computing parameter summaries...")
        param_summaries = self.compute_parameter_summaries()
        print(f"   Processed {len(param_summaries)} parameters")
        
        print("\n3. Computing convergence diagnostics...")
        conv_summary = self.compute_convergence_summary()
        print(f"   Acceptance rate: {conv_summary.acceptance_rate:.1%}")
        print(f"   R-hat range: [{conv_summary.r_hat_min:.3f}, {conv_summary.r_hat_max:.3f}]")
        print(f"   Min ESS: {conv_summary.ess_min:.0f}")
        
        print("\n4. Computing market summary...")
        market_summary = self.compute_market_summary()
        if market_summary:
            print(f"   Options: {market_summary.n_options}")
            print(f"   Spot: {market_summary.spot}")
        
        print("\n5. Computing Feller condition...")
        feller_summary = self.compute_feller_summary()
        print(f"   Feller satisfied: {feller_summary.satisfied_pct:.1f}%")
        
        print("\n6. Saving JSON data...")
        self.save_json_data(param_summaries, conv_summary, market_summary, feller_summary)
        
        print("\n7. Generating HTML report...")
        html = self.generate_html(plots, param_summaries, conv_summary, market_summary, feller_summary)
        
        html_path = self.output_dir / "index.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"\nReport generated successfully!")
        print(f"   HTML report: {html_path}")
        print(f"   Figures: {self.figures_dir}")
        print(f"   Data: {self.data_dir}")
        print("=" * 60)
        
        return html_path


# =============================================================================
# Main Function
# =============================================================================

def find_latest_result(results_dir: Path) -> Optional[Path]:
    """
    Find the latest calibration result file.
    
    Args:
        results_dir: Directory containing calibration results
    
    Returns:
        Path to latest result file, or None if not found
    """
    # Check for latest_calibration.npz
    latest_file = results_dir / "latest_calibration.npz"
    if latest_file.exists():
        return latest_file
    
    # Find most recent timestamped file
    files = list(results_dir.glob("calibration_result_*.npz"))
    if not files:
        return None
    
    return max(files, key=lambda p: p.stat().st_mtime)


def load_market_data(data_path: Path) -> Optional[MarketData]:
    """
    Load market data from synthetic data file.
    
    Args:
        data_path: Path to market data NPZ file
    
    Returns:
        MarketData object or None if not found
    """
    if not data_path.exists():
        return None
    
    with np.load(data_path) as data:
        return MarketData(
            strikes=data['strikes'],
            prices=data['observed_prices'],
            spot=float(data['spot']),
            rate=float(data['rate']),
            tenor=float(data['tenor']),
        )


def main() -> None:
    """Main entry point for the script."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate calibration report from MCMC results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate report from latest calibration
    python generate_report.py

    # Generate report from specific file
    python generate_report.py --input results/chains/calibration_result_20260101_120000.npz

    # Regenerate all plots
    python generate_report.py --regenerate-plots
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to calibration result NPZ file (default: latest)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/report",
        help="Output directory for report (default: ../results/report)"
    )
    parser.add_argument(
        "--regenerate-plots",
        action="store_true",
        help="Regenerate all plots even if they exist"
    )
    parser.add_argument(
        "--market-data",
        type=str,
        default="../data/synthetic/synthetic_market_data.npz",
        help="Path to market data NPZ file"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results" / "chains"
    output_dir = base_dir / args.output
    market_data_path = base_dir / args.market_data
    
    print("\n" + "=" * 70)
    print("HESTON MODEL CALIBRATION REPORT GENERATOR")
    print("=" * 70)
    print(f"\nBase directory: {base_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find result file
    if args.input:
        result_path = Path(args.input)
        if not result_path.exists():
            print(f"\nError: Result file not found: {result_path}")
            sys.exit(1)
    else:
        result_path = find_latest_result(results_dir)
        if result_path is None:
            print(f"\nError: No calibration results found in {results_dir}")
            print("   Please run calibration first (notebook 02)")
            sys.exit(1)
    
    print(f"\nLoading results from: {result_path}")
    
    try:
        # Load calibration result
        result = CalibrationResult.load(str(result_path))
        print(f"   Samples: {len(result.samples)}")
        print(f"   Acceptance rate: {result.acceptance_rate:.1%}")
        
        # Load market data
        market_data = load_market_data(market_data_path)
        if market_data:
            print(f"\nLoaded market data: {market_data.n_options} options")
        else:
            print(f"\n⚠ Market data not found: {market_data_path}")
            print("   Posterior predictive plot will be skipped")
        
        # Generate report
        report_gen = ReportGenerator(result, market_data, output_dir)
        report_gen.generate(regenerate_plots=args.regenerate_plots)
        
    except Exception as e:
        print(f"\nError generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
