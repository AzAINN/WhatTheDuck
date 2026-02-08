"""
Portfolio Optimization + VaR Pipeline
End-to-end orchestration with metadata storage.

Pipeline Flow:
1. Load/Generate return data
2. Optimize portfolio (CVaR or Variance mode)
3. Calculate VaR (Classical MC and/or IQAE)
4. Store metadata for UI consumption
5. Generate summary report
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import pandas as pd

from portfolio_backend.portfolio_optimizer import (
    PortfolioOptimizer,
    PortfolioSpec,
    PortfolioResult,
    OptimizationMode,
    save_portfolio_result,
    load_portfolio_result
)
from portfolio_backend.var_calculator import (
    VaRCalculator,
    VaRSpec,
    VaRResult,
    VaRMethod,
    save_var_result,
    load_var_result,
    compare_var_methods
)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Optimization settings
    optimization_mode: OptimizationMode
    cardinality: Optional[int] = None
    max_weight: float = 0.3
    risk_aversion: float = 1.0  # For variance mode
    confidence_level_cvar: float = 0.95  # For CVaR mode

    # VaR settings
    var_methods: List[VaRMethod] = None  # List of methods to run
    confidence_level_var: float = 0.95
    n_mc_samples: int = 10000
    iqae_epsilon: float = 0.01
    distribution: str = "empirical"  # "empirical", "normal", "student_t"
    df_student_t: Optional[float] = None

    # Solver settings
    solver: str = "qihd"  # "qihd", "cvxpy"

    # Output settings
    output_dir: Path = Path("results/pipeline")
    run_name: Optional[str] = None

    def __post_init__(self):
        if self.var_methods is None:
            self.var_methods = [VaRMethod.CLASSICAL_MC, VaRMethod.IQAE]
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"run_{timestamp}"


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    config: PipelineConfig
    portfolio_result: PortfolioResult
    var_results: Dict[str, VaRResult]
    metadata: Dict[str, Any]
    timestamp: str
    success: bool
    message: str


class PortfolioPipeline:
    """End-to-end portfolio optimization and VaR pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = self.config.output_dir / self.config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("PORTFOLIO OPTIMIZATION + VaR PIPELINE")
        print("=" * 80)
        print(f"Run name: {self.config.run_name}")
        print(f"Output directory: {self.run_dir}")
        print(f"Optimization mode: {self.config.optimization_mode.value}")
        print(f"VaR methods: {[m.value for m in self.config.var_methods]}")
        print("=" * 80 + "\n")

    def run(self, returns_data: np.ndarray, asset_names: Optional[List[str]] = None) -> PipelineResult:
        """
        Run complete pipeline.

        Args:
            returns_data: Historical returns (n_samples, n_assets)
            asset_names: Optional asset names for reporting

        Returns:
            PipelineResult with all outputs
        """
        timestamp = datetime.now().isoformat()
        n_samples, n_assets = returns_data.shape

        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]

        print(f"[Pipeline] Starting pipeline with {n_samples} samples, {n_assets} assets\n")

        # Step 1: Optimize Portfolio
        print("STEP 1: PORTFOLIO OPTIMIZATION")
        print("-" * 80)
        portfolio_result = self._optimize_portfolio(returns_data)

        if not portfolio_result.success:
            return self._create_failed_result(
                portfolio_result=portfolio_result,
                message=f"Portfolio optimization failed: {portfolio_result.message}",
                timestamp=timestamp
            )

        self._save_and_report_portfolio(portfolio_result, asset_names)

        # Step 2: Calculate VaR for Optimized Portfolio
        print("\nSTEP 2: VaR CALCULATION")
        print("-" * 80)
        var_results = self._calculate_var_all_methods(
            portfolio_weights=portfolio_result.weights,
            returns_data=returns_data
        )

        self._save_and_report_var(var_results)

        # Step 3: Generate Metadata
        print("\nSTEP 3: METADATA GENERATION")
        print("-" * 80)
        metadata = self._generate_metadata(
            returns_data=returns_data,
            portfolio_result=portfolio_result,
            var_results=var_results,
            asset_names=asset_names
        )

        # Step 4: Save Complete Result
        pipeline_result = PipelineResult(
            config=self.config,
            portfolio_result=portfolio_result,
            var_results=var_results,
            metadata=metadata,
            timestamp=timestamp,
            success=True,
            message="Pipeline completed successfully"
        )

        self._save_pipeline_result(pipeline_result)
        self._generate_summary_report(pipeline_result, asset_names)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results saved to: {self.run_dir}")
        print("=" * 80 + "\n")

        return pipeline_result

    def _optimize_portfolio(self, returns_data: np.ndarray) -> PortfolioResult:
        """Step 1: Optimize portfolio."""
        spec = PortfolioSpec(
            returns=returns_data,
            mode=self.config.optimization_mode,
            cardinality=self.config.cardinality,
            max_weight=self.config.max_weight,
            risk_aversion=self.config.risk_aversion,
            confidence_level=self.config.confidence_level_cvar,
            solver=self.config.solver
        )

        optimizer = PortfolioOptimizer(spec)
        result = optimizer.optimize()

        return result

    def _calculate_var_all_methods(
        self,
        portfolio_weights: np.ndarray,
        returns_data: np.ndarray
    ) -> Dict[str, VaRResult]:
        """Step 2: Calculate VaR using all specified methods."""
        var_results = {}

        for method in self.config.var_methods:
            print(f"\nCalculating VaR using {method.value}...")

            spec = VaRSpec(
                portfolio_weights=portfolio_weights,
                returns_data=returns_data,
                method=method,
                confidence_level=self.config.confidence_level_var,
                n_samples=self.config.n_mc_samples,
                epsilon=self.config.iqae_epsilon,
                distribution=self.config.distribution,
                df=self.config.df_student_t
            )

            calculator = VaRCalculator(spec)
            result = calculator.calculate()
            var_results[method.value] = result

        return var_results

    def _generate_metadata(
        self,
        returns_data: np.ndarray,
        portfolio_result: PortfolioResult,
        var_results: Dict[str, VaRResult],
        asset_names: List[str]
    ) -> Dict[str, Any]:
        """Step 3: Generate comprehensive metadata for UI."""
        n_samples, n_assets = returns_data.shape
        weights = portfolio_result.weights

        # Portfolio characteristics
        portfolio_returns = returns_data @ weights
        portfolio_mean = np.mean(portfolio_returns)
        portfolio_std = np.std(portfolio_returns)
        sharpe_ratio = portfolio_mean / portfolio_std if portfolio_std > 0 else 0

        # Active positions
        active_mask = weights > 1e-6
        active_positions = [
            {"asset": asset_names[i], "weight": float(weights[i])}
            for i in range(n_assets) if active_mask[i]
        ]
        active_positions.sort(key=lambda x: x["weight"], reverse=True)

        # VaR summary
        var_summary = {}
        for method_name, var_result in var_results.items():
            if var_result.success:
                var_summary[method_name] = {
                    "value": float(var_result.var_value),
                    "confidence_level": float(var_result.confidence_level),
                    "ci_lower": float(var_result.confidence_interval[0]) if var_result.confidence_interval else None,
                    "ci_upper": float(var_result.confidence_interval[1]) if var_result.confidence_interval else None,
                    "cvar": var_result.metadata.get("cvar"),
                    "elapsed_time": var_result.metadata.get("elapsed_time")
                }

        # Risk metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_risk = np.std(downside_returns) if len(downside_returns) > 0 else 0

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "data": {
                "n_samples": int(n_samples),
                "n_assets": int(n_assets),
                "sample_period": "historical"  # Could be enhanced with date range
            },
            "optimization": {
                "mode": self.config.optimization_mode.value,
                "objective_value": float(portfolio_result.objective_value),
                "solver": self.config.solver,
                "cardinality": self.config.cardinality,
                "active_assets": int(np.sum(active_mask)),
                "max_weight": float(self.config.max_weight)
            },
            "portfolio": {
                "weights": weights.tolist(),
                "active_positions": active_positions,
                "expected_return": float(portfolio_mean),
                "volatility": float(portfolio_std),
                "sharpe_ratio": float(sharpe_ratio),
                "downside_risk": float(downside_risk),
                "max_drawdown": float(self._compute_max_drawdown(portfolio_returns))
            },
            "var": var_summary,
            "config": {
                "confidence_level_var": self.config.confidence_level_var,
                "distribution": self.config.distribution,
                "n_mc_samples": self.config.n_mc_samples
            }
        }

        return metadata

    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown from return series."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def _save_and_report_portfolio(self, result: PortfolioResult, asset_names: List[str]):
        """Save portfolio result and print report."""
        # Save to JSON
        filepath = self.run_dir / "portfolio_result.json"
        save_portfolio_result(result, filepath)

        # Print report
        print(f"\nOptimization: {result.mode}")
        print(f"Success: {result.success}")
        print(f"Objective value: {result.objective_value:.6f}")
        print(f"Active assets: {result.metadata.get('active_assets', 'N/A')}")

        # Top holdings
        weights = result.weights
        top_indices = np.argsort(weights)[::-1][:5]
        print("\nTop 5 holdings:")
        for i in top_indices:
            if weights[i] > 1e-6:
                print(f"  {asset_names[i]:<15} {weights[i]:>8.2%}")

    def _save_and_report_var(self, var_results: Dict[str, VaRResult]):
        """Save VaR results and print report."""
        for method_name, result in var_results.items():
            # Save to JSON
            filepath = self.run_dir / f"var_result_{method_name}.json"
            save_var_result(result, filepath)

            # Print report
            print(f"\n{method_name.upper()}:")
            print(f"  VaR_{result.confidence_level*100:.0f}%: {result.var_value:.4f}")
            if result.confidence_interval:
                print(f"  CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            if result.metadata.get("cvar"):
                print(f"  CVaR: {result.metadata['cvar']:.4f}")
            print(f"  Time: {result.metadata.get('elapsed_time', 0):.3f}s")

    def _save_pipeline_result(self, result: PipelineResult):
        """Save complete pipeline result."""
        # Save metadata (most important for UI)
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(result.metadata, f, indent=2)
        print(f"\n[Pipeline] Metadata saved to {metadata_file}")

        # Save full result
        full_result_file = self.run_dir / "pipeline_result.json"
        data = {
            "timestamp": result.timestamp,
            "success": result.success,
            "message": result.message,
            "config": {
                "optimization_mode": result.config.optimization_mode.value,
                "var_methods": [m.value for m in result.config.var_methods],
                "cardinality": result.config.cardinality,
                "max_weight": result.config.max_weight,
                "confidence_level_var": result.config.confidence_level_var
            },
            "portfolio": {
                "weights": result.portfolio_result.weights.tolist(),
                "objective": float(result.portfolio_result.objective_value),
                "mode": result.portfolio_result.mode
            },
            "var": {
                method: {
                    "value": float(var_result.var_value),
                    "success": var_result.success
                }
                for method, var_result in result.var_results.items()
            },
            "metadata": result.metadata
        }

        with open(full_result_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Pipeline] Full result saved to {full_result_file}")

    def _generate_summary_report(self, result: PipelineResult, asset_names: List[str]):
        """Generate human-readable summary report."""
        report_file = self.run_dir / "summary_report.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PORTFOLIO OPTIMIZATION + VaR PIPELINE SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Run: {self.config.run_name}\n")
            f.write(f"Timestamp: {result.timestamp}\n\n")

            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Optimization mode: {self.config.optimization_mode.value}\n")
            f.write(f"Solver: {self.config.solver}\n")
            f.write(f"Cardinality: {self.config.cardinality or 'None'}\n")
            f.write(f"Max weight: {self.config.max_weight:.2%}\n")
            f.write(f"VaR confidence: {self.config.confidence_level_var:.2%}\n")
            f.write(f"Distribution: {self.config.distribution}\n\n")

            # Portfolio
            f.write("OPTIMIZED PORTFOLIO\n")
            f.write("-" * 80 + "\n")
            f.write(f"Expected return: {result.metadata['portfolio']['expected_return']:.4f}\n")
            f.write(f"Volatility: {result.metadata['portfolio']['volatility']:.4f}\n")
            f.write(f"Sharpe ratio: {result.metadata['portfolio']['sharpe_ratio']:.4f}\n")
            f.write(f"Active assets: {result.metadata['optimization']['active_assets']}\n\n")

            f.write("Active positions:\n")
            for pos in result.metadata['portfolio']['active_positions']:
                f.write(f"  {pos['asset']:<15} {pos['weight']:>8.2%}\n")
            f.write("\n")

            # VaR
            f.write("VALUE-AT-RISK ESTIMATES\n")
            f.write("-" * 80 + "\n")
            for method_name, var_info in result.metadata['var'].items():
                f.write(f"\n{method_name.upper()}:\n")
                f.write(f"  VaR: {var_info['value']:.4f}\n")
                if var_info.get('ci_lower') and var_info.get('ci_upper'):
                    f.write(f"  95% CI: [{var_info['ci_lower']:.4f}, {var_info['ci_upper']:.4f}]\n")
                if var_info.get('cvar'):
                    f.write(f"  CVaR: {var_info['cvar']:.4f}\n")
                f.write(f"  Time: {var_info['elapsed_time']:.3f}s\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"[Pipeline] Summary report saved to {report_file}")

    def _create_failed_result(
        self,
        portfolio_result: PortfolioResult,
        message: str,
        timestamp: str
    ) -> PipelineResult:
        """Create failed pipeline result."""
        return PipelineResult(
            config=self.config,
            portfolio_result=portfolio_result,
            var_results={},
            metadata={},
            timestamp=timestamp,
            success=False,
            message=message
        )


def load_pipeline_result(run_dir: Path) -> PipelineResult:
    """Load pipeline result from directory."""
    # Load full result
    with open(run_dir / "pipeline_result.json", 'r') as f:
        data = json.load(f)

    # Load portfolio result
    portfolio_result = load_portfolio_result(run_dir / "portfolio_result.json")

    # Load VaR results
    var_results = {}
    for method in data['var'].keys():
        var_file = run_dir / f"var_result_{method}.json"
        if var_file.exists():
            var_results[method] = load_var_result(var_file)

    # Reconstruct config
    config_data = data['config']
    config = PipelineConfig(
        optimization_mode=OptimizationMode(config_data['optimization_mode']),
        var_methods=[VaRMethod(m) for m in config_data['var_methods']],
        cardinality=config_data.get('cardinality'),
        max_weight=config_data['max_weight'],
        confidence_level_var=config_data['confidence_level_var'],
        output_dir=run_dir.parent,
        run_name=run_dir.name
    )

    return PipelineResult(
        config=config,
        portfolio_result=portfolio_result,
        var_results=var_results,
        metadata=data['metadata'],
        timestamp=data['timestamp'],
        success=data['success'],
        message=data['message']
    )
