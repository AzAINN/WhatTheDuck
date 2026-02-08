"""
Example Usage of Portfolio Optimization + VaR Backend

This script demonstrates the complete workflow:
1. Generate synthetic return data (or load your own)
2. Run CVaR-focused optimization
3. Run Variance-focused optimization
4. Calculate VaR for both portfolios
5. Compare results
"""

import numpy as np
from pathlib import Path

from portfolio_backend import (
    PortfolioPipeline,
    PipelineConfig,
    OptimizationMode,
    VaRMethod,
    load_pipeline_result
)


def generate_sample_returns(n_samples=1000, n_assets=10, seed=42):
    """Generate synthetic return data using factor model."""
    np.random.seed(seed)

    # Factor model: r = μ + F @ B^T + ε
    n_factors = 3
    mu = np.random.uniform(0.05, 0.15, n_assets) / 252  # Daily returns
    factor_loadings = np.random.randn(n_assets, n_factors) * 0.5

    # Generate factors and noise
    factors = np.random.randn(n_samples, n_factors) * 0.02
    idiosyncratic = np.random.randn(n_samples, n_assets) * 0.01

    # Generate returns
    returns = mu + factors @ factor_loadings.T + idiosyncratic

    # Asset names
    asset_names = [f"STOCK_{chr(65 + i)}" for i in range(n_assets)]

    return returns, asset_names


def run_cvar_portfolio(returns, asset_names):
    """Run CVaR-focused portfolio optimization."""
    print("\n" + "=" * 80)
    print("RUNNING CVaR-FOCUSED PORTFOLIO OPTIMIZATION")
    print("=" * 80)

    config = PipelineConfig(
        optimization_mode=OptimizationMode.CVAR,
        cardinality=6,                   # Max 6 assets
        max_weight=0.30,                 # Max 30% per asset
        confidence_level_cvar=0.95,      # 95% CVaR
        var_methods=[VaRMethod.CLASSICAL_MC],
        confidence_level_var=0.95,       # 95% VaR
        n_mc_samples=10000,
        distribution="student_t",        # Use Student-t for fat tails
        df_student_t=5.0,                # Degrees of freedom
        solver="scipy",                  # Reliable solver
        output_dir=Path("results/pipeline"),
        run_name="example_cvar"
    )

    pipeline = PortfolioPipeline(config)
    result = pipeline.run(returns, asset_names)

    return result


def run_variance_portfolio(returns, asset_names):
    """Run Variance-focused portfolio optimization."""
    print("\n" + "=" * 80)
    print("RUNNING VARIANCE-FOCUSED PORTFOLIO OPTIMIZATION")
    print("=" * 80)

    config = PipelineConfig(
        optimization_mode=OptimizationMode.VARIANCE,
        cardinality=6,                   # Max 6 assets
        max_weight=0.30,                 # Max 30% per asset
        risk_aversion=1.0,               # Risk aversion parameter
        var_methods=[VaRMethod.CLASSICAL_MC],
        confidence_level_var=0.95,       # 95% VaR
        n_mc_samples=10000,
        distribution="student_t",
        df_student_t=5.0,
        solver="scipy",
        output_dir=Path("results/pipeline"),
        run_name="example_variance"
    )

    pipeline = PortfolioPipeline(config)
    result = pipeline.run(returns, asset_names)

    return result


def compare_portfolios(cvar_result, variance_result):
    """Compare CVaR and Variance portfolios."""
    print("\n" + "=" * 80)
    print("PORTFOLIO COMPARISON: CVaR vs Variance")
    print("=" * 80)

    # Extract metadata
    cvar_meta = cvar_result.metadata
    var_meta = variance_result.metadata

    # Portfolio characteristics
    print("\nPORTFOLIO CHARACTERISTICS")
    print("-" * 80)
    print(f"{'Metric':<25} {'CVaR Portfolio':<20} {'Variance Portfolio':<20}")
    print("-" * 80)
    print(f"{'Active Assets':<25} {cvar_meta['optimization']['active_assets']:<20} "
          f"{var_meta['optimization']['active_assets']:<20}")
    print(f"{'Expected Return':<25} {cvar_meta['portfolio']['expected_return']:<20.6f} "
          f"{var_meta['portfolio']['expected_return']:<20.6f}")
    print(f"{'Volatility':<25} {cvar_meta['portfolio']['volatility']:<20.6f} "
          f"{var_meta['portfolio']['volatility']:<20.6f}")
    print(f"{'Sharpe Ratio':<25} {cvar_meta['portfolio']['sharpe_ratio']:<20.4f} "
          f"{var_meta['portfolio']['sharpe_ratio']:<20.4f}")
    print(f"{'Max Drawdown':<25} {cvar_meta['portfolio']['max_drawdown']:<20.6f} "
          f"{var_meta['portfolio']['max_drawdown']:<20.6f}")

    # VaR metrics
    cvar_var = cvar_meta['var']['classical_mc']
    var_var = var_meta['var']['classical_mc']

    print("\nRISK METRICS (VaR & CVaR)")
    print("-" * 80)
    print(f"{'Metric':<25} {'CVaR Portfolio':<20} {'Variance Portfolio':<20}")
    print("-" * 80)
    print(f"{'VaR_95%':<25} {cvar_var['value']:<20.6f} {var_var['value']:<20.6f}")
    print(f"{'CVaR_95%':<25} {cvar_var['cvar']:<20.6f} {var_var['cvar']:<20.6f}")
    print(f"{'VaR CI (Lower)':<25} {cvar_var['ci_lower']:<20.6f} {var_var['ci_lower']:<20.6f}")
    print(f"{'VaR CI (Upper)':<25} {cvar_var['ci_upper']:<20.6f} {var_var['ci_upper']:<20.6f}")

    # Top holdings comparison
    print("\nTOP 3 HOLDINGS")
    print("-" * 80)
    print(f"{'CVaR Portfolio':<40} {'Variance Portfolio':<40}")
    print("-" * 80)

    cvar_positions = cvar_meta['portfolio']['active_positions'][:3]
    var_positions = var_meta['portfolio']['active_positions'][:3]

    max_rows = max(len(cvar_positions), len(var_positions))
    for i in range(max_rows):
        cvar_str = f"{cvar_positions[i]['asset']}: {cvar_positions[i]['weight']:.2%}" if i < len(cvar_positions) else ""
        var_str = f"{var_positions[i]['asset']}: {var_positions[i]['weight']:.2%}" if i < len(var_positions) else ""
        print(f"{cvar_str:<40} {var_str:<40}")

    # Key insights
    print("\nKEY INSIGHTS")
    print("-" * 80)

    if cvar_var['value'] < var_var['value']:
        print("✓ CVaR portfolio has LOWER tail risk (VaR)")
    else:
        print("✓ Variance portfolio has LOWER tail risk (VaR)")

    if cvar_meta['portfolio']['sharpe_ratio'] > var_meta['portfolio']['sharpe_ratio']:
        print("✓ CVaR portfolio has BETTER risk-adjusted returns (Sharpe)")
    else:
        print("✓ Variance portfolio has BETTER risk-adjusted returns (Sharpe)")

    if cvar_meta['portfolio']['max_drawdown'] > var_meta['portfolio']['max_drawdown']:
        print("✓ Variance portfolio has SMALLER max drawdown")
    else:
        print("✓ CVaR portfolio has SMALLER max drawdown")

    print("\n" + "=" * 80)


def main():
    """Run complete example workflow."""
    print("\n" + "=" * 80)
    print("PORTFOLIO OPTIMIZATION + VaR BACKEND - EXAMPLE WORKFLOW")
    print("=" * 80)

    # Step 1: Generate sample data
    print("\nStep 1: Generating sample return data...")
    returns, asset_names = generate_sample_returns(n_samples=800, n_assets=10)
    print(f"Generated {returns.shape[0]} samples for {returns.shape[1]} assets")
    print(f"Assets: {', '.join(asset_names)}")

    # Step 2: Run CVaR portfolio
    print("\nStep 2: Running CVaR-focused optimization...")
    cvar_result = run_cvar_portfolio(returns, asset_names)

    # Step 3: Run Variance portfolio
    print("\nStep 3: Running Variance-focused optimization...")
    variance_result = run_variance_portfolio(returns, asset_names)

    # Step 4: Compare results
    print("\nStep 4: Comparing portfolios...")
    compare_portfolios(cvar_result, variance_result)

    # Step 5: Show where results are saved
    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print("\nCVaR Portfolio:")
    print(f"  Metadata: results/pipeline/example_cvar/metadata.json")
    print(f"  Report:   results/pipeline/example_cvar/summary_report.txt")
    print("\nVariance Portfolio:")
    print(f"  Metadata: results/pipeline/example_variance/metadata.json")
    print(f"  Report:   results/pipeline/example_variance/summary_report.txt")

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Load metadata JSON files into your UI")
    print("2. Visualize optimal weights, risk metrics, and VaR/CVaR")
    print("3. For real data: replace generate_sample_returns() with your own data")
    print("4. For quantum IQAE: add VaRMethod.IQAE when Classiq/Qiskit integrated")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
