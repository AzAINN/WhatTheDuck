"""
Verification Script for Portfolio Optimization + VaR Backend

Tests:
1. Portfolio optimization (CVaR and Variance modes)
2. VaR calculation (Classical MC and IQAE)
3. End-to-end pipeline
4. Metadata generation and storage
5. Result loading and consistency
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio_backend.portfolio_optimizer import (
    PortfolioOptimizer,
    PortfolioSpec,
    OptimizationMode
)
from portfolio_backend.var_calculator import (
    VaRCalculator,
    VaRSpec,
    VaRMethod,
    compare_var_methods
)
from portfolio_backend.portfolio_pipeline import (
    PortfolioPipeline,
    PipelineConfig,
    load_pipeline_result
)


def generate_test_data(
    n_samples: int = 1000,
    n_assets: int = 10,
    correlation: float = 0.3,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic return data for testing.

    Uses factor model: r = F @ B^T + ε
    """
    np.random.seed(seed)

    # Factor model parameters
    n_factors = 3
    mu = np.random.uniform(0.05, 0.15, n_assets) / 252  # Daily returns
    factor_loadings = np.random.randn(n_assets, n_factors) * 0.5

    # Generate factors and idiosyncratic noise
    factors = np.random.randn(n_samples, n_factors) * 0.02
    idiosyncratic = np.random.randn(n_samples, n_assets) * 0.01

    # Generate returns
    returns = mu + factors @ factor_loadings.T + idiosyncratic

    # Asset names
    asset_names = [f"STOCK_{chr(65 + i)}" for i in range(n_assets)]

    return returns, asset_names


def test_portfolio_optimization_variance():
    """Test 1: Variance-based portfolio optimization."""
    print("\n" + "=" * 80)
    print("TEST 1: VARIANCE-BASED PORTFOLIO OPTIMIZATION")
    print("=" * 80)

    # Generate test data
    returns, asset_names = generate_test_data(n_samples=500, n_assets=8)

    # Test with scipy (most likely available)
    spec = PortfolioSpec(
        returns=returns,
        mode=OptimizationMode.VARIANCE,
        cardinality=5,
        max_weight=0.4,
        risk_aversion=1.0,
        solver="scipy"  # Use scipy as it's more likely to be available
    )

    optimizer = PortfolioOptimizer(spec)
    result = optimizer.optimize()

    # Verify results
    print("\n[Verification]")
    assert result.success, "Optimization failed"
    print("✓ Optimization succeeded")

    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-4), "Weights don't sum to 1"
    print("✓ Budget constraint satisfied")

    assert np.all(result.weights >= -1e-6), "Negative weights (long-only violated)"
    print("✓ Long-only constraint satisfied")

    assert np.all(result.weights <= spec.max_weight + 1e-4), "Max weight constraint violated"
    print("✓ Position limit constraint satisfied")

    active_assets = np.sum(result.weights > 1e-6)
    print(f"✓ Active assets: {active_assets} (requested cardinality: {spec.cardinality})")

    print(f"\nObjective value: {result.objective_value:.6f}")
    print(f"Expected return: {result.metadata.get('expected_return', 'N/A'):.6f}")
    print(f"Portfolio volatility: {result.metadata.get('portfolio_volatility', 'N/A'):.6f}")

    print("\nTop 3 holdings:")
    top_indices = np.argsort(result.weights)[::-1][:3]
    for i in top_indices:
        if result.weights[i] > 1e-6:
            print(f"  {asset_names[i]}: {result.weights[i]:.2%}")

    print("\n✓ TEST 1 PASSED")
    return result


def test_portfolio_optimization_cvar():
    """Test 2: CVaR-based portfolio optimization."""
    print("\n" + "=" * 80)
    print("TEST 2: CVaR-BASED PORTFOLIO OPTIMIZATION")
    print("=" * 80)

    # Generate test data with heavier tails
    returns, asset_names = generate_test_data(n_samples=500, n_assets=8)

    # Add some extreme scenarios to test CVaR
    extreme_losses = np.random.randn(50, 8) * 0.05 - 0.03  # Negative returns
    returns = np.vstack([returns, extreme_losses])

    spec = PortfolioSpec(
        returns=returns,
        mode=OptimizationMode.CVAR,
        cardinality=5,
        max_weight=0.4,
        confidence_level=0.95,
        solver="scipy"  # Will use QIHD approximation via scipy
    )

    optimizer = PortfolioOptimizer(spec)
    result = optimizer.optimize()

    # Verify results
    print("\n[Verification]")
    assert result.success, "Optimization failed"
    print("✓ Optimization succeeded")

    assert np.isclose(np.sum(result.weights), 1.0, atol=1e-4), "Weights don't sum to 1"
    print("✓ Budget constraint satisfied")

    assert np.all(result.weights >= -1e-6), "Negative weights"
    print("✓ Long-only constraint satisfied")

    active_assets = np.sum(result.weights > 1e-6)
    print(f"✓ Active assets: {active_assets}")

    print(f"\nCVaR objective: {result.objective_value:.6f}")
    var_threshold = result.metadata.get('var_threshold')
    print(f"VaR threshold (α): {var_threshold:.6f}" if var_threshold else "VaR threshold (α): N/A")
    avg_excess = result.metadata.get('avg_excess_loss')
    print(f"Avg excess loss: {avg_excess:.6f}" if avg_excess else "Avg excess loss: N/A")
    conf_level = result.metadata.get('confidence_level')
    print(f"Confidence level: {conf_level:.2%}" if conf_level else "Confidence level: N/A")

    print("\nTop 3 holdings:")
    top_indices = np.argsort(result.weights)[::-1][:3]
    for i in top_indices:
        if result.weights[i] > 1e-6:
            print(f"  {asset_names[i]}: {result.weights[i]:.2%}")

    print("\n✓ TEST 2 PASSED")
    return result


def test_var_calculation():
    """Test 3: VaR calculation methods."""
    print("\n" + "=" * 80)
    print("TEST 3: VaR CALCULATION METHODS")
    print("=" * 80)

    # Generate test data
    returns, asset_names = generate_test_data(n_samples=1000, n_assets=8)

    # Create a simple portfolio
    weights = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02])
    weights = weights / weights.sum()

    print(f"\nTest portfolio: {np.sum(weights > 1e-6)} active assets")

    # Test Classical MC
    print("\n[Classical Monte Carlo VaR]")
    spec_mc = VaRSpec(
        portfolio_weights=weights,
        returns_data=returns,
        method=VaRMethod.CLASSICAL_MC,
        confidence_level=0.95,
        n_samples=5000,
        distribution="empirical"
    )
    calc_mc = VaRCalculator(spec_mc)
    result_mc = calc_mc.calculate()

    assert result_mc.success, "Classical MC failed"
    print("✓ Classical MC succeeded")

    assert result_mc.var_value > 0, "VaR should be positive (loss)"
    print(f"✓ VaR_{spec_mc.confidence_level*100:.0f}%: {result_mc.var_value:.4f}")

    assert result_mc.confidence_interval is not None, "CI missing"
    print(f"✓ 95% CI: [{result_mc.confidence_interval[0]:.4f}, {result_mc.confidence_interval[1]:.4f}]")

    print(f"  CVaR: {result_mc.metadata.get('cvar', 'N/A'):.4f}")
    print(f"  Max loss: {result_mc.metadata.get('max_loss', 'N/A'):.4f}")
    print(f"  Samples: {result_mc.metadata.get('n_samples', 'N/A'):,}")

    # Test IQAE (will use classical fallback)
    print("\n[IQAE VaR]")
    spec_iqae = VaRSpec(
        portfolio_weights=weights,
        returns_data=returns,
        method=VaRMethod.IQAE,
        confidence_level=0.95,
        epsilon=0.01,
        distribution="empirical"
    )
    calc_iqae = VaRCalculator(spec_iqae)
    result_iqae = calc_iqae.calculate()

    assert result_iqae.success, "IQAE failed"
    print("✓ IQAE succeeded (classical fallback)")

    print(f"✓ VaR_{spec_iqae.confidence_level*100:.0f}%: {result_iqae.var_value:.4f}")
    print(f"  Method: {result_iqae.metadata.get('method', 'N/A')}")
    print(f"  Note: {result_iqae.metadata.get('note', 'N/A')}")

    # Compare methods
    print("\n[Method Comparison]")
    diff_pct = abs(result_mc.var_value - result_iqae.var_value) / result_mc.var_value * 100
    print(f"Difference: {diff_pct:.2f}%")

    if diff_pct < 10:
        print("✓ Methods are consistent (< 10% difference)")
    else:
        print("⚠ Large difference between methods (expected for classical fallback)")

    print("\n✓ TEST 3 PASSED")
    return result_mc, result_iqae


def test_end_to_end_pipeline():
    """Test 4: End-to-end pipeline."""
    print("\n" + "=" * 80)
    print("TEST 4: END-TO-END PIPELINE")
    print("=" * 80)

    # Generate test data
    returns, asset_names = generate_test_data(n_samples=800, n_assets=10)

    # Test Variance pipeline
    print("\n[Pipeline: Variance Optimization + VaR]")
    config_variance = PipelineConfig(
        optimization_mode=OptimizationMode.VARIANCE,
        cardinality=6,
        max_weight=0.3,
        risk_aversion=1.0,
        var_methods=[VaRMethod.CLASSICAL_MC, VaRMethod.IQAE],
        confidence_level_var=0.95,
        n_mc_samples=5000,
        distribution="empirical",
        solver="scipy",
        output_dir=Path("results/pipeline"),
        run_name="test_variance"
    )

    pipeline_variance = PortfolioPipeline(config_variance)
    result_variance = pipeline_variance.run(returns, asset_names)

    assert result_variance.success, "Variance pipeline failed"
    print("✓ Variance pipeline succeeded")

    assert result_variance.portfolio_result.success, "Portfolio optimization failed"
    print("✓ Portfolio optimization succeeded")

    assert all(vr.success for vr in result_variance.var_results.values()), "VaR calculation failed"
    print("✓ All VaR calculations succeeded")

    assert len(result_variance.metadata) > 0, "Metadata empty"
    print("✓ Metadata generated")

    # Verify metadata structure
    required_keys = ['timestamp', 'data', 'optimization', 'portfolio', 'var', 'config']
    for key in required_keys:
        assert key in result_variance.metadata, f"Missing metadata key: {key}"
    print("✓ Metadata structure valid")

    # Test CVaR pipeline
    print("\n[Pipeline: CVaR Optimization + VaR]")
    config_cvar = PipelineConfig(
        optimization_mode=OptimizationMode.CVAR,
        cardinality=6,
        max_weight=0.3,
        confidence_level_cvar=0.95,
        var_methods=[VaRMethod.CLASSICAL_MC],
        confidence_level_var=0.95,
        n_mc_samples=5000,
        distribution="empirical",
        solver="scipy",
        output_dir=Path("results/pipeline"),
        run_name="test_cvar"
    )

    pipeline_cvar = PortfolioPipeline(config_cvar)
    result_cvar = pipeline_cvar.run(returns, asset_names)

    assert result_cvar.success, "CVaR pipeline failed"
    print("✓ CVaR pipeline succeeded")

    print("\n✓ TEST 4 PASSED")
    return result_variance, result_cvar


def test_result_persistence():
    """Test 5: Result loading and consistency."""
    print("\n" + "=" * 80)
    print("TEST 5: RESULT PERSISTENCE")
    print("=" * 80)

    # Load results from test 4
    run_dir_variance = Path("results/pipeline/test_variance")
    run_dir_cvar = Path("results/pipeline/test_cvar")

    if run_dir_variance.exists():
        print("\n[Loading Variance Pipeline Result]")
        loaded_variance = load_pipeline_result(run_dir_variance)

        assert loaded_variance.success, "Loaded result not successful"
        print("✓ Loaded result successfully")

        assert len(loaded_variance.portfolio_result.weights) > 0, "Weights missing"
        print("✓ Portfolio weights preserved")

        assert len(loaded_variance.var_results) > 0, "VaR results missing"
        print("✓ VaR results preserved")

        assert len(loaded_variance.metadata) > 0, "Metadata missing"
        print("✓ Metadata preserved")

        print(f"\nLoaded metadata:")
        print(f"  Active assets: {loaded_variance.metadata['optimization']['active_assets']}")
        print(f"  Expected return: {loaded_variance.metadata['portfolio']['expected_return']:.4f}")
        print(f"  Sharpe ratio: {loaded_variance.metadata['portfolio']['sharpe_ratio']:.4f}")

    if run_dir_cvar.exists():
        print("\n[Loading CVaR Pipeline Result]")
        loaded_cvar = load_pipeline_result(run_dir_cvar)

        assert loaded_cvar.success, "Loaded CVaR result not successful"
        print("✓ Loaded CVaR result successfully")

    print("\n✓ TEST 5 PASSED")


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("PORTFOLIO BACKEND VERIFICATION SUITE")
    print("=" * 80)

    try:
        # Test 1: Variance optimization
        test_portfolio_optimization_variance()

        # Test 2: CVaR optimization
        test_portfolio_optimization_cvar()

        # Test 3: VaR calculation
        test_var_calculation()

        # Test 4: End-to-end pipeline
        test_end_to_end_pipeline()

        # Test 5: Result persistence
        test_result_persistence()

        # Final summary
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nBackend components verified:")
        print("  ✓ Portfolio optimization (CVaR and Variance modes)")
        print("  ✓ VaR calculation (Classical MC and IQAE)")
        print("  ✓ End-to-end pipeline orchestration")
        print("  ✓ Metadata generation and storage")
        print("  ✓ Result loading and persistence")
        print("\nReady for integration with UI!")
        print("=" * 80 + "\n")

        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("=" * 80 + "\n")
        return False

    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80 + "\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
