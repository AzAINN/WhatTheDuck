"""
Portfolio Optimization + VaR Backend
=====================================

Robust implementation using:
- QIHD (GPU): Quantum-Inspired Hamiltonian Dynamics for MIQP
- Gurobi: Refinement of continuous variables
- CVXPY: CVaR LP formulation alternative

Components:
- portfolio_optimizer: CVaR and Variance optimization
- var_calculator: Classical MC and IQAE VaR estimation
- portfolio_pipeline: End-to-end orchestration

Quick Start:
    from portfolio_backend import (
        PortfolioOptimizer, PortfolioSpec, OptimizationMode,
        VaRCalculator, VaRSpec, VaRMethod
    )

    # Optimize portfolio
    spec = PortfolioSpec(
        returns=returns,
        mode=OptimizationMode.CVAR,
        cardinality=10,
        max_weight=0.25,
        solver='qihd'
    )
    result = PortfolioOptimizer(spec).optimize()

    # Calculate VaR
    var_spec = VaRSpec(
        portfolio_weights=result.weights,
        returns_data=returns,
        method=VaRMethod.CLASSICAL_MC
    )
    var_result = VaRCalculator(var_spec).calculate()
"""

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

from portfolio_backend.portfolio_pipeline import (
    PortfolioPipeline,
    PipelineConfig,
    PipelineResult,
    load_pipeline_result
)

__all__ = [
    # Optimization
    'PortfolioOptimizer',
    'PortfolioSpec',
    'PortfolioResult',
    'OptimizationMode',
    'save_portfolio_result',
    'load_portfolio_result',
    # VaR
    'VaRCalculator',
    'VaRSpec',
    'VaRResult',
    'VaRMethod',
    'save_var_result',
    'load_var_result',
    'compare_var_methods',
    # Pipeline
    'PortfolioPipeline',
    'PipelineConfig',
    'PipelineResult',
    'load_pipeline_result',
]

__version__ = '2.0.0'
