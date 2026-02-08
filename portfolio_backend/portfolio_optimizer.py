"""
Portfolio Optimization Backend - Robust Implementation
Uses QIHD (GPU) + Gurobi Refinement for cardinality-constrained optimization.
Supports CVaR-focused and Variance-focused optimization modes.

Solvers:
- QIHD: Quantum-Inspired Hamiltonian Dynamics (GPU-accelerated)
- Gurobi: For refinement of continuous variables
- CVXPY: For CVaR LP formulation (no cardinality) and comparison

No scipy fallbacks - uses production solvers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, List
from enum import Enum
import json
from pathlib import Path
import time


class OptimizationMode(Enum):
    CVAR = "cvar"
    VARIANCE = "variance"


@dataclass
class PortfolioSpec:
    """Portfolio optimization specification."""
    returns: np.ndarray  # (n_scenarios, n_assets) historical or simulated returns
    mode: OptimizationMode
    cardinality: Optional[int] = None  # Max number of assets (K)
    max_weight: float = 0.3  # Maximum weight per asset
    risk_aversion: float = 1.0  # λ for variance mode
    confidence_level: float = 0.95  # β for CVaR mode
    long_only: bool = True
    solver: str = "qihd"  # "qihd" (default), "cvxpy"

    # QIHD-specific parameters
    qihd_n_shots: int = 200
    qihd_n_steps: int = 5000
    qihd_dt: float = 0.15
    qihd_device: str = "gpu"

    # Gurobi refinement parameters
    gurobi_time_limit: float = 10.0

    def __post_init__(self):
        """Validate and convert mode."""
        if isinstance(self.mode, str):
            self.mode = OptimizationMode(self.mode)
        if self.cardinality is None:
            self.cardinality = self.returns.shape[1]  # Default: all assets


@dataclass
class PortfolioResult:
    """Portfolio optimization result with full metadata."""
    weights: np.ndarray
    selection: np.ndarray  # Binary asset selection
    objective_value: float
    mode: str
    metadata: Dict[str, Any]
    success: bool
    message: str


class PortfolioOptimizer:
    """
    Unified portfolio optimizer supporting CVaR and Variance modes.

    Architecture:
    - QIHD (GPU): Solves MIQP for asset selection + initial weights
    - Gurobi: Refines continuous weights given fixed binary selection
    - CVXPY: Alternative for CVaR LP (no cardinality) or comparison
    """

    def __init__(self, spec: PortfolioSpec):
        self.spec = spec
        self.n_scenarios, self.n_assets = spec.returns.shape

        # Compute statistics
        self.mu = np.mean(spec.returns, axis=0)
        self.cov = np.cov(spec.returns, rowvar=False)
        if self.cov.ndim == 0:
            self.cov = np.array([[self.cov]])

    def optimize(self) -> PortfolioResult:
        """Optimize portfolio based on mode and solver."""
        mode_str = self.spec.mode.value if hasattr(self.spec.mode, 'value') else str(self.spec.mode)

        print(f"\n[Portfolio Optimizer] Mode: {mode_str.upper()}")
        print(f"[Portfolio Optimizer] Assets: {self.n_assets}, Scenarios: {self.n_scenarios}")
        print(f"[Portfolio Optimizer] Cardinality: {self.spec.cardinality}, Max weight: {self.spec.max_weight}")
        print(f"[Portfolio Optimizer] Solver: {self.spec.solver}")

        if mode_str == "variance":
            return self._optimize_variance()
        elif mode_str == "cvar":
            return self._optimize_cvar()
        else:
            raise ValueError(f"Unknown mode: {mode_str}")

    # =========================================================================
    # VARIANCE OPTIMIZATION (Mean-Variance with Cardinality Constraint)
    # =========================================================================

    def _optimize_variance(self) -> PortfolioResult:
        """
        Variance optimization using QIHD + Gurobi.

        Formulation:
            minimize: λ * w^T Σ w - μ^T w
            subject to:
                Σ w_i = 1 (budget)
                w_i >= 0 (long-only)
                w_i <= max_weight * y_i (linking)
                Σ y_i <= K (cardinality)
                y_i ∈ {0,1}
        """
        if self.spec.solver == "qihd":
            return self._optimize_variance_qihd()
        elif self.spec.solver == "cvxpy":
            return self._optimize_variance_cvxpy()
        else:
            raise ValueError(f"Unknown solver: {self.spec.solver}. Use 'qihd' or 'cvxpy'.")

    def _optimize_variance_qihd(self) -> PortfolioResult:
        """Variance optimization using QIHD (GPU) + Gurobi refinement."""
        from portfolio_qihd.miqp_builder import PortfolioSpec as QIHDSpec, build_portfolio_miqp
        from phisolve import QIHD
        from phisolve.phi_miqp import PhiMIQP
        from phisolve.refiners.gurobi import GurobiRefiner

        start_time = time.time()

        # Build MIQP problem
        qihd_spec = QIHDSpec(
            mu=self.mu,
            cov=self.cov,
            upper=np.full(self.n_assets, self.spec.max_weight),
            max_positions=self.spec.cardinality,
            lambda_risk=self.spec.risk_aversion,
            budget=1.0
        )

        miqp = build_portfolio_miqp(qihd_spec)

        # Configure QIHD backend (GPU)
        backend = QIHD(
            n_shots=self.spec.qihd_n_shots,
            n_steps=self.spec.qihd_n_steps,
            dt=self.spec.qihd_dt,
            device=self.spec.qihd_device,
        )

        # Configure Gurobi refiner
        refiner = GurobiRefiner(
            gurobi_options={
                'TimeLimit': self.spec.gurobi_time_limit,
                'OutputFlag': 0,
                'Threads': 8
            }
        )

        # Solve
        solver = PhiMIQP(
            problem_instance=miqp,
            backend=backend,
            refiner=refiner,
        )

        print(f"[QIHD] Running with {self.spec.qihd_n_shots} shots, {self.spec.qihd_n_steps} steps on {self.spec.qihd_device.upper()}")
        response = solver.solve(if_refine=True)

        total_time = time.time() - start_time

        # Extract results
        solution = response.minimizer
        selection = solution[:self.n_assets]
        weights = solution[self.n_assets:]

        # Ensure proper binary values
        selection = (selection > 0.5).astype(float)

        # Zero out weights for unselected assets
        weights = weights * selection

        # Renormalize if needed
        if weights.sum() > 0 and not np.isclose(weights.sum(), 1.0, atol=1e-4):
            weights = weights / weights.sum()

        return PortfolioResult(
            weights=weights,
            selection=selection,
            objective_value=float(response.minimum()),
            mode="variance",
            metadata={
                "expected_return": float(np.dot(self.mu, weights)),
                "portfolio_variance": float(weights @ self.cov @ weights),
                "portfolio_volatility": float(np.sqrt(weights @ self.cov @ weights)),
                "risk_aversion": self.spec.risk_aversion,
                "solver": "qihd+gurobi",
                "device": self.spec.qihd_device,
                "qihd_time": response.detailed_time.get("QIHD_Time", 0),
                "refinement_time": response.detailed_time.get("Refinement", 0),
                "total_time": total_time,
                "n_shots": self.spec.qihd_n_shots,
                "n_steps": self.spec.qihd_n_steps,
                "active_assets": int(np.sum(selection)),
                "selected_indices": np.where(selection > 0.5)[0].tolist(),
            },
            success=True,
            message=f"Variance optimization completed in {total_time:.2f}s (QIHD+Gurobi)"
        )

    def _optimize_variance_cvxpy(self) -> PortfolioResult:
        """Variance optimization using CVXPY (for comparison, no cardinality)."""
        import cvxpy as cp

        start_time = time.time()

        w = cp.Variable(self.n_assets)

        # Mean-variance objective: λ * w^T Σ w - μ^T w
        objective = self.spec.risk_aversion * cp.quad_form(w, self.cov) - self.mu @ w

        constraints = [
            cp.sum(w) == 1,  # Budget
            w >= 0,  # Long-only
            w <= self.spec.max_weight  # Position limits
        ]

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        total_time = time.time() - start_time

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            weights = w.value

            # Apply cardinality by zeroing smallest weights
            if self.spec.cardinality < self.n_assets:
                K = self.spec.cardinality
                threshold = np.partition(weights, -K)[-K]
                weights[weights < threshold] = 0
                weights = weights / weights.sum()

            selection = (weights > 1e-6).astype(float)

            return PortfolioResult(
                weights=weights,
                selection=selection,
                objective_value=float(objective.value),
                mode="variance",
                metadata={
                    "expected_return": float(np.dot(self.mu, weights)),
                    "portfolio_variance": float(weights @ self.cov @ weights),
                    "portfolio_volatility": float(np.sqrt(weights @ self.cov @ weights)),
                    "solver": "cvxpy",
                    "cvxpy_solver": "CLARABEL",
                    "total_time": total_time,
                    "active_assets": int(np.sum(selection)),
                    "note": "Cardinality enforced post-hoc (not in optimization)"
                },
                success=True,
                message=f"Variance optimization completed in {total_time:.2f}s (CVXPY)"
            )
        else:
            raise RuntimeError(f"CVXPY optimization failed: {problem.status}")

    # =========================================================================
    # CVaR OPTIMIZATION (Conditional Value-at-Risk)
    # =========================================================================

    def _optimize_cvar(self) -> PortfolioResult:
        """
        CVaR optimization.

        For cardinality-constrained CVaR: Use QIHD with tail-focused objective
        For standard CVaR (no cardinality): Use CVXPY LP formulation
        """
        if self.spec.solver == "qihd":
            return self._optimize_cvar_qihd()
        elif self.spec.solver == "cvxpy":
            return self._optimize_cvar_cvxpy()
        else:
            raise ValueError(f"Unknown solver: {self.spec.solver}")

    def _optimize_cvar_qihd(self) -> PortfolioResult:
        """
        CVaR optimization using QIHD with tail-focused covariance.

        Approach: Minimize variance computed on tail scenarios (worst α%)
        This approximates CVaR minimization with cardinality constraints.
        """
        from portfolio_qihd.miqp_builder import PortfolioSpec as QIHDSpec, build_portfolio_miqp
        from phisolve import QIHD
        from phisolve.phi_miqp import PhiMIQP
        from phisolve.refiners.gurobi import GurobiRefiner

        start_time = time.time()

        # Identify tail scenarios (worst α%)
        alpha = 1 - self.spec.confidence_level
        portfolio_returns_eq = self.spec.returns.mean(axis=1)  # Equal-weighted proxy
        threshold = np.quantile(portfolio_returns_eq, alpha)
        tail_mask = portfolio_returns_eq <= threshold

        n_tail = tail_mask.sum()
        print(f"[CVaR] Using {n_tail} tail scenarios ({alpha*100:.0f}% worst)")

        # Compute tail-focused statistics
        if n_tail > 1:
            tail_returns = self.spec.returns[tail_mask]
            tail_mu = np.mean(tail_returns, axis=0)
            tail_cov = np.cov(tail_returns, rowvar=False)
            if tail_cov.ndim == 0:
                tail_cov = np.array([[tail_cov]])
        else:
            tail_mu = self.mu
            tail_cov = self.cov

        # Build MIQP with tail statistics
        qihd_spec = QIHDSpec(
            mu=tail_mu,
            cov=tail_cov,
            upper=np.full(self.n_assets, self.spec.max_weight),
            max_positions=self.spec.cardinality,
            lambda_risk=2.0,  # Higher risk aversion for tail focus
            budget=1.0
        )

        miqp = build_portfolio_miqp(qihd_spec)

        # Configure QIHD backend (GPU)
        backend = QIHD(
            n_shots=self.spec.qihd_n_shots,
            n_steps=self.spec.qihd_n_steps,
            dt=self.spec.qihd_dt,
            device=self.spec.qihd_device,
        )

        # Configure Gurobi refiner
        refiner = GurobiRefiner(
            gurobi_options={
                'TimeLimit': self.spec.gurobi_time_limit,
                'OutputFlag': 0,
                'Threads': 8
            }
        )

        # Solve
        solver = PhiMIQP(
            problem_instance=miqp,
            backend=backend,
            refiner=refiner,
        )

        print(f"[QIHD] Running CVaR approximation on {self.spec.qihd_device.upper()}")
        response = solver.solve(if_refine=True)

        total_time = time.time() - start_time

        # Extract results
        solution = response.minimizer
        selection = (solution[:self.n_assets] > 0.5).astype(float)
        weights = solution[self.n_assets:] * selection

        if weights.sum() > 0:
            weights = weights / weights.sum()

        # Compute actual CVaR for the optimized portfolio
        portfolio_losses = -self.spec.returns @ weights
        var_value = np.quantile(portfolio_losses, self.spec.confidence_level)
        cvar_value = np.mean(portfolio_losses[portfolio_losses >= var_value])

        return PortfolioResult(
            weights=weights,
            selection=selection,
            objective_value=float(cvar_value),
            mode="cvar",
            metadata={
                "cvar": float(cvar_value),
                "var": float(var_value),
                "confidence_level": self.spec.confidence_level,
                "tail_scenarios": int(n_tail),
                "expected_return": float(np.dot(self.mu, weights)),
                "portfolio_volatility": float(np.sqrt(weights @ self.cov @ weights)),
                "solver": "qihd+gurobi",
                "device": self.spec.qihd_device,
                "qihd_time": response.detailed_time.get("QIHD_Time", 0),
                "refinement_time": response.detailed_time.get("Refinement", 0),
                "total_time": total_time,
                "active_assets": int(np.sum(selection)),
                "selected_indices": np.where(selection > 0.5)[0].tolist(),
                "approximation": "tail_variance_minimization"
            },
            success=True,
            message=f"CVaR optimization completed in {total_time:.2f}s (QIHD+Gurobi)"
        )

    def _optimize_cvar_cvxpy(self) -> PortfolioResult:
        """
        CVaR optimization using CVXPY LP formulation.

        Formulation:
            minimize: α + (1/(1-β)) * (1/S) * Σ u_s
            subject to:
                u_s >= L^(s)(w) - α  ∀s (excess loss)
                u_s >= 0             ∀s
                Σ w_i = 1            (budget)
                w_i >= 0             (long-only)
                w_i <= max_weight    (position limits)

        Note: No cardinality constraint (use QIHD for that)
        """
        import cvxpy as cp

        start_time = time.time()

        S = self.n_scenarios
        beta = self.spec.confidence_level

        # Decision variables
        w = cp.Variable(self.n_assets)  # Portfolio weights
        alpha = cp.Variable()  # VaR threshold
        u = cp.Variable(S)  # Excess losses

        # Portfolio losses in each scenario: L^(s) = -r^(s) @ w
        losses = -self.spec.returns @ w  # (S,)

        # CVaR objective
        cvar_obj = alpha + (1 / (1 - beta)) * cp.sum(u) / S

        # Constraints
        constraints = [
            u >= losses - alpha,  # Excess loss definition
            u >= 0,  # Non-negative excess
            cp.sum(w) == 1,  # Budget
            w >= 0,  # Long-only
            w <= self.spec.max_weight  # Position limits
        ]

        problem = cp.Problem(cp.Minimize(cvar_obj), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        total_time = time.time() - start_time

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            weights = w.value

            # Apply cardinality post-hoc
            if self.spec.cardinality < self.n_assets:
                K = self.spec.cardinality
                threshold = np.partition(weights, -K)[-K]
                weights[weights < threshold] = 0
                weights = weights / weights.sum()

            selection = (weights > 1e-6).astype(float)

            # Recompute CVaR for final portfolio
            portfolio_losses = -self.spec.returns @ weights
            var_value = np.quantile(portfolio_losses, self.spec.confidence_level)
            cvar_value = np.mean(portfolio_losses[portfolio_losses >= var_value])

            return PortfolioResult(
                weights=weights,
                selection=selection,
                objective_value=float(cvar_value),
                mode="cvar",
                metadata={
                    "cvar": float(cvar_value),
                    "var": float(var_value),
                    "var_threshold_opt": float(alpha.value),
                    "confidence_level": self.spec.confidence_level,
                    "expected_return": float(np.dot(self.mu, weights)),
                    "solver": "cvxpy",
                    "cvxpy_solver": "CLARABEL",
                    "total_time": total_time,
                    "active_assets": int(np.sum(selection)),
                    "note": "Cardinality enforced post-hoc"
                },
                success=True,
                message=f"CVaR optimization completed in {total_time:.2f}s (CVXPY)"
            )
        else:
            raise RuntimeError(f"CVXPY optimization failed: {problem.status}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_portfolio_result(result: PortfolioResult, filepath: Path):
    """Save portfolio result to JSON."""
    data = {
        "weights": result.weights.tolist(),
        "selection": result.selection.tolist(),
        "objective_value": float(result.objective_value),
        "mode": result.mode,
        "metadata": result.metadata,
        "success": result.success,
        "message": result.message
    }
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[Portfolio Optimizer] Result saved to {filepath}")


def load_portfolio_result(filepath: Path) -> PortfolioResult:
    """Load portfolio result from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return PortfolioResult(
        weights=np.array(data["weights"]),
        selection=np.array(data["selection"]),
        objective_value=data["objective_value"],
        mode=data["mode"],
        metadata=data["metadata"],
        success=data["success"],
        message=data["message"]
    )
