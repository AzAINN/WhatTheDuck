"""
Value-at-Risk (VaR) Calculator Backend
Supports Classical Monte Carlo and IQAE methods for portfolio VaR estimation.

VaR_α: The loss level that will not be exceeded with probability α
Example: VaR_95% = 2.5% means 95% confident that daily loss < 2.5%
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import json
from pathlib import Path
import time


class VaRMethod(Enum):
    CLASSICAL_MC = "classical_mc"
    IQAE = "iqae"


@dataclass
class VaRSpec:
    """VaR calculation specification."""
    portfolio_weights: np.ndarray  # (n_assets,)
    returns_data: np.ndarray  # (n_samples, n_assets) - historical or simulated
    method: VaRMethod
    confidence_level: float = 0.95  # α (e.g., 95% VaR)
    n_samples: int = 10000  # Number of MC samples
    distribution: str = "empirical"  # "empirical", "normal", "student_t"
    df: Optional[float] = None  # Degrees of freedom for Student-t

    # IQAE-specific parameters
    iqae_epsilon: float = 0.01
    iqae_alpha_fail: float = 0.05
    iqae_n_qubits: int = 8


@dataclass
class VaRResult:
    """VaR calculation result with full metadata."""
    var_value: float
    cvar_value: float  # Conditional VaR (Expected Shortfall)
    method: str
    confidence_level: float
    confidence_interval: Optional[Tuple[float, float]]
    metadata: Dict[str, Any]
    success: bool
    message: str


class VaRCalculator:
    """
    VaR calculator supporting Classical MC and IQAE methods.

    Architecture:
    - Classical MC: Bootstrap-based with multiple distribution options
    - IQAE: Quantum amplitude estimation (requires Classiq/Qiskit)
    """

    def __init__(self, spec: VaRSpec):
        self.spec = spec
        self.n_assets = len(spec.portfolio_weights)

        # Normalize weights
        weight_sum = np.sum(spec.portfolio_weights)
        if not np.isclose(weight_sum, 1.0):
            self.spec.portfolio_weights = spec.portfolio_weights / weight_sum

    def calculate(self) -> VaRResult:
        """Calculate VaR based on method."""
        method_str = self.spec.method.value if hasattr(self.spec.method, 'value') else str(self.spec.method)

        print(f"\n[VaR Calculator] Method: {method_str}")
        print(f"[VaR Calculator] Confidence: {self.spec.confidence_level}")
        print(f"[VaR Calculator] Distribution: {self.spec.distribution}")

        if method_str == "classical_mc":
            return self._calculate_classical_mc()
        elif method_str == "iqae":
            return self._calculate_iqae()
        else:
            raise ValueError(f"Unknown VaR method: {method_str}")

    def _calculate_classical_mc(self) -> VaRResult:
        """
        Classical Monte Carlo VaR estimation.

        Process:
        1. Generate scenarios from distribution
        2. Compute portfolio returns: r_p = Σ w_i * r_i
        3. Compute losses: L = -r_p
        4. VaR_α = quantile(L, α)
        5. CVaR_α = E[L | L >= VaR_α]
        """
        start_time = time.time()

        # Generate scenarios
        scenarios = self._generate_scenarios(self.spec.n_samples)
        n_scenarios = scenarios.shape[0]

        # Compute portfolio returns and losses
        portfolio_returns = scenarios @ self.spec.portfolio_weights
        portfolio_losses = -portfolio_returns

        # Calculate VaR
        var_value = np.quantile(portfolio_losses, self.spec.confidence_level)

        # Calculate CVaR (Expected Shortfall)
        tail_losses = portfolio_losses[portfolio_losses >= var_value]
        cvar_value = np.mean(tail_losses) if len(tail_losses) > 0 else var_value

        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(portfolio_losses)

        elapsed_time = time.time() - start_time

        return VaRResult(
            var_value=float(var_value),
            cvar_value=float(cvar_value),
            method="classical_mc",
            confidence_level=self.spec.confidence_level,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            metadata={
                "n_samples": n_scenarios,
                "distribution": self.spec.distribution,
                "max_loss": float(np.max(portfolio_losses)),
                "min_loss": float(np.min(portfolio_losses)),
                "mean_loss": float(np.mean(portfolio_losses)),
                "std_loss": float(np.std(portfolio_losses)),
                "elapsed_time": elapsed_time,
                "tail_count": int(len(tail_losses)),
                "convergence_error": float(np.std(portfolio_losses) / np.sqrt(n_scenarios))
            },
            success=True,
            message=f"Classical MC VaR completed with {n_scenarios:,} samples in {elapsed_time:.2f}s"
        )

    def _calculate_iqae(self) -> VaRResult:
        """
        IQAE VaR estimation using quantum amplitude estimation.

        This uses the hyperparameter-sweep IQAE implementation if available,
        otherwise provides a classical simulation with IQAE-like interface.
        """
        start_time = time.time()

        try:
            # Try to use the real IQAE implementation
            return self._calculate_iqae_real()
        except ImportError as e:
            print(f"[VaR Calculator] IQAE modules not available: {e}")
            print("[VaR Calculator] Using classical simulation with IQAE interface")
            return self._calculate_iqae_simulated()

    def _calculate_iqae_real(self) -> VaRResult:
        """Real IQAE using Classiq/Qiskit."""
        # Import from hyperparameter-sweep
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "hyperparameter-sweep"))

        from quantum_estimator import QuantumIQAECDF, ProbEstimate

        start_time = time.time()

        # Generate portfolio return distribution
        scenarios = self._generate_scenarios(10000)
        portfolio_returns = scenarios @ self.spec.portfolio_weights

        # Discretize for quantum encoding
        n_bins = 2 ** self.spec.iqae_n_qubits
        pmf, grid = self._discretize_distribution(portfolio_returns, n_bins)

        # Convert to losses
        loss_grid = -grid[::-1]
        loss_pmf = pmf[::-1]

        # Create quantum estimator
        estimator = QuantumIQAECDF(loss_pmf.tolist(), self.spec.iqae_n_qubits)

        # Binary search for VaR
        var_value, total_queries = self._iqae_bisection(
            estimator, loss_grid, loss_pmf,
            target_prob=1 - self.spec.confidence_level
        )

        # Compute CVaR classically
        portfolio_losses = -portfolio_returns
        cvar_value = np.mean(portfolio_losses[portfolio_losses >= var_value])

        elapsed_time = time.time() - start_time

        return VaRResult(
            var_value=float(var_value),
            cvar_value=float(cvar_value),
            method="iqae",
            confidence_level=self.spec.confidence_level,
            confidence_interval=None,  # IQAE provides different CI structure
            metadata={
                "epsilon": self.spec.iqae_epsilon,
                "alpha_fail": self.spec.iqae_alpha_fail,
                "n_qubits": self.spec.iqae_n_qubits,
                "total_oracle_queries": total_queries,
                "elapsed_time": elapsed_time,
                "grid_size": n_bins
            },
            success=True,
            message=f"IQAE VaR completed in {elapsed_time:.2f}s with {total_queries} oracle queries"
        )

    def _calculate_iqae_simulated(self) -> VaRResult:
        """Simulated IQAE for when quantum libraries aren't available."""
        start_time = time.time()

        # Use classical calculation but report as IQAE simulation
        scenarios = self._generate_scenarios(self.spec.n_samples)
        portfolio_returns = scenarios @ self.spec.portfolio_weights
        portfolio_losses = -portfolio_returns

        # Discretize to simulate quantum grid
        n_bins = 2 ** self.spec.iqae_n_qubits
        pmf, grid = self._discretize_distribution(portfolio_losses, n_bins)

        # Find VaR using discretized distribution
        cumsum = np.cumsum(pmf)
        var_idx = np.searchsorted(cumsum, self.spec.confidence_level)
        var_value = grid[min(var_idx, len(grid) - 1)]

        # CVaR
        cvar_value = np.mean(portfolio_losses[portfolio_losses >= var_value])

        # Simulate IQAE query count
        simulated_queries = int(np.log2(1 / self.spec.iqae_epsilon) * 100)

        elapsed_time = time.time() - start_time

        return VaRResult(
            var_value=float(var_value),
            cvar_value=float(cvar_value),
            method="iqae_simulated",
            confidence_level=self.spec.confidence_level,
            confidence_interval=None,
            metadata={
                "epsilon": self.spec.iqae_epsilon,
                "alpha_fail": self.spec.iqae_alpha_fail,
                "n_qubits": self.spec.iqae_n_qubits,
                "simulated_oracle_queries": simulated_queries,
                "elapsed_time": elapsed_time,
                "grid_size": n_bins,
                "note": "Simulated IQAE (Classiq/Qiskit not available)"
            },
            success=True,
            message=f"Simulated IQAE VaR completed in {elapsed_time:.2f}s"
        )

    def _generate_scenarios(self, n_samples: int) -> np.ndarray:
        """Generate return scenarios based on distribution type."""
        if self.spec.distribution == "empirical":
            # Bootstrap from historical data
            indices = np.random.choice(len(self.spec.returns_data), size=n_samples, replace=True)
            return self.spec.returns_data[indices]

        elif self.spec.distribution == "normal":
            mu = np.mean(self.spec.returns_data, axis=0)
            cov = np.cov(self.spec.returns_data, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            return np.random.multivariate_normal(mu, cov, size=n_samples)

        elif self.spec.distribution == "student_t":
            from scipy.stats import chi2
            df = self.spec.df if self.spec.df else 5.0

            mu = np.mean(self.spec.returns_data, axis=0)
            cov = np.cov(self.spec.returns_data, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])

            normal_samples = np.random.multivariate_normal(mu, cov, size=n_samples)
            chi_sq = chi2.rvs(df, size=n_samples)
            scale = np.sqrt(df / chi_sq)
            return mu + (normal_samples - mu) * scale[:, np.newaxis]

        else:
            raise ValueError(f"Unknown distribution: {self.spec.distribution}")

    def _discretize_distribution(self, data: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Discretize data into n_bins for quantum encoding."""
        # Use quantile-based binning
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(data, quantiles)

        # Compute histogram
        counts, _ = np.histogram(data, bins=bin_edges)
        pmf = counts / counts.sum()

        # Grid points (bin centers)
        grid = (bin_edges[:-1] + bin_edges[1:]) / 2

        return pmf, grid

    def _bootstrap_ci(self, losses: np.ndarray, n_bootstrap: int = 1000, ci_level: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval for VaR."""
        bootstrap_vars = []
        n = len(losses)

        for _ in range(n_bootstrap):
            sample = np.random.choice(losses, size=n, replace=True)
            bootstrap_vars.append(np.quantile(sample, self.spec.confidence_level))

        alpha = 1 - ci_level
        return (
            np.quantile(bootstrap_vars, alpha / 2),
            np.quantile(bootstrap_vars, 1 - alpha / 2)
        )

    def _iqae_bisection(self, estimator, grid, pmf, target_prob, max_iter=20):
        """Binary search for VaR using IQAE tail probability estimates."""
        low, high = 0, len(grid) - 1
        total_queries = 0

        for _ in range(max_iter):
            mid = (low + high) // 2

            # Create query object
            class Query:
                def __init__(self, idx):
                    self.index = idx

            result = estimator.estimate_tail_prob(
                Query(mid),
                epsilon=self.spec.iqae_epsilon,
                alpha=self.spec.iqae_alpha_fail
            )

            total_queries += result.cost

            if result.p_hat < target_prob:
                high = mid - 1
            else:
                low = mid + 1

            if low >= high:
                break

        return grid[low], total_queries


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_var_result(result: VaRResult, filepath: Path):
    """Save VaR result to JSON."""
    data = {
        "var_value": float(result.var_value),
        "cvar_value": float(result.cvar_value),
        "method": result.method,
        "confidence_level": result.confidence_level,
        "confidence_interval": result.confidence_interval,
        "metadata": result.metadata,
        "success": result.success,
        "message": result.message
    }
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[VaR Calculator] Result saved to {filepath}")


def load_var_result(filepath: Path) -> VaRResult:
    """Load VaR result from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return VaRResult(
        var_value=data["var_value"],
        cvar_value=data["cvar_value"],
        method=data["method"],
        confidence_level=data["confidence_level"],
        confidence_interval=tuple(data["confidence_interval"]) if data.get("confidence_interval") else None,
        metadata=data["metadata"],
        success=data["success"],
        message=data["message"]
    )


def compare_var_methods(
    portfolio_weights: np.ndarray,
    returns_data: np.ndarray,
    confidence_level: float = 0.95,
    n_samples: int = 10000
) -> Dict[str, VaRResult]:
    """Compare Classical MC vs IQAE VaR estimation."""
    results = {}

    for method in [VaRMethod.CLASSICAL_MC, VaRMethod.IQAE]:
        spec = VaRSpec(
            portfolio_weights=portfolio_weights,
            returns_data=returns_data,
            method=method,
            confidence_level=confidence_level,
            n_samples=n_samples,
            distribution="empirical"
        )
        calculator = VaRCalculator(spec)
        results[method.value] = calculator.calculate()

    # Print comparison
    print("\n" + "=" * 70)
    print("VaR METHOD COMPARISON")
    print("=" * 70)
    print(f"{'Method':<20} {'VaR':<12} {'CVaR':<12} {'Time (s)':<10}")
    print("-" * 70)

    for method_name, result in results.items():
        if result.success:
            elapsed = result.metadata.get('elapsed_time', 0)
            print(f"{method_name:<20} {result.var_value:<12.4f} {result.cvar_value:<12.4f} {elapsed:<10.3f}")

    print("=" * 70 + "\n")
    return results
