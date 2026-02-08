# Portfolio Optimization + VaR Backend

Comprehensive backend for portfolio optimization with **CVaR** (tail risk-focused) and **Variance** (volatility-focused) modes, plus **Value-at-Risk** estimation via Classical Monte Carlo and Quantum IQAE methods.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PortfolioPipeline                        │
│           End-to-End Orchestration + Metadata              │
└─────────┬───────────────────────────────┬──────────────────┘
          │                               │
          v                               v
┌─────────────────────┐         ┌──────────────────────┐
│ PortfolioOptimizer  │         │   VaRCalculator      │
│  - CVaR mode        │         │  - Classical MC      │
│  - Variance mode    │         │  - IQAE (quantum)    │
└─────────┬───────────┘         └──────────┬───────────┘
          │                                │
          v                                v
┌─────────────────────┐         ┌──────────────────────┐
│  OpenPhiSolve QIHD  │         │  Scenario Generator  │
│  CVXPY (LP/QP)      │         │  Distribution Fit    │
└─────────────────────┘         └──────────────────────┘
```

---

## Components

### 1. `portfolio_optimizer.py`

**Portfolio optimization with two modes:**

#### **CVaR Mode** (Conditional Value-at-Risk)
Minimizes tail losses (worst-case scenarios).

**Formulation:**
```
minimize: α + (1/(1-β)) * (1/S) * Σ u_s

subject to:
    u_s ≥ -Σ w_i * r_i^(s) - α    ∀s (excess loss beyond VaR)
    u_s ≥ 0                        ∀s
    Σ w_i = 1                         (budget)
    w_i ≥ 0                           (long-only)
    w_i ≤ max_weight                  (position limits)
    Σ z_i ≤ K                         (cardinality)

Variables:
    w_i  : weight for asset i
    α    : VaR threshold
    u_s  : excess loss in scenario s
    z_i  : binary indicator (if cardinality constraint)
```

**Why CVaR?**
- Captures **tail risk** (fat tails, crashes)
- Works with **Student-t** and heavy-tailed distributions
- Industry standard (Basel III)
- Doesn't penalize gains (only focuses on losses)

#### **Variance Mode** (Mean-Variance)
Minimizes total volatility (Markowitz optimization).

**Formulation:**
```
minimize: λ * w^T Σ w - μ^T w

subject to: (same constraints as CVaR)

Variables:
    w : portfolio weights
    Σ : covariance matrix
    μ : expected returns
    λ : risk aversion parameter
```

**Solvers:**
- **QIHD** (Quantum-Inspired Hamiltonian Dynamics) - via `portfolio_qihd/`
- **CVXPY** (Convex optimization) - standard LP/QP solver

---

### 2. `var_calculator.py`

**VaR estimation for optimized portfolio:**

#### **Classical Monte Carlo**
```python
Process:
1. Generate N scenarios from distribution (empirical/normal/Student-t)
2. Compute portfolio returns: r_p = Σ w_i * r_i
3. Compute losses: L = -r_p
4. VaR_α = quantile(L, α)
```

**Features:**
- Bootstrap confidence intervals
- CVaR (Expected Shortfall) calculation
- Distribution fitting (empirical, normal, Student-t)
- Convergence error estimates

#### **IQAE (Iterative Quantum Amplitude Estimation)**
```python
Process:
1. Discretize portfolio return distribution (2^n qubits)
2. Encode in quantum state: |ψ⟩ = Σ √p_i |i⟩
3. Define payoff oracle: marks |i⟩ where loss > threshold
4. IQAE estimates P(loss > threshold)
5. Binary search to find VaR_α
```

**Note:** Current implementation uses **classical fallback** with simulated IQAE metadata. For true quantum execution, integrate:
- `hyperparameter_sweep/quantum_iqae.py` (Classiq)
- `hyperparameter_sweep/qae_sweep.py` (Qiskit + Aer GPU)

---

### 3. `portfolio_pipeline.py`

**End-to-end orchestration:**

```python
Pipeline Flow:
1. Load/Generate return data
2. Optimize portfolio (CVaR or Variance)
3. Calculate VaR (Classical MC and/or IQAE)
4. Generate metadata for UI
5. Save results (JSON + summary report)
```

**Metadata Structure:**
```json
{
  "timestamp": "2026-02-05T...",
  "data": {
    "n_samples": 1000,
    "n_assets": 10
  },
  "optimization": {
    "mode": "cvar",
    "objective_value": 0.0234,
    "active_assets": 6
  },
  "portfolio": {
    "weights": [...],
    "active_positions": [...],
    "expected_return": 0.0012,
    "volatility": 0.0156,
    "sharpe_ratio": 0.76
  },
  "var": {
    "classical_mc": {
      "value": 0.0234,
      "ci_lower": 0.0221,
      "ci_upper": 0.0247,
      "cvar": 0.0312
    },
    "iqae": {
      "value": 0.0238,
      ...
    }
  }
}
```

**Output Files:**
- `metadata.json` - **Key file for UI consumption**
- `portfolio_result.json` - Full optimization result
- `var_result_*.json` - VaR results per method
- `pipeline_result.json` - Complete pipeline output
- `summary_report.txt` - Human-readable report

---

## Usage

### Quick Start

```python
from portfolio_backend import (
    PortfolioPipeline,
    PipelineConfig,
    OptimizationMode,
    VaRMethod
)
import numpy as np

# Load your return data
returns = np.load("historical_returns.npy")  # (n_samples, n_assets)
asset_names = ["AAPL", "MSFT", "GOOGL", ...]

# Configure pipeline
config = PipelineConfig(
    optimization_mode=OptimizationMode.CVAR,  # or OptimizationMode.VARIANCE
    var_methods=[VaRMethod.CLASSICAL_MC, VaRMethod.IQAE],
    cardinality=10,              # Max 10 assets
    max_weight=0.25,             # Max 25% per asset
    confidence_level_var=0.95,   # 95% VaR
    n_mc_samples=10000,
    distribution="student_t",    # Use Student-t for fat tails
    df_student_t=5.0,
    solver="cvxpy"               # or "qihd"
)

# Run pipeline
pipeline = PortfolioPipeline(config)
result = pipeline.run(returns, asset_names)

# Access results
print(f"Success: {result.success}")
print(f"Optimal weights: {result.portfolio_result.weights}")
print(f"VaR (Classical MC): {result.var_results['classical_mc'].var_value:.4f}")
print(f"CVaR: {result.var_results['classical_mc'].metadata['cvar']:.4f}")

# Metadata saved to: results/pipeline/{run_name}/metadata.json
```

### Individual Components

#### Portfolio Optimization Only

```python
from portfolio_backend import (
    PortfolioOptimizer,
    PortfolioSpec,
    OptimizationMode
)

spec = PortfolioSpec(
    returns=returns,
    mode=OptimizationMode.CVAR,
    cardinality=8,
    max_weight=0.3,
    confidence_level=0.95,
    solver="cvxpy"
)

optimizer = PortfolioOptimizer(spec)
result = optimizer.optimize()

print(f"Objective: {result.objective_value}")
print(f"Weights: {result.weights}")
print(f"Active assets: {result.metadata['active_assets']}")
```

#### VaR Calculation Only

```python
from portfolio_backend import (
    VaRCalculator,
    VaRSpec,
    VaRMethod
)

# Assume you have portfolio weights
weights = np.array([0.3, 0.2, 0.2, 0.15, 0.1, 0.05])

spec = VaRSpec(
    portfolio_weights=weights,
    returns_data=returns,
    method=VaRMethod.CLASSICAL_MC,
    confidence_level=0.95,
    n_samples=10000,
    distribution="empirical"
)

calculator = VaRCalculator(spec)
result = calculator.calculate()

print(f"VaR_95%: {result.var_value:.4f}")
print(f"CVaR: {result.metadata['cvar']:.4f}")
print(f"CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
```

---

## CVaR vs Variance: Comparison Table

| Aspect | Variance | CVaR |
|--------|----------|------|
| **What it measures** | Total spread (up & down) | Tail losses only |
| **Penalizes gains?** | Yes ✗ | No ✓ |
| **Captures crashes?** | Poorly | Very well ✓ |
| **Works with fat tails?** | No (assumes normal) | Yes ✓ |
| **Math complexity** | QP (easy) | LP (easy!) |
| **Variables needed** | n | n + S + 1 |
| **Computational cost** | Low | Medium (S scenarios) |
| **Regulatory use** | Academic | Basel III ✓ |
| **Real-world preference** | Rarely used | Industry standard ✓ |

**Recommendation:** Use **CVaR with Student-t** distribution for realistic crash modeling.

---

## Student-t Distribution Workflow

```
Historical data
    ↓
Fit Student-t distribution (df, μ, Σ)
    ↓
Generate S scenarios from Student-t
    ↓
[OpenPhiSolve/CVXPY] Minimize CVaR → optimal weights w*
    ↓
[Classical MC / IQAE] Estimate VaR_95%(w*)
    ↓
Report: "VaR = X% (95% confident loss < X%)"
```

---

## Testing & Verification

Run the verification suite:

```bash
cd portfolio_backend
python verify_portfolio_backend.py
```

**Tests:**
1. Variance optimization (constraints, feasibility)
2. CVaR optimization (tail scenarios, excess loss)
3. VaR calculation (Classical MC, IQAE, confidence intervals)
4. End-to-end pipeline (both modes)
5. Result persistence (save/load, metadata integrity)

---

## Integration with UI

**Key file for real-time UI:** `results/pipeline/{run_name}/metadata.json`

This file contains:
- Portfolio weights and active positions
- Expected return, volatility, Sharpe ratio
- VaR/CVaR estimates with confidence intervals
- Optimization metadata (mode, cardinality, etc.)

**Example UI consumption:**

```python
import json
from pathlib import Path

# Load latest run
metadata_file = Path("results/pipeline/latest_run/metadata.json")
with open(metadata_file) as f:
    metadata = json.load(f)

# Display in UI
ui.display_weights(metadata['portfolio']['active_positions'])
ui.display_metrics({
    'return': metadata['portfolio']['expected_return'],
    'volatility': metadata['portfolio']['volatility'],
    'sharpe': metadata['portfolio']['sharpe_ratio']
})
ui.display_var({
    'classical': metadata['var']['classical_mc']['value'],
    'quantum': metadata['var']['iqae']['value']
})
```

---

## Dependencies

**Required:**
- `numpy`
- `scipy`
- `pandas`
- `cvxpy` (for CVaR/Variance optimization)
- `portfolio_qihd` (for QIHD solver)

**Optional (for true quantum IQAE):**
- `classiq` (circuit synthesis)
- `qiskit` + `qiskit-aer` (quantum simulation)
- `cuquantum` (GPU-accelerated simulation)

---

## Future Enhancements

1. **True IQAE Integration:**
   - Connect to `hyperparameter_sweep/quantum_iqae.py`
   - Replace classical fallback with actual quantum circuit execution
   - GPU acceleration via Qiskit Aer + cuQuantum

2. **Advanced Distributions:**
   - GARCH models for time-varying volatility
   - Copula-based multi-asset dependencies
   - Mixture models for regime-switching

3. **Multi-Period Optimization:**
   - Dynamic rebalancing (T > 1 periods)
   - Transaction cost modeling
   - Turnover constraints

4. **Real-Time Streaming:**
   - Incremental updates as new data arrives
   - WebSocket integration for live UI updates
   - Background job scheduling

5. **Factor Models:**
   - Fama-French factor integration
   - Custom factor constraints (ESG, sector limits)
   - Factor risk decomposition

---

## Contact & Support

For issues or questions:
- Check `verify_portfolio_backend.py` for usage examples
- Review `portfolio_app/app.py` for Streamlit integration
- See `hyperparameter_sweep/` for quantum IQAE details

**License:** MIT
