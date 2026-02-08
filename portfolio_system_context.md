# Portfolio Optimization & VaR System - Complete Context

## System Overview

This project implements an end-to-end quantum-classical hybrid pipeline for portfolio optimization and Value-at-Risk (VaR) estimation, combining:

1. **Portfolio Optimization** via QIHD (Quantum-Inspired Hamiltonian Dynamics) using OpenPhiSolve
2. **VaR Calculation** via multiple methods: Classical, GPU Monte Carlo, and Quantum IQAE
3. **Interactive Web App** for parameter tuning and visualization

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PORTFOLIO PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

Step 1: Stock Universe & Returns
  ├─ Real/Synthetic data (20-stock universe, sector-based factor model)
  ├─ Student-t distribution for fat tails
  └─ Output: returns (n_days × M), μ, Σ

Step 2: Portfolio Optimization (portfolio_qihd/)
  ├─ MIQP Formulation (miqp_builder.py)
  ├─ QIHD Solver (solver.py via OpenPhiSolve)
  └─ Output: weights w, selection y

Step 3: VaR Calculation (multiple methods)
  ├─ Classical Historical VaR
  ├─ GPU Monte Carlo VaR (pipeline.py)
  └─ Quantum IQAE VaR (VaR_Quantum.py)

Step 4: Visualization & Reporting
  ├─ Per-asset distributions
  ├─ Portfolio loss distribution
  ├─ Risk contributions
  └─ CSV/PNG outputs
```

---

## Part 1: Portfolio Optimization (`portfolio_qihd/`)

### Mathematical Formulation (MIQP)

The portfolio optimization problem is formulated as a **Mixed-Integer Quadratic Program (MIQP)**:

```
Decision Variables:
  - y_i ∈ {0,1}  : binary selection variables (M assets)
  - w_i ∈ [0, u_i]: continuous weight variables (M assets)

Objective (minimize):
  λ · w^T Σ w  -  μ^T w
  \_________/     \_____/
   risk term    return term

Constraints:
  1. Budget:      Σ w_i = 1             (fully invested)
  2. Long-only:   w_i ≥ 0               (no short selling)
  3. Linking:     w_i ≤ u_i · y_i       (weight caps only if selected)
  4. Cardinality: Σ y_i ≤ K             (max K holdings)
```

**Key Parameters:**
- `λ` (lambda_risk): Risk aversion (higher = more conservative)
- `K` (max_positions): Maximum number of assets to hold
- `u_i` (upper caps): Per-asset maximum weight (e.g., 0.10 = 10%)

### File Structure

```
portfolio_qihd/
├── __init__.py           # Module exports
├── miqp_builder.py       # MIQP matrix construction
├── solver.py             # QIHD solver wrapper
├── data.py               # Data utilities (μ, Σ computation)
├── example.py            # CLI demo
└── outputs/              # Saved weights
```

### Core Components

#### 1. `miqp_builder.py` - MIQP Formulation

**Backend Formulation Location:** This file contains the complete mathematical model encoding.

**Key Classes/Functions:**
```python
@dataclass
class PortfolioSpec:
    mu: np.ndarray         # Expected returns (M,)
    cov: np.ndarray        # Covariance matrix (M, M)
    upper: np.ndarray      # Per-asset caps (M,)
    max_positions: int     # Cardinality K
    lambda_risk: float     # Risk aversion λ
    budget: float = 1.0    # Total capital

def build_portfolio_miqp(spec: PortfolioSpec) -> MIQP:
    """Encode portfolio problem as MIQP for OpenPhiSolve."""
    # Returns: MIQP(Q, w, A, b, C, d, bounds, n_binary_vars)
```

**Matrix Encoding:**
- **Decision vector:** `x = [y_0..y_{M-1}, w_0..w_{M-1}]` (2M variables, first M are binary)
- **Q matrix (quadratic):** Risk term `2λΣ` applied only to weights block
- **w vector (linear):** `-μ` for weights, 0 for binaries
- **A, b (inequalities):** Linking constraints + cardinality constraint
- **C, d (equalities):** Budget constraint `Σw_i = 1`
- **Bounds:** `y_i ∈ {0,1}`, `w_i ∈ [0, u_i]`

#### 2. `solver.py` - QIHD Solver

**How QIHD Works:**
QIHD (Quantum-Inspired Hamiltonian Dynamics) solves optimization problems by simulating classical Hamiltonian dynamics in continuous space, then rounding to discrete solutions.

```python
@dataclass
class PortfolioResult:
    weights: np.ndarray      # Optimal weights w
    selection: np.ndarray    # Binary selections y
    objective: float         # Objective value
    response: Any            # Full solver response

def solve_portfolio(spec: PortfolioSpec,
                   backend_kwargs: dict = None,
                   if_refine: bool = False) -> PortfolioResult:
    """
    Solve portfolio MIQP using QIHD.

    backend_kwargs:
        n_shots: Number of QIHD trajectories (default: 500)
        n_steps: Integration steps per trajectory (default: 5000)
        dt: Time step size (default: 0.2)
        device: 'gpu' or 'cpu'
    """
```

**QIHD Parameters:**
- `n_shots`: More shots → better exploration → slower
- `n_steps`: More steps → finer integration → slower
- `dt`: Smaller dt → more accurate → slower
- `device`: GPU accelerates via JAX

**Optional PDQP Refinement:** Post-processes QIHD solution with gradient-based continuous refinement (requires `mpax` package).

#### 3. `data.py` - Data Utilities

```python
def compute_mu_cov(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute sample mean and covariance from return matrix (n_samples, n_assets)."""

def synthetic_factor_model(n_samples, n_assets, n_factors, seed) -> np.ndarray:
    """Generate synthetic returns using factor model: r = F @ B^T + ε."""

def caps_from_constant(max_weight: float, n_assets: int) -> np.ndarray:
    """Create uniform per-asset caps."""
```

---

## Part 2: Interactive App (`portfolio_app/`)

### File: `app.py` - Streamlit Interface

**Purpose:** Provides a user-friendly UI for portfolio optimization + VaR estimation.

**Workflow:**
```
1. Load Data
   ├─ Upload CSV returns OR
   └─ Generate synthetic factor model

2. Configure Portfolio
   ├─ K (max holdings): 1-300
   ├─ u_i (per-asset cap): 0.01-0.50
   ├─ λ (risk aversion): 0.1-20.0
   └─ QIHD hyperparameters (shots, steps, dt, device)

3. Solve Portfolio
   └─ Calls portfolio_qihd.solve_portfolio()

4. Compute Classical VaR
   └─ Historical simulation on loaded returns

5. Display Results
   ├─ Selected assets + weights
   ├─ Objective value
   └─ VaR metric
```

**VaR Calculation in App:**

```python
def classical_var(pnl: np.ndarray, alpha: float) -> float:
    """Historical VaR (quantile of losses)."""
    losses = -pnl  # Convert P&L to losses
    return float(np.quantile(losses, alpha))

def compute_portfolio_var(returns: pd.DataFrame,
                         weights: np.ndarray,
                         alpha: float) -> float:
    pnl = returns.values @ weights  # Portfolio P&L
    return classical_var(pnl, alpha)
```

**Location:** `portfolio_app/app.py:50-62`

**How to Run:**
```bash
PYTHONPATH=OpenPhiSolve:. streamlit run portfolio_app/app.py
```

---

## Part 3: VaR Calculation Methods

### Overview

The system supports **three VaR estimation methods**:

| Method | File | Speed | Accuracy | Hardware |
|--------|------|-------|----------|----------|
| Classical Historical | `portfolio_app/app.py` | Fast | Baseline | CPU |
| GPU Monte Carlo | `pipeline.py` | Fast | High | GPU/CPU |
| Quantum IQAE | `VaR_Quantum.py` | Slow | High (with speedup) | Quantum/Simulator |

### Method 1: Classical Historical VaR

**File:** `portfolio_app/app.py`

**Method:**
1. Compute portfolio returns: `r_p = Σ w_i r_i`
2. Sort returns
3. Find α-quantile of losses: `VaR_α = -quantile(r_p, α)`

**Pros:** Simple, fast, no assumptions
**Cons:** Limited to historical scenarios, no extrapolation

---

### Method 2: GPU Monte Carlo VaR

**File:** `pipeline.py:256-351`

**Method:**
1. Sample from multivariate Student-t distribution: `r ~ t_df(0, Σ_selected)`
2. Compute portfolio losses: `L = -w^T r`
3. Estimate VaR as empirical α-quantile of L
4. Estimate CVaR as mean of tail losses

**GPU Acceleration:**
```python
import jax
import jax.numpy as jnp

# Sample Student-t via Gaussian + chi-square mixing
Z = jax.random.normal(key, (n_scenarios, M))
chi2 = 2.0 * jax.random.gamma(key, df/2, (n_scenarios, 1))
T = Z * jnp.sqrt(df / chi2)  # Student-t samples
R = T @ L.T  # Correlated returns (L = Cholesky of Σ)
```

**Output:**
```python
{
    "portfolio_var": float,      # VaR_α
    "portfolio_cvar": float,     # CVaR_α (Expected Shortfall)
    "per_asset_metrics": [       # Per-asset VaR/CVaR
        {"ticker": str, "var": float, "cvar": float, ...}
    ],
    "device": str,               # "gpu" or "cpu"
}
```

**Location for formulation:** `pipeline.py:256-351` (function `gpu_monte_carlo_var`)

---

### Method 3: Quantum IQAE VaR

**File:** `VaR_Quantum.py`

**Method Overview:**

Quantum VaR estimation uses **Iterative Quantum Amplitude Estimation (IQAE)** to estimate tail probabilities with quadratic speedup over classical Monte Carlo.

**Pipeline:**
```
1. Discretize Distribution
   ├─ Build PMF on 2^n grid points from continuous distribution
   └─ Options: uniform grid OR tail-focused grid

2. State Preparation (Quantum Circuit)
   ├─ Encode PMF as qubit amplitudes: |ψ⟩ = Σ√p_i |i⟩
   └─ Via Classiq's inplace_prepare_state()

3. Threshold Oracle
   ├─ Mark states where asset_index < threshold
   └─ Flip indicator qubit for tail events

4. IQAE Estimation
   ├─ Estimate P(indicator=1) = CDF(threshold)
   ├─ Complexity: O(1/ε) oracle queries (vs O(1/ε²) classical MC)
   └─ Returns: estimation, confidence_interval, grover_calls

5. Bisection Search
   ├─ Binary search over threshold indices
   ├─ Find threshold where CDF(threshold) ≈ α
   └─ Optional: GPU warm start to narrow bracket
```

**Key Functions:**

**Distribution Building:**
```python
def build_uniform_pmf_from_dist(dist, num_qubits, lo, hi):
    """Discretize scipy.stats distribution on uniform grid."""
    # VaR_Quantum.py:147-153

def build_tail_focused_pmf_from_dist(dist, num_qubits, tail_alpha, ...):
    """Non-uniform grid: allocate more bins to tail region."""
    # VaR_Quantum.py:66-137
```

**Quantum Circuit:**
```python
@qfunc
def state_preparation(asset: QArray[QBit], ind: QBit):
    """A operator: load distribution + apply threshold oracle."""
    load_distribution(asset=asset)
    payoff(asset=asset, ind=ind)

@qfunc
def load_distribution(asset: QNum):
    """Encode PMF into amplitudes."""
    inplace_prepare_state(probs, bound=0, target=asset)

@qperm
def payoff(asset: Const[QNum], ind: QBit):
    """Threshold oracle: flip ind when asset < THRESHOLD_INDEX."""
    ind ^= asset < THRESHOLD_INDEX
```

**IQAE Execution:**
```python
def estimate_tail_probability(threshold_index, epsilon, alpha_fail):
    """
    Use IQAE to estimate P(asset < threshold_index).

    Returns:
        estimation: float (point estimate)
        confidence_interval: (ci_low, ci_high)
        resources: {shots_total, grover_calls, ks_used}
    """
    # VaR_Quantum.py:357-383
```

**Bisection Search:**
```python
def quantum_value_at_risk(grid, pmf, alpha, tolerance, epsilon, bracket=None):
    """
    Compute VaR using IQAE + bisection.

    Args:
        bracket: Optional (lo, hi) from MC warm start

    Returns:
        var_index, var_value, final_estimate, confidence_interval,
        total_oracle_queries, bisection_steps
    """
    # VaR_Quantum.py:389-468
```

**GPU Warm Start (Hybrid Approach):**
```python
def mc_warm_start(grid, pmf, alpha, n_samples, method="gpu", ...):
    """
    Use cheap MC to get tight bracket [lo, hi] around VaR index.
    Reduces bisection from log2(N) steps to log2(hi-lo) steps.

    Methods:
        - "gpu": PyTorch multinomial sampling on CUDA
        - "qmc": Quasi-Monte Carlo with Sobol sequences
        - "plain": Standard numpy MC

    Returns: {lo, hi, var_est_idx, bracket_width, device, ...}
    """
    # VaR_Quantum.py:213-288
```

**Resource Complexity:**
- **Without warm start:** `B × C(ε) × S(n)` where:
  - `B = log₂(N)` bisection steps (N = 2^n grid points)
  - `C(ε) = O(1/ε)` oracle queries per IQAE call
  - `S(n) = O(2^n)` circuit size (state prep)

- **With warm start:** Replace `log₂(N)` with `log₂(bracket_width)`, typically 3-5 steps

**Location for formulation:** `VaR_Quantum.py:1-970` (entire file)

**Key Hyperparameters:**
- `NUM_QUBITS = 7` → 2^7 = 128 grid points
- `IQAE_EPSILON = 0.05` → target estimation precision
- `IQAE_ALPHA_FAIL = 0.01` → failure probability for CI
- `ALPHA = 0.05` → VaR confidence level (95%)

---

## Part 4: End-to-End Pipeline (`pipeline.py`)

### Complete Integration

**File:** `pipeline.py`

**Purpose:** Orchestrates the full workflow from portfolio optimization to VaR estimation, with publication-quality visualizations.

**Pipeline Steps:**

```
Step 1: Build Stock Universe
  ├─ 20 stocks across 7 sectors (Tech, Finance, Healthcare, Energy, ...)
  ├─ Sector-based factor model (3 factors: market, sector rotation, volatility)
  ├─ Student-t innovations for fat tails
  └─ Output: tickers, sectors, returns (252×20), μ, Σ

Step 2: Portfolio Optimization
  ├─ Try QIHD solver first (portfolio_qihd)
  ├─ Fallback to scipy SLSQP if QIHD unavailable
  └─ Output: weights w, selected indices

Step 3: GPU Monte Carlo VaR
  ├─ Sample N scenarios from t_df(0, Σ_selected)
  ├─ Compute portfolio losses: L = -w^T r
  ├─ Estimate VaR, CVaR, per-asset metrics
  └─ Device: GPU (JAX) or CPU (numpy)

Step 4: Visualization
  ├─ Per-asset distribution plots
  ├─ Portfolio loss distribution
  ├─ Asset allocation pie chart
  ├─ Risk contribution bar chart
  └─ Combined summary figure

Step 5: Structured Output
  ├─ Formatted console table
  └─ CSV report (results/pipeline/report.csv)

Step 6: Quantum IQAE (Optional)
  ├─ Discretize portfolio losses onto 2^n grid
  ├─ GPU warm start for tight bracket
  ├─ Run IQAE + bisection within bracket
  └─ Compare quantum vs classical VaR
```

**Key Integration Point:**

```python
# portfolio_qihd integration
from portfolio_qihd import PortfolioSpec, solve_portfolio, caps_from_constant

spec = PortfolioSpec(mu=mu, cov=cov, upper=caps,
                    max_positions=K, lambda_risk=λ)
result = solve_portfolio(spec, backend_kwargs={...})
weights = result.weights

# VaR pipeline
mc_results = gpu_monte_carlo_var(cov, weights, ...)
quantum_results = run_quantum_var(mc_results["portfolio_losses"], ...)
```

**Location:** `pipeline.py:1-866`

**Run Commands:**
```bash
# GPU MC only
python pipeline.py --no-quantum --n-scenarios 100000

# Full pipeline with quantum IQAE
python pipeline.py --quantum --epsilon 0.05

# Custom portfolio config
python pipeline.py --max-positions 5 --lambda-risk 10.0 --cap 0.20
```

---

## Part 5: How to Add New VaR Methods

### Example: Adding Parametric VaR

**Step 1: Implement method in new file or add to `pipeline.py`**

```python
def parametric_var(weights, cov, alpha=0.05, dist="normal"):
    """
    Parametric VaR assuming normal or t-distribution.

    Args:
        weights: Portfolio weights (M,)
        cov: Covariance matrix (M, M)
        alpha: Confidence level
        dist: "normal" or "t"

    Returns:
        var: VaR estimate
        cvar: CVaR estimate (if analytical formula available)
    """
    # Portfolio variance
    port_var = weights @ cov @ weights
    port_std = np.sqrt(port_var)

    if dist == "normal":
        # Normal VaR: σ_p × z_α
        z_alpha = scipy.stats.norm.ppf(alpha)
        var = -port_std * z_alpha  # Negative because loss
        # CVaR for normal: σ_p × φ(z_α) / α
        cvar = port_std * scipy.stats.norm.pdf(z_alpha) / alpha

    elif dist == "t":
        # Student-t VaR (more conservative for fat tails)
        df = 4  # degrees of freedom
        t_alpha = scipy.stats.t.ppf(alpha, df)
        scale = port_std * np.sqrt((df - 2) / df)  # t-distribution scaling
        var = -scale * t_alpha
        # CVaR approximation for t-distribution
        cvar = scale * scipy.stats.t.pdf(t_alpha, df) * (df + t_alpha**2) / (alpha * (df - 1))

    return float(var), float(cvar)
```

**Step 2: Integrate into pipeline**

Add to `pipeline.py` after GPU Monte Carlo:

```python
# In main() function, after Step 3
print()
print("=" * 65)
print("STEP 3B: Parametric VaR (Normal)")
print("=" * 65)
param_var_norm, param_cvar_norm = parametric_var(
    weights, cov, alpha=args.alpha, dist="normal"
)
print(f"  Parametric VaR (Normal):  {param_var_norm:.6f}")
print(f"  Parametric CVaR (Normal): {param_cvar_norm:.6f}")

print()
print("STEP 3C: Parametric VaR (Student-t)")
print("=" * 65)
param_var_t, param_cvar_t = parametric_var(
    weights, cov, alpha=args.alpha, dist="t"
)
print(f"  Parametric VaR (t-dist):  {param_var_t:.6f}")
print(f"  Parametric CVaR (t-dist): {param_cvar_t:.6f}")
```

**Step 3: Add to Streamlit app**

In `portfolio_app/app.py`, after classical VaR:

```python
st.write("Computing parametric VaR …")

# Convert daily cov to return-scale cov
cov_selected = cov[np.ix_(selected, selected)]
w_selected = weights[selected]

var_norm, cvar_norm = parametric_var(w_selected, cov_selected, alpha, "normal")
var_t, cvar_t = parametric_var(w_selected, cov_selected, alpha, "t")

col1, col2 = st.columns(2)
with col1:
    st.metric(f"{int(alpha*100)}% VaR (Normal)", f"{var_norm:.4f}")
    st.metric(f"{int(alpha*100)}% CVaR (Normal)", f"{cvar_norm:.4f}")
with col2:
    st.metric(f"{int(alpha*100)}% VaR (Student-t)", f"{var_t:.4f}")
    st.metric(f"{int(alpha*100)}% CVaR (Student-t)", f"{cvar_t:.4f}")
```

**Step 4: Add visualization comparison**

Create comparison plot showing all methods:

```python
def plot_var_comparison(mc_var, param_var_norm, param_var_t, quantum_var=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ["Historical", "MC", "Parametric\n(Normal)", "Parametric\n(t-dist)"]
    values = [historical_var, mc_var, param_var_norm, param_var_t]
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_DANGER]

    if quantum_var is not None:
        methods.append("Quantum\nIQAE")
        values.append(quantum_var)
        colors.append(COLOR_BOUND_UPPER)

    bars = ax.bar(methods, values, color=colors, edgecolor="white", linewidth=2)
    ax.set_ylabel("VaR", fontweight="500")
    ax.set_title("VaR Method Comparison", fontweight="600")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontweight="500")

    fig.tight_layout()
    return fig
```

---

## Part 6: Key File Locations Reference

### Portfolio Optimization Backend

| Component | File | Function/Class |
|-----------|------|----------------|
| MIQP Formulation | `portfolio_qihd/miqp_builder.py` | `build_portfolio_miqp()` |
| Objective Function | `portfolio_qihd/miqp_builder.py:56-63` | Q matrix (risk) + w vector (return) |
| Constraints | `portfolio_qihd/miqp_builder.py:66-92` | A, b (inequalities) + C, d (equalities) |
| QIHD Solver | `portfolio_qihd/solver.py` | `solve_portfolio()` |
| Data Utilities | `portfolio_qihd/data.py` | `compute_mu_cov()`, `synthetic_factor_model()` |

### VaR Calculation

| Method | File | Function | Line Range |
|--------|------|----------|-----------|
| Classical Historical | `portfolio_app/app.py` | `classical_var()` | 50-53 |
| GPU Monte Carlo | `pipeline.py` | `gpu_monte_carlo_var()` | 256-351 |
| Quantum IQAE | `VaR_Quantum.py` | `quantum_value_at_risk()` | 389-468 |
| IQAE Core | `VaR_Quantum.py` | `estimate_tail_probability()` | 357-383 |
| MC Warm Start | `VaR_Quantum.py` | `mc_warm_start()` | 213-288 |
| State Prep | `VaR_Quantum.py` | `state_preparation()` | 294-298 |
| Threshold Oracle | `VaR_Quantum.py` | `payoff()` | 307-310 |

### Integration & Pipeline

| Component | File | Function | Line Range |
|-----------|------|----------|-----------|
| Full Pipeline | `pipeline.py` | `main()` | 745-866 |
| Stock Universe | `pipeline.py` | `build_stock_universe()` | 99-172 |
| Optimization Step | `pipeline.py` | `optimize_portfolio()` | 178-250 |
| Quantum Bridge | `pipeline.py` | `run_quantum_var()` | 623-688 |
| Streamlit App | `portfolio_app/app.py` | (entire file) | 1-142 |

---

## Part 7: How QIHD Works (Conceptual)

**QIHD (Quantum-Inspired Hamiltonian Dynamics)** is a classical algorithm inspired by quantum annealing that solves optimization problems by:

1. **Hamiltonian Formulation:**
   - Map MIQP to Hamiltonian: `H(x) = objective(x) + penalties(constraints)`
   - Binary variables → Ising spins: `y_i ∈ {0,1}` → `s_i ∈ {-1,+1}`

2. **Continuous Relaxation:**
   - Embed in continuous space with auxiliary momentum variables
   - System: `(x, p)` where `x` are positions, `p` are momenta

3. **Hamiltonian Dynamics:**
   - Evolve system via Hamilton's equations:
     ```
     dx/dt = ∂H/∂p
     dp/dt = -∂H/∂x
     ```
   - Numerical integration via symplectic methods (preserves energy)

4. **Parallel Trajectories:**
   - Run `n_shots` independent trajectories from random initial states
   - Each trajectory explores solution space via Hamiltonian flow

5. **Rounding & Selection:**
   - At the end, round continuous `x` to binary `y`
   - Select trajectory with lowest objective value

**Why it works:**
- Hamiltonian dynamics naturally explore energy landscape
- Momentum helps escape local minima
- Parallel trajectories provide diversity
- GPU acceleration makes it competitive with traditional solvers

**Comparison to alternatives:**
- **Branch-and-bound (Gurobi, CPLEX):** Exact but slow for large problems
- **Genetic algorithms:** Heuristic, no convergence guarantees
- **Simulated annealing:** Stochastic, requires careful cooling schedule
- **QIHD:** Deterministic trajectories, fast on GPU, good empirical performance

---

## Part 8: Usage Examples

### Example 1: Portfolio Optimization Only

```bash
# Optimize 50-asset portfolio, select top 10, save weights
python -m portfolio_qihd.example \
  --assets 50 \
  --k 10 \
  --lambda-risk 5.0 \
  --cap 0.15 \
  --n-shots 500 \
  --device gpu

# Output saved to: portfolio_qihd/outputs/weights.npy
```

### Example 2: Portfolio + Classical VaR (Streamlit)

```bash
PYTHONPATH=OpenPhiSolve:. streamlit run portfolio_app/app.py

# In browser:
# 1. Set K=8, cap=0.10, λ=5.0
# 2. Click "Solve + Compute VaR"
# 3. View weights + VaR metric
```

### Example 3: Full Pipeline (GPU MC + Quantum)

```bash
# GPU Monte Carlo + Quantum IQAE
python pipeline.py \
  --max-positions 8 \
  --lambda-risk 5.0 \
  --n-scenarios 100000 \
  --quantum \
  --epsilon 0.05 \
  --output-dir results/full_run

# Outputs:
# - results/full_run/summary.png
# - results/full_run/report.csv
# - results/full_run/portfolio_distribution.png
# - results/full_run/asset_*.png (per-asset)
```

### Example 4: Quantum VaR Epsilon Sweep

```bash
# Measure IQAE scaling: query complexity vs precision
python VaR_Quantum.py \
  --sweep \
  --sweep-min 0.005 \
  --sweep-max 0.30 \
  --sweep-points 20 \
  --sweep-runs 3 \
  --sweep-gpu-warmstart

# Output: results/iqae_epsilon_sweep.csv
# Columns: epsilon, a_hat_mean, abs_error_mean, grover_calls_mean, ...
```

### Example 5: Quantum VaR with GPU Warm Start

```bash
# Hybrid: GPU MC warm start → tight bracket → IQAE
python VaR_Quantum.py \
  --mc-method gpu \
  --mc-samples 50000 \
  --mc-confidence 0.99 \
  --epsilon 0.05

# Benefits:
# - MC samples: 50,000 (cheap on GPU)
# - Bracket width: ~5-10 (vs 128 full search)
# - IQAE steps: 3-4 (vs 7 without warm start)
# - Total cost: MC + 3-4 × IQAE(ε)
```

---

## Part 9: Performance & Scalability

### Portfolio Optimization

| Assets (M) | Binary Vars | Continuous Vars | QIHD Time (GPU) | Memory |
|------------|-------------|-----------------|-----------------|--------|
| 20         | 20          | 20              | ~5s             | <1 GB  |
| 50         | 50          | 50              | ~10s            | ~2 GB  |
| 100        | 100         | 100             | ~20s            | ~4 GB  |
| 300        | 300         | 300             | ~60s            | ~10 GB |

*Typical: n_shots=500, n_steps=5000, dt=0.2*

### VaR Estimation

| Method | Scenarios | Time (GPU) | Time (CPU) | Accuracy |
|--------|-----------|------------|------------|----------|
| Classical Historical | 252 days | <1ms | <1ms | Baseline |
| GPU Monte Carlo | 100,000 | ~50ms | ~5s | High |
| GPU Monte Carlo | 1,000,000 | ~200ms | ~50s | Very High |
| Quantum IQAE (no warmstart) | - | ~60s | N/A | High |
| Quantum IQAE (GPU warmstart) | 50k MC + IQAE | ~15s | N/A | High |

*IQAE time includes circuit synthesis + execution on simulator*

### Quantum IQAE Scaling

**Without warm start:**
- Bisection steps: `B = log₂(N)` = 7 for N=128
- IQAE calls per step: 1
- Oracle queries per IQAE: `O(1/ε)` ≈ 20-100 for ε=0.05
- **Total: ~7 × 50 = 350 oracle queries**

**With GPU warm start (bracket width ≈ 5):**
- Bisection steps: `B = log₂(5)` ≈ 3
- MC samples: 50,000 (GPU: ~10ms)
- **Total: 50k MC + 3 × 50 = ~50,150 total samples**
- **Speedup: 7× fewer IQAE calls**

---

## Part 10: Common Issues & Troubleshooting

### Issue 1: QIHD solver fails

**Symptom:** `ImportError: No module named 'phisolve'`

**Fix:**
```bash
cd OpenPhiSolve
pip install -e .
```

### Issue 2: GPU not detected

**Symptom:** `device: cpu` instead of `gpu` or `cuda`

**Fix:**
```bash
# Check JAX/PyTorch GPU support
python -c "import jax; print(jax.default_backend())"
python -c "import torch; print(torch.cuda.is_available())"

# Install GPU versions
pip install --upgrade "jax[cuda12]"  # or cuda11
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: Classiq authentication fails

**Symptom:** `classiq.authenticate() fails` or `Preferences not found`

**Fix:**
```bash
# Login via CLI (one-time)
classiq auth login

# Or set token in env
export CLASSIQ_IDE_TOKEN="your_token_here"
```

### Issue 4: Quantum circuit synthesis hangs

**Symptom:** `synthesize()` takes >5 minutes

**Possible causes:**
- Large `num_qubits` (>10) → exponential circuit size
- Complex distribution PMF → deeper state prep circuit
- Network latency (Classiq API calls)

**Fix:**
- Reduce `num_qubits` to 7-8
- Use simpler distributions (uniform grid)
- Check Classiq API status

### Issue 5: IQAE estimation inaccurate

**Symptom:** `|a_hat - a_true| > epsilon`

**Causes:**
- `epsilon` too large (increase precision)
- Insufficient IQAE iterations (decrease `alpha_fail`)
- Poor bracket from warm start

**Fix:**
```bash
# Tighten IQAE precision
python VaR_Quantum.py --epsilon 0.01 --alpha-fail 0.001

# Increase MC warm-start samples
python VaR_Quantum.py --mc-samples 100000
```

---

## Part 11: Future Extensions

### 1. Real Market Data Integration
- Replace synthetic factor model with actual stock returns (Yahoo Finance, Alpha Vantage)
- Handle missing data, outliers, corporate actions

### 2. Transaction Costs & Rebalancing
- Add turnover penalties to MIQP: `+ λ_TC · Σ|w_i - w_i^old|`
- Multi-period optimization with rebalancing costs

### 3. Additional VaR Methods
- **Cornish-Fisher VaR:** Adjust for skewness & kurtosis
- **GARCH VaR:** Time-varying volatility models
- **Copula-based VaR:** Model tail dependence explicitly

### 4. Risk Parity & Factor Models
- Equal risk contribution: `w_i · (Σw)_i / σ_p = const`
- Factor risk models: decompose risk by systematic factors

### 5. Quantum Hardware Execution
- Port IQAE to real quantum hardware (IBM, AWS Braket)
- Noise mitigation & error correction
- Benchmark speedup on actual QPUs

### 6. Multi-Asset Class Portfolios
- Extend to stocks + bonds + commodities + crypto
- Cross-asset correlations & regime switching

### 7. Backtesting Framework
- Out-of-sample performance evaluation
- Rolling window optimization
- VaR backtesting (Kupiec test, Christoffersen test)

---

## Glossary

- **MIQP:** Mixed-Integer Quadratic Program (binary + continuous variables, quadratic objective)
- **QIHD:** Quantum-Inspired Hamiltonian Dynamics (optimization via classical Hamiltonian flow)
- **VaR:** Value at Risk (α-quantile of loss distribution)
- **CVaR:** Conditional VaR / Expected Shortfall (mean of losses beyond VaR)
- **IQAE:** Iterative Quantum Amplitude Estimation (quantum algorithm, O(1/ε) complexity)
- **QAE:** Quantum Amplitude Estimation (QPE-based, fixed precision)
- **PMF:** Probability Mass Function (discrete distribution)
- **CDF:** Cumulative Distribution Function
- **Grover oracle:** Quantum subroutine marking "good" states (used in amplitude estimation)
- **Bisection:** Binary search algorithm (halves search space each iteration)
- **Symplectic integrator:** Numerical method preserving Hamiltonian structure
- **Cholesky decomposition:** L such that Σ = LL^T (used to generate correlated samples)

---

## References & Dependencies

### Core Libraries
- **OpenPhiSolve:** QIHD/PDQP solvers (local package in `OpenPhiSolve/`)
- **Classiq:** Quantum circuit synthesis & IQAE (`pip install classiq`)
- **JAX:** GPU-accelerated numerical computing (`pip install jax[cuda]`)
- **NumPy/SciPy:** Scientific computing
- **Streamlit:** Web UI framework
- **Matplotlib:** Plotting

### File Dependencies
```
portfolio_app/app.py
  └─ imports portfolio_qihd

portfolio_qihd/
  └─ imports phisolve (from OpenPhiSolve/)

pipeline.py
  ├─ imports portfolio_qihd
  └─ imports VaR_Quantum_highD (optional quantum bridge)

VaR_Quantum.py
  └─ imports classiq
```

---

## Contact & Contribution

**Project maintainer:** [Your team/contact info]

**How to contribute:**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-var-method`
3. Implement changes (follow existing code style)
4. Add tests if applicable
5. Submit pull request with clear description

**Key areas for contribution:**
- New VaR estimation methods
- Real market data connectors
- Performance optimizations
- Documentation improvements
- Quantum circuit optimizations
- Visualization enhancements

---

## Summary

This system provides a complete end-to-end pipeline for:

1. **Portfolio Optimization** via QIHD (quantum-inspired solver)
2. **VaR Estimation** via multiple methods (classical, GPU MC, quantum IQAE)
3. **Interactive Exploration** via Streamlit web app
4. **Publication-Quality Outputs** (plots, CSV reports)

**Key innovation:** Hybrid quantum-classical approach using GPU Monte Carlo warm start to accelerate quantum IQAE VaR estimation.

**Production-ready components:**
- Portfolio optimization (QIHD solver)
- GPU Monte Carlo VaR
- Streamlit app

**Research components:**
- Quantum IQAE VaR (requires Classiq SDK + simulator/hardware access)

Start with the Streamlit app for interactive exploration, then scale to the full pipeline for production workflows.
