# Portfolio Optimizer — Streamlit App

Interactive web application for cardinality-constrained portfolio optimization using QIHD (Quantum-Inspired Hamiltonian Dynamics) with Value at Risk analysis.

## Features

- **Portfolio Optimization**: Mean-variance optimization with cardinality constraints via OpenPhiSolve QIHD solver
- **GPU Acceleration**: Automatic GPU detection for WSL2/CUDA (JAX backend)
- **VaR Analysis**: Historical and Monte Carlo Value at Risk (VaR) and Conditional VaR (CVaR)
- **Data Sources**: CSV upload or synthetic factor model
- **Visualizations**:
  - Portfolio allocation bar chart
  - Loss distribution histogram
  - Marginal risk contributions
- **Anthropic-Inspired Design**: Clean, hierarchical UI with warm color palette and gradients

## Installation

```bash
# Install OpenPhiSolve (required)
cd /mnt/c/Users/azain/WhatTheDuck
pip install -e OpenPhiSolve

# Install Streamlit (if not already installed)
pip install streamlit pandas numpy matplotlib
```

## Run the App

```bash
# From the project root
PYTHONPATH=OpenPhiSolve:. streamlit run portfolio_app/app.py
```

The app will open in your default browser at `http://localhost:8501`.

## Usage

### 1. Configure Data
- **Upload CSV**: Provide a returns matrix (rows = time periods, columns = assets)
- **Or use Synthetic**: Generate factor-model returns with configurable seed

### 2. Set Portfolio Constraints
- **Max holdings (K)**: Cardinality constraint (e.g., K=15 means hold at most 15 assets)
- **Per-asset cap**: Upper bound on individual weights (e.g., 0.10 = 10% max per asset)
- **Risk aversion (λ)**: Higher values → more conservative (penalizes variance)

### 3. Solver Settings
- **Device**: `gpu` (if CUDA available) or `cpu`
- **Shots**: Number of parallel QIHD trajectories (default: 500)
- **Steps**: Integration steps per trajectory (default: 5000)
- **dt**: Time step size (default: 0.2)

### 4. VaR Parameters
- **Confidence level**: 90%, 95%, 97.5%, or 99% (default: 95%)
- **MC scenarios**: Number of Monte Carlo simulations for VaR (default: 50,000)

### 5. Run Optimization
Click **🚀 Run Optimization** to:
1. Solve the MIQP using QIHD
2. Display optimal weights and allocation
3. Compute Historical and Monte Carlo VaR/CVaR
4. Generate visualizations

## CSV Format

Your CSV should have:
- **Rows**: Time periods (e.g., daily returns)
- **Columns**: Assets (numeric values only)
- **Values**: Returns (e.g., 0.01 = 1% return)

Example:
```csv
AAPL,MSFT,GOOGL,AMZN,TSLA
0.012,0.008,0.015,-0.003,0.025
-0.005,0.002,0.001,0.010,-0.012
...
```

## GPU Support (WSL2)

The app auto-detects GPU availability:
- **JAX backend**: Uses `jax.default_backend()` to check for GPU
- **WSL detection**: Identifies Microsoft kernel for WSL-specific notes
- **Status badges**: Green badge shows GPU model, orange badge shows CPU/WSL fallback

To enable GPU in WSL2:
```bash
# Verify CUDA is accessible
nvidia-smi

# Check JAX GPU backend
python -c "import jax; print(jax.default_backend())"
# Should output: gpu
```

## Troubleshooting

### "QIHD Missing" badge
```bash
pip install -e OpenPhiSolve
```

### CSV upload error: "CSV has no numeric columns"
Ensure your CSV contains only numeric data (returns). Remove any date/text columns or convert them to numeric.

### GPU not detected (WSL)
```bash
# Install JAX with CUDA support
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Solver fails with large portfolios
Reduce `n_shots` or `n_steps` to lower memory usage, or switch device to `cpu`.

## Design Philosophy

The UI follows Anthropic's design principles:
- **Warm, neutral palette**: Terracotta (#C87941), forest green (#2D7A5F), deep blue (#2C4A6E)
- **Gradients**: Subtle linear gradients on backgrounds, buttons, and badges
- **Hierarchy**: Clear visual separation between sections
- **Cards & shadows**: Metric cards with hover effects and soft shadows
- **Typography**: Clean sans-serif with proper weight/spacing
- **Emojis**: Contextual icons for visual anchors (📊 📁 ⚖️ 🔧 📉)

## Performance

Typical solve times (WSL2 + NVIDIA GPU):
- **20 assets, K=8**: ~4-5 seconds (500 shots, 5000 steps)
- **50 assets, K=15**: ~8-12 seconds
- **100 assets, K=20**: ~20-30 seconds

Monte Carlo VaR (50k scenarios): ~0.5 seconds on GPU, ~2 seconds on CPU.

## License

Part of the WhatTheDuck portfolio optimization suite.
