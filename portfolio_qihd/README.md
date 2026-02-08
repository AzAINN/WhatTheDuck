# Portfolio MIQP with OpenPhiSolve (QIHD)

This folder adds a self‑contained, mean–variance + cardinality portfolio optimizer built on **OpenPhiSolve**’s MIQP stack. It leaves existing project files untouched.

## Model (y first, then w)
- Decision vector: `x = [y_0..y_{M-1}, w_0..w_{M-1}]`
- Objective: minimize `λ · wᵀΣw – μᵀw`
- Constraints:
  - Budget: `∑ w_i = 1`
  - Long-only: `w_i ≥ 0`
  - Linking: `w_i − u_i y_i ≤ 0` (caps activate only if selected)
  - Cardinality: `∑ y_i ≤ K`
- MIQP encoding uses `MIQP(Q, w, A, b, C, d, bounds, n_binary_vars=M)` with first `M` variables binary.

## Setup
```bash
cd /mnt/c/Users/azain/WhatTheDuck
pip install -e OpenPhiSolve        # editable install to expose `phisolve`
pip install -r requirements.txt    # if not already installed
```

## Try the synthetic example
```bash
python -m portfolio_qihd.example --assets 50 --k 10 --lambda-risk 5.0 --seed 7
```
It prints selected tickers, weights, objective value, and saves the weight vector to `portfolio_qihd/outputs/weights.npy` for downstream VaR code.

## Integrating with your VaR pipeline
The solver returns a NumPy weight vector `w` (length M). Feed it directly into your GPU Monte Carlo → discretize → QAE/IQAE pipeline:
```python
from portfolio_qihd.example import main
w = main(...).weights
```
Or read the saved `weights.npy` file and continue with the VaR scripts already in the repo.

## File map
- `miqp_builder.py` — constructs the MIQP matrices for the portfolio model.
- `data.py` — utilities to compute `μ, Σ` from returns or generate a synthetic factor model.
- `solver.py` — runs QIHD/PDQP and extracts `y, w`.
- `example.py` — CLI demo wiring everything together.
