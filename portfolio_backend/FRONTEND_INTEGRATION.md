# Frontend Integration Guide

## Architecture

```
Frontend (React + Recharts)          Backend (FastAPI + portfolio_backend)
  localhost:5173                        localhost:8000
  ┌──────────────┐                    ┌──────────────────────────┐
  │  App.tsx      │───/api/health────>│  server.py               │
  │               │───/api/ticker────>│    ├─ yfinance            │
  │  Vite proxy   │───/api/optimize──>│    ├─ portfolio_qihd      │
  │  /api -> 8000 │───/api/var───────>│    ├─ portfolio_backend   │
  └──────────────┘                    │    └─ VaRCalculator       │
                                      └──────────────────────────┘
```

## Starting the System

```bash
# Terminal 1: Backend
cd /path/to/WhatTheDuck
PYTHONPATH=OpenPhiSolve:. python portfolio_backend/server.py

# Terminal 2: Frontend
cd portfolio_frontend
npm run dev
```

Open http://localhost:5173

## API Endpoints

### GET /api/health

Returns server capabilities. Frontend uses this to show connection status and
disable features when optional dependencies are missing.

**Response:**
```json
{
  "status": "ok",
  "qihd_available": true,
  "backend_available": true,
  "yfinance_available": true
}
```

### POST /api/ticker-data

Fetches daily returns from yfinance.

**Request:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "lookback_days": 252
}
```

**Response:**
```json
{
  "success": true,
  "returns": [[0.012, -0.003, 0.005], ...],
  "asset_names": ["AAPL", "MSFT", "GOOGL"],
  "n_samples": 251,
  "n_assets": 3
}
```

### POST /api/optimize

Runs portfolio optimization (QIHD if available, CVXPY fallback).
Echoes back returns matrix so the frontend can pass it to /api/var.

**Request:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "mode": "cvar",
  "cardinality": 8,
  "max_weight": 0.25,
  "confidence_level": 0.95,
  "returns": [[...], ...],
  "asset_names": ["AAPL", "MSFT", "GOOGL"],
  "var_method": "classical_mc",
  "n_shots": 200,
  "n_steps": 5000,
  "dt": 0.15,
  "device": "gpu",
  "risk_aversion": 1.0
}
```

**Response:**
```json
{
  "success": true,
  "weights": [0.0, 0.30, 0.10, ...],
  "selected_assets": ["MSFT", "AAPL", "GOOGL"],
  "var_95": 0.0167,
  "cvar_95": 0.0226,
  "expected_return": 0.0012,
  "volatility": 0.0156,
  "sharpe_ratio": 0.77,
  "returns": [[...], ...],
  "metadata": {
    "solver": "qihd",
    "device": "gpu",
    "qihd_time": 4.65,
    "refinement_time": 0.0,
    "total_time": 5.28,
    "n_shots": 200,
    "n_steps": 5000,
    "active_assets": 4,
    "active_positions": [
      {"asset": "MSFT", "weight": 0.30},
      {"asset": "AAPL", "weight": 0.30}
    ],
    "n_samples": 251,
    "n_assets": 10,
    "cardinality_requested": 8,
    "max_weight": 0.25,
    "mode": "cvar"
  }
}
```

### POST /api/var

Calculates VaR using Classical MC or IQAE. Also returns VaR at
multiple confidence levels for the comparison chart.

**Request:**
```json
{
  "weights": [0.0, 0.30, 0.10, ...],
  "returns": [[...], ...],
  "method": "classical_mc",
  "confidence_level": 0.95,
  "distribution": "empirical"
}
```

**Response:**
```json
{
  "var_value": 0.0167,
  "cvar_value": 0.0226,
  "ci_lower": 0.0162,
  "ci_upper": 0.0172,
  "method": "classical_mc",
  "metadata": {
    "n_samples": 10000,
    "distribution": "empirical",
    "elapsed_time": 0.234,
    "var_90": 0.0130,
    "var_95": 0.0167,
    "var_99": 0.0225,
    "cvar_90": 0.0180,
    "cvar_95": 0.0226,
    "cvar_99": 0.0298
  }
}
```

## Frontend Data Flow

```
1. User enters tickers or uploads CSV
2. Click "Run Optimization"
3. If no CSV: POST /api/ticker-data -> get returns matrix
4. POST /api/optimize (with returns) -> weights, metrics, metadata
5. POST /api/var (classical_mc) \
                                  } in parallel -> VaR comparison data
   POST /api/var (iqae)          /
6. mapApiSnapshot() builds Snapshot from all three responses
7. React re-renders: allocation pie, risk bars, loss distribution,
   classical vs quantum VaR chart, analysis panel with metadata
```

## Graceful Degradation

The server starts even when optional dependencies are missing:

| Dependency | Missing behavior |
|---|---|
| phisolve (QIHD) | Falls back to portfolio_backend with CVXPY |
| portfolio_backend | Falls back to inline numpy VaR |
| yfinance | Ticker fetch returns 503; CSV upload still works |
| Classiq/Qiskit | IQAE VaR falls back to simulated mode |

The frontend checks /api/health on mount and shows appropriate warnings.
