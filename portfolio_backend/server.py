"""
Portfolio Optimizer API Server
==============================
FastAPI server wrapping portfolio_backend and portfolio_qihd libraries.

Start:
    PYTHONPATH=OpenPhiSolve:. python portfolio_backend/server.py

Or:
    PYTHONPATH=OpenPhiSolve:. uvicorn portfolio_backend.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import queue
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Graceful imports — server starts even when optional deps are missing
# ---------------------------------------------------------------------------
_QIHD_AVAILABLE = False
try:
    from portfolio_qihd import (
        PortfolioSpec as QIHDSpec,
        solve_portfolio,
        compute_mu_cov,
        caps_from_constant,
    )
    _QIHD_AVAILABLE = True
except Exception:
    pass

_BACKEND_AVAILABLE = False
try:
    from portfolio_backend import (
        PortfolioOptimizer,
        PortfolioSpec,
        OptimizationMode,
        VaRCalculator,
        VaRSpec,
        VaRResult,
        VaRMethod,
    )
    _BACKEND_AVAILABLE = True
except Exception:
    pass

_YF_AVAILABLE = False
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except Exception:
    pass

_CLASSIQ_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quantum_VaR"))
    from VaR_Quantum import (
        setup_distribution, classical_var,
        quantum_value_at_risk, build_uniform_pmf_from_dist,
    )
    _CLASSIQ_AVAILABLE = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Portfolio Optimizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TickerRequest(BaseModel):
    tickers: List[str]
    lookback_days: int = 252

class TickerResponse(BaseModel):
    success: bool
    returns: List[List[float]]
    asset_names: List[str]
    n_samples: int
    n_assets: int

class OptimizeRequest(BaseModel):
    tickers: List[str] = []
    mode: str = "cvar"
    cardinality: int = 8
    max_weight: float = 0.25
    confidence_level: float = 0.95
    returns: Optional[List[List[float]]] = None
    asset_names: Optional[List[str]] = None
    var_method: str = "classical_mc"
    n_shots: int = 200
    n_steps: int = 5000
    dt: float = 0.15
    device: str = "gpu"
    risk_aversion: float = 1.0

class OptimizeResponse(BaseModel):
    success: bool
    weights: List[float]
    selected_assets: List[str]
    var_95: float
    cvar_95: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    returns: Optional[List[List[float]]] = None
    metadata: Dict[str, Any]

class VaRRequest(BaseModel):
    weights: List[float]
    returns: List[List[float]]
    method: str = "classical_mc"
    confidence_level: float = 0.95
    distribution: str = "empirical"

class VaRResponse(BaseModel):
    var_value: float
    cvar_value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    method: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_returns(tickers: List[str], lookback_days: int = 252):
    """Fetch daily returns from yfinance."""
    if not _YF_AVAILABLE:
        raise HTTPException(status_code=503, detail="yfinance not installed on server")

    symbols = [t.strip().upper() for t in tickers if t.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="No valid ticker symbols")

    end = datetime.today()
    start = end - timedelta(days=int(lookback_days * 1.5))  # extra margin for weekends

    try:
        data = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"yfinance download failed: {exc}")

    if data.empty:
        raise HTTPException(status_code=404, detail="No data returned for given tickers")

    # Handle single vs multi-ticker DataFrames
    import pandas as pd
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            prices = data.iloc[:, :len(symbols)]
    else:
        prices = data[["Close"]] if "Close" in data.columns else data
        if len(symbols) == 1:
            prices.columns = symbols

    returns_df = prices.pct_change().dropna(how="all").dropna(axis=1, how="all")
    if returns_df.empty or returns_df.shape[0] < 10:
        raise HTTPException(status_code=404, detail="Not enough history to compute returns")

    # Trim to requested lookback
    returns_df = returns_df.tail(lookback_days)

    return returns_df.values, [str(c) for c in returns_df.columns]


def _inline_mc_var(weights: np.ndarray, returns_np: np.ndarray,
                   confidence_level: float, n_samples: int = 10000):
    """Simple inline MC VaR when portfolio_backend is unavailable."""
    rng = np.random.default_rng(42)
    idx = rng.integers(0, returns_np.shape[0], size=n_samples)
    pnl = returns_np[idx] @ weights
    losses = -pnl
    var_val = float(np.quantile(losses, confidence_level))
    tail = losses[losses >= var_val]
    cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val

    # Multi-confidence for comparison chart
    multi = {}
    for cl in [0.90, 0.95, 0.99]:
        multi[f"var_{int(cl * 100)}"] = float(np.quantile(losses, cl))
        t = losses[losses >= multi[f"var_{int(cl * 100)}"]]
        multi[f"cvar_{int(cl * 100)}"] = float(np.mean(t)) if len(t) > 0 else multi[f"var_{int(cl * 100)}"]

    # Bootstrap CI
    ci_lower, ci_upper = None, None
    try:
        boot_vars = []
        for _ in range(500):
            b_idx = rng.integers(0, len(losses), size=len(losses))
            boot_vars.append(float(np.quantile(losses[b_idx], confidence_level)))
        ci_lower = float(np.percentile(boot_vars, 2.5))
        ci_upper = float(np.percentile(boot_vars, 97.5))
    except Exception:
        pass

    return var_val, cvar_val, ci_lower, ci_upper, multi


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "qihd_available": _QIHD_AVAILABLE,
        "backend_available": _BACKEND_AVAILABLE,
        "yfinance_available": _YF_AVAILABLE,
        "classiq_available": _CLASSIQ_AVAILABLE,
    }


@app.post("/api/ticker-data", response_model=TickerResponse)
async def fetch_ticker_data(request: TickerRequest):
    returns_np, asset_names = _fetch_returns(request.tickers, request.lookback_days)
    return TickerResponse(
        success=True,
        returns=returns_np.tolist(),
        asset_names=asset_names,
        n_samples=returns_np.shape[0],
        n_assets=returns_np.shape[1],
    )


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize_portfolio(request: OptimizeRequest):
    # --- Step 1: Get returns data ---
    returns_np: Optional[np.ndarray] = None
    asset_names: List[str] = request.asset_names or []

    if request.returns is not None and len(request.returns) > 0:
        returns_np = np.array(request.returns, dtype=float)
        if not asset_names:
            asset_names = [f"Asset {i+1}" for i in range(returns_np.shape[1])]
    elif request.tickers and _YF_AVAILABLE:
        arr, names = _fetch_returns(request.tickers)
        returns_np = arr
        asset_names = names
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide returns data or tickers (requires yfinance on server)",
        )

    n_samples, n_assets = returns_np.shape

    # --- Step 2: Optimize ---
    t0 = time.time()
    solver_name = "none"
    qihd_time = 0.0
    refinement_time = 0.0
    device_used = "cpu"

    if _QIHD_AVAILABLE:
        mu, cov = compute_mu_cov(returns_np)
        caps = caps_from_constant(request.max_weight, n_assets)
        k = min(request.cardinality, n_assets)

        spec = QIHDSpec(
            mu=mu, cov=cov, upper=caps,
            max_positions=k, lambda_risk=request.risk_aversion,
        )
        backend_kwargs = dict(
            n_shots=request.n_shots, n_steps=request.n_steps,
            dt=request.dt, device=request.device,
        )

        t_qihd = time.time()
        result = solve_portfolio(spec, backend_kwargs=backend_kwargs, if_refine=False)
        qihd_time = time.time() - t_qihd

        weights = np.clip(result.weights, 0, None)
        wsum = weights.sum()
        if wsum > 0:
            weights = weights / wsum
        else:
            weights = np.ones(n_assets) / n_assets

        solver_name = "qihd"
        device_used = request.device

    elif _BACKEND_AVAILABLE:
        mode_enum = OptimizationMode(request.mode)
        spec = PortfolioSpec(
            returns=returns_np,
            mode=mode_enum,
            cardinality=min(request.cardinality, n_assets),
            max_weight=request.max_weight,
            confidence_level=request.confidence_level,
            risk_aversion=request.risk_aversion,
            solver="cvxpy",
        )
        opt_result = PortfolioOptimizer(spec).optimize()
        weights = opt_result.weights
        solver_name = opt_result.metadata.get("solver", "cvxpy")
        refinement_time = opt_result.metadata.get("refinement_time", 0.0)
    else:
        raise HTTPException(
            status_code=503,
            detail="No solver available (install phisolve or cvxpy)",
        )

    total_time = time.time() - t0

    # --- Step 3: Compute portfolio metrics ---
    mu_vec = np.mean(returns_np, axis=0)
    cov_mat = np.cov(returns_np, rowvar=False)
    if cov_mat.ndim == 0:
        cov_mat = np.array([[float(cov_mat)]])

    expected_return = float(np.dot(mu_vec, weights))
    port_var = float(weights @ cov_mat @ weights)
    volatility = float(np.sqrt(max(port_var, 0)))
    sharpe = expected_return / volatility if volatility > 1e-9 else 0.0

    # Quick historical VaR
    pnl = returns_np @ weights
    losses = -pnl
    var_95 = float(np.quantile(losses, request.confidence_level))
    tail = losses[losses >= var_95]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

    # Active positions
    active_mask = weights > 1e-6
    selected_assets = [asset_names[i] for i in range(n_assets) if i < len(asset_names) and active_mask[i]]
    active_positions = [
        {"asset": asset_names[i] if i < len(asset_names) else f"Asset {i+1}",
         "weight": round(float(weights[i]), 6)}
        for i in range(n_assets) if active_mask[i]
    ]
    active_positions.sort(key=lambda x: x["weight"], reverse=True)

    return OptimizeResponse(
        success=True,
        weights=weights.tolist(),
        selected_assets=selected_assets,
        var_95=var_95,
        cvar_95=cvar_95,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
        returns=returns_np.tolist(),
        metadata={
            "solver": solver_name,
            "device": device_used,
            "qihd_time": round(qihd_time, 3),
            "refinement_time": round(refinement_time, 3),
            "total_time": round(total_time, 3),
            "n_shots": request.n_shots if _QIHD_AVAILABLE else 0,
            "n_steps": request.n_steps if _QIHD_AVAILABLE else 0,
            "active_assets": int(np.sum(active_mask)),
            "active_positions": active_positions,
            "n_samples": n_samples,
            "n_assets": n_assets,
            "cardinality_requested": request.cardinality,
            "max_weight": request.max_weight,
            "mode": request.mode,
        },
    )


@app.post("/api/var", response_model=VaRResponse)
async def calculate_var(request: VaRRequest):
    weights = np.array(request.weights, dtype=float)
    returns_np = np.array(request.returns, dtype=float)

    wsum = weights.sum()
    if wsum > 0 and not np.isclose(wsum, 1.0):
        weights = weights / wsum

    if _BACKEND_AVAILABLE:
        method_enum = VaRMethod(request.method)
        spec = VaRSpec(
            portfolio_weights=weights,
            returns_data=returns_np,
            method=method_enum,
            confidence_level=request.confidence_level,
            distribution=request.distribution,
            n_samples=10000,
        )

        result = VaRCalculator(spec).calculate()

        # Compute multi-confidence VaR for comparison chart
        pnl = returns_np @ weights
        losses = -pnl
        multi = {}
        for cl in [0.90, 0.95, 0.99]:
            v = float(np.quantile(losses, cl))
            multi[f"var_{int(cl * 100)}"] = v
            t = losses[losses >= v]
            multi[f"cvar_{int(cl * 100)}"] = float(np.mean(t)) if len(t) > 0 else v

        meta = dict(result.metadata) if result.metadata else {}
        meta.update(multi)

        return VaRResponse(
            var_value=result.var_value,
            cvar_value=result.cvar_value,
            ci_lower=result.confidence_interval[0] if result.confidence_interval else None,
            ci_upper=result.confidence_interval[1] if result.confidence_interval else None,
            method=result.method,
            metadata=meta,
        )
    else:
        # Inline fallback
        var_val, cvar_val, ci_lower, ci_upper, multi = _inline_mc_var(
            weights, returns_np, request.confidence_level,
        )
        return VaRResponse(
            var_value=var_val,
            cvar_value=cvar_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            method="classical_mc_inline",
            metadata={
                "n_samples": 10000,
                "note": "Inline fallback (portfolio_backend not available)",
                **multi,
            },
        )


# ---------------------------------------------------------------------------
# Distribution endpoint
# ---------------------------------------------------------------------------

class DistributionRequest(BaseModel):
    weights: List[float]
    returns: List[List[float]]
    n_bins: int = 100

@app.post("/api/distribution")
async def get_distribution(request: DistributionRequest):
    weights = np.array(request.weights, dtype=float)
    returns_np = np.array(request.returns, dtype=float)
    pnl = returns_np @ weights  # positive = profit
    hist, bin_edges = np.histogram(pnl, bins=request.n_bins, density=True)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Return as fractional P&L (not ×100) — frontend scales by portfolio value
    points = [{"pnl": float(c), "density": float(d)} for c, d in zip(centres, hist)]
    return {
        "points": points,
        "mean": float(np.mean(pnl)),
        "std": float(np.std(pnl)),
    }


# ---------------------------------------------------------------------------
# SSE streaming VaR endpoint
# ---------------------------------------------------------------------------

class VaRStreamRequest(BaseModel):
    weights: List[float]
    returns: List[List[float]]
    confidence_level: float = 0.95
    num_qubits: int = 7
    iqae_epsilon: float = 0.05
    mc_samples: int = 10000


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/api/var-stream")
async def var_stream(request: VaRStreamRequest):
    weights = np.array(request.weights, dtype=float)
    returns_np = np.array(request.returns, dtype=float)
    alpha = 1.0 - request.confidence_level  # tail probability
    num_qubits = request.num_qubits
    mc_samples = request.mc_samples

    q: queue.Queue = queue.Queue()

    def _worker():
        try:
            # --- Phase 1: Classical MC VaR ---
            # Convention: VaR/CVaR returned as POSITIVE fractions of portfolio
            # e.g. var_val=0.023 means 2.3% loss, CVaR >= VaR always
            q.put(("phase", "Computing classical MC VaR..."))
            var_val, cvar_val, ci_lower, ci_upper, multi = _inline_mc_var(
                weights, returns_np, request.confidence_level, n_samples=mc_samples,
            )
            q.put(("classical_result", {
                "var_value": var_val,
                "cvar_value": cvar_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "method": "classical_mc",
                **multi,
            }))

            # --- Phase 2: Quantum IQAE VaR ---
            if not _CLASSIQ_AVAILABLE:
                q.put(("log", "Classiq not available — running simulated quantum VaR"))
                q.put(("phase", "Running simulated quantum VaR..."))
                q.put(("quantum_result", {
                    "var_value": var_val,
                    "cvar_value": cvar_val,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "method": "iqae_simulated",
                    "oracle_queries": 0,
                    "bisection_steps": 0,
                    "note": "Classiq not installed — simulated result",
                }))
                q.put(("done", ""))
                return

            q.put(("phase", "Discretizing portfolio P&L..."))
            q.put(("log", f"Discretizing into 2^{num_qubits} = {2**num_qubits} bins"))

            # Discretize portfolio P&L (positive = profit, negative = loss)
            # The quantum module's classical_var finds CDF reaching alpha
            # from the LEFT (worst outcomes), which is correct for P&L.
            pnl = returns_np @ weights
            n_bins = 2 ** num_qubits
            hist, bin_edges = np.histogram(pnl, bins=n_bins)
            pmf = hist / hist.sum()
            grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            pmf_list = pmf.tolist()

            q.put(("log", f"P&L range: [{grid[0]:.6f}, {grid[-1]:.6f}]"))

            # Setup module globals for quantum circuits
            setup_distribution(grid, pmf_list, num_qubits)
            q.put(("log", "Distribution loaded into quantum module"))

            # Run pure IQAE bisection (no warm start)
            q.put(("phase", "Running IQAE bisection..."))
            q.put(("log", f"alpha={alpha:.4f}  epsilon={request.iqae_epsilon}  pure IQAE (no warm start)"))

            # Capture stdout from quantum_value_at_risk
            old_stdout = sys.stdout
            capture = io.StringIO()

            class TeeWriter:
                def __init__(self, q, capture):
                    self._q = q
                    self._capture = capture
                    self._buf = ""
                def write(self, s):
                    self._capture.write(s)
                    self._buf += s
                    while "\n" in self._buf:
                        line, self._buf = self._buf.split("\n", 1)
                        if line.strip():
                            self._q.put(("log", line.strip()))
                def flush(self):
                    if self._buf.strip():
                        self._q.put(("log", self._buf.strip()))
                    self._buf = ""
                    self._capture.flush()

            sys.stdout = TeeWriter(q, capture)
            try:
                q_idx, q_var_pnl, q_est, q_ci, q_queries, q_steps = quantum_value_at_risk(
                    grid, pmf_list, alpha,
                    epsilon=request.iqae_epsilon,
                    bracket=None,  # Pure IQAE — no warm start
                )
            finally:
                sys.stdout.flush()
                sys.stdout = old_stdout

            # q_var_pnl is on the P&L scale (negative = loss)
            # VaR = positive magnitude of the loss threshold
            q_var_positive = abs(float(q_var_pnl))

            # CVaR: average of P&L values in the tail WORSE than VaR
            # (i.e. pnl values <= q_var_pnl, which are more negative)
            grid_arr = np.array(grid)
            pmf_arr = np.array(pmf_list)
            tail_mask = grid_arr <= float(q_var_pnl)
            tail_grid = grid_arr[tail_mask]
            tail_pmf = pmf_arr[tail_mask]
            if tail_pmf.sum() > 1e-12:
                q_cvar_pnl = float(np.average(tail_grid, weights=tail_pmf))
                q_cvar_positive = abs(q_cvar_pnl)
            else:
                q_cvar_positive = q_var_positive

            # Ensure CVaR >= VaR (mathematical requirement)
            q_cvar_positive = max(q_cvar_positive, q_var_positive)

            q.put(("log", f"Quantum VaR (positive): {q_var_positive:.6f}  CVaR: {q_cvar_positive:.6f}"))

            q.put(("quantum_result", {
                "var_value": q_var_positive,
                "cvar_value": q_cvar_positive,
                "ci_lower": abs(float(q_ci[0])) if q_ci else None,
                "ci_upper": abs(float(q_ci[1])) if q_ci else None,
                "method": "iqae",
                "oracle_queries": q_queries,
                "bisection_steps": q_steps,
            }))
            q.put(("done", ""))

        except Exception as exc:
            q.put(("error", str(exc)))
            q.put(("done", ""))

    def _generate():
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        while True:
            try:
                event, data = q.get(timeout=60)
            except queue.Empty:
                yield _sse_event("error", "Timeout waiting for result")
                break
            yield _sse_event(event, data)
            if event == "done":
                break

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print(f"[server] QIHD available: {_QIHD_AVAILABLE}")
    print(f"[server] Backend available: {_BACKEND_AVAILABLE}")
    print(f"[server] yfinance available: {_YF_AVAILABLE}")
    print(f"[server] Classiq available: {_CLASSIQ_AVAILABLE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
