"""Constructs a cardinality-constrained mean–variance portfolio MIQP for OpenPhiSolve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from phisolve.problems import MIQP


@dataclass
class PortfolioSpec:
    """Problem inputs for the portfolio optimizer."""

    mu: np.ndarray                 # Expected returns, shape (M,)
    cov: np.ndarray                # Covariance matrix Σ, shape (M, M)
    upper: np.ndarray              # Per-asset cap u_i, shape (M,)
    max_positions: int             # Cardinality cap K
    lambda_risk: float = 1.0       # Risk aversion λ
    budget: float = 1.0            # Total capital (usually 1 for weights)


def _validate_spec(spec: PortfolioSpec) -> int:
    mu = np.asarray(spec.mu, dtype=float)
    cov = np.asarray(spec.cov, dtype=float)
    upper = np.asarray(spec.upper, dtype=float)

    if mu.ndim != 1:
        raise ValueError("mu must be 1-D")
    M = mu.shape[0]
    if cov.shape != (M, M):
        raise ValueError(f"cov must be ({M},{M}), got {cov.shape}")
    if upper.shape != (M,):
        raise ValueError(f"upper must be length {M}, got {upper.shape}")
    if spec.max_positions <= 0 or spec.max_positions > M:
        raise ValueError("max_positions must be in [1, M]")
    if np.any(upper <= 0):
        raise ValueError("upper caps must be positive")
    return M


def build_portfolio_miqp(spec: PortfolioSpec) -> MIQP:
    """
    Encode the portfolio problem as an MIQP compatible with OpenPhiSolve.

    Variable order: x = [y_0..y_{M-1}, w_0..w_{M-1}]
    - y_i ∈ {0,1}
    - w_i ≥ 0
    """

    M = _validate_spec(spec)

    # Quadratic term: only on weights block (risk term λ w^T Σ w)
    Q_w = 2.0 * spec.lambda_risk * np.asarray(spec.cov, dtype=float)
    Q = sp.block_diag(
        [sp.csr_matrix((M, M)), sp.csr_matrix(Q_w)], format="csr"
    )

    # Linear term: -μ on weights, 0 on binaries
    w_linear = -np.asarray(spec.mu, dtype=float)
    c = np.concatenate([np.zeros(M, dtype=float), w_linear])

    # Inequalities A x <= b
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_list: list[float] = []
    row_idx = 0

    # Linking: w_i - u_i y_i <= 0
    for i, cap in enumerate(np.asarray(spec.upper, dtype=float)):
        rows.extend([row_idx, row_idx])
        cols.extend([i, M + i])          # y_i, w_i
        data.extend([-cap, 1.0])
        b_list.append(0.0)
        row_idx += 1

    # Cardinality: sum y_i <= K
    rows.extend([row_idx] * M)
    cols.extend(list(range(M)))
    data.extend([1.0] * M)
    b_list.append(float(spec.max_positions))
    row_idx += 1

    A = sp.coo_matrix((data, (rows, cols)), shape=(row_idx, 2 * M)).tocsr()
    b = np.asarray(b_list, dtype=float)

    # Equality: budget sum w_i = budget
    C = sp.csr_matrix(([1.0] * M, ([0] * M, list(range(M, 2 * M)))), shape=(1, 2 * M))
    d = np.asarray([float(spec.budget)], dtype=float)

    # Bounds
    lb = np.zeros(2 * M, dtype=float)
    ub = np.concatenate([np.ones(M, dtype=float), np.asarray(spec.upper, dtype=float)])

    return MIQP(
        Q=Q,
        w=c,
        A=A,
        b=b,
        C=C,
        d=d,
        bounds=(lb, ub),
        n_binary_vars=M,
    )


def split_solution(x: Sequence[float], n_assets: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split a full solution vector x into (y, w)."""
    arr = np.asarray(x, dtype=float)
    if arr.shape[0] != 2 * n_assets:
        raise ValueError(f"expected solution length {2*n_assets}, got {arr.shape[0]}")
    y = arr[:n_assets]
    w = arr[n_assets:]
    return y, w
