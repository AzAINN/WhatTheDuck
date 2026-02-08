"""Solver wrapper that runs OpenPhiSolve on the portfolio MIQP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from phisolve import QIHD
from phisolve.phi_miqp import PhiMIQP

from .miqp_builder import PortfolioSpec, build_portfolio_miqp, split_solution


@dataclass
class PortfolioResult:
    weights: np.ndarray          # w
    selection: np.ndarray        # y (binary)
    objective: float             # λ w^T Σ w − μ^T w
    response: Any                # PhiMIQP response (coarse + refined samples)


def solve_portfolio(
    spec: PortfolioSpec,
    *,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    refiner_iters: int = 500,
    if_refine: bool = False,
) -> PortfolioResult:
    """Build the MIQP and solve it with QIHD (+ optional PDQP refinement)."""
    backend_kwargs = backend_kwargs or {}
    problem = build_portfolio_miqp(spec)

    backend = QIHD(**backend_kwargs)
    refiner = None
    if if_refine:
        # Lazy import to avoid requiring mpax unless refinement is requested.
        from phisolve import PDQP
        refiner = PDQP(iterations=refiner_iters)

    solver = PhiMIQP(
        problem_instance=problem,
        backend=backend,
        refiner=refiner,
    )

    response = solver.solve(if_refine=if_refine)
    best = np.asarray(response.minimizer, dtype=float)
    y, w = split_solution(best, len(spec.mu))
    obj = float(spec.lambda_risk * w @ spec.cov @ w - spec.mu @ w)
    return PortfolioResult(weights=w, selection=y, objective=obj, response=response)
