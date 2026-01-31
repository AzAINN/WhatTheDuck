# estimators_quantum.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from classiq import (
    qfunc,
    qperm,
    QArray,
    QBit,
    QNum,
    Const,
    inplace_prepare_state,
    Constraints,
    Preferences,
)
from classiq.applications.iqae.iqae import IQAE

from structures import TailQuery, ProbEstimate, QuantumProbEstimator

# These are module-level because Classiq decorators run at definition time.
GLOBAL_INDEX = 0
PROBS: list[float] = []


@qfunc(synthesize_separately=True)
def state_preparation(asset: QArray[QBit], ind: QBit):
    load_distribution(asset=asset)
    payoff(asset=asset, ind=ind)


@qfunc
def load_distribution(asset: QNum):
    inplace_prepare_state(PROBS, bound=0, target=asset)


@qperm
def payoff(asset: Const[QNum], ind: QBit):
    # Marks states with asset < GLOBAL_INDEX
    ind ^= asset < GLOBAL_INDEX


def _iqae_cost_from_result(iqae_res: Any, cost_mode: str = "A_calls") -> tuple[int, Dict[str, Any]]:
    """
    Uses iqae_res.iterations_data[*].grover_iterations (k) and sample_results.num_shots.
    cost_mode:
      - "shots": sum of shots across iterations
      - "A_calls": sum shots * (2k + 1)    (common AE accounting)
      - "AA_calls": sum shots * (4k + 1)   (roughly counts A and A† usage)
    """
    meta: Dict[str, Any] = {"cost_mode": cost_mode}
    total = 0

    iters = getattr(iqae_res, "iterations_data", None)
    if iters is None:
        # Fall back: unknown structure
        meta["cost_warning"] = "IQAEResult has no iterations_data; cost set to 0"
        return 0, meta

    per_iter = []
    for item in iters:
        k = int(getattr(item, "grover_iterations", 0))
        sample_results = getattr(item, "sample_results", None)
        shots = int(getattr(sample_results, "num_shots", 0)) if sample_results is not None else 0

        if cost_mode == "shots":
            c = shots
        elif cost_mode == "AA_calls":
            c = shots * (4 * k + 1)
        else:  # default "A_calls"
            c = shots * (2 * k + 1)

        total += c
        per_iter.append({"k": k, "shots": shots, "iter_cost": c})

    meta["per_iteration"] = per_iter
    meta["sum_shots"] = int(sum(x["shots"] for x in per_iter))
    return int(total), meta


class ClassiqIQAECDFEstimator(QuantumProbEstimator):
    """
    Quantum estimator using Classiq IQAE to estimate p = P(asset < index).

    Cost accounting:
      - derived from IQAEResult.iterations_data (Grover powers and shots)
    """

    def __init__(
        self,
        *,
        probs: list[float],
        num_qubits: int,
        constraints: Optional[Constraints] = None,
        preferences: Optional[Preferences] = None,
        cost_mode: str = "A_calls",
    ):
        global PROBS
        PROBS = list(probs)
        self.num_qubits = int(num_qubits)
        self.constraints = constraints or Constraints(max_width=28)
        self.preferences = preferences or Preferences(machine_precision=num_qubits)
        self.cost_mode = cost_mode

    def estimate_tail_prob(
        self,
        query: TailQuery,
        *,
        epsilon: float,
        alpha: float,
        max_total_queries: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> ProbEstimate:
        if query.index is None:
            raise ValueError("ClassiqIQAECDFEstimator requires query.index")

        global GLOBAL_INDEX
        GLOBAL_INDEX = int(query.index)

        iqae = IQAE(
            state_prep_op=state_preparation,
            problem_vars_size=self.num_qubits,
            constraints=self.constraints,
            preferences=self.preferences,
        )

        # Note: Classiq IQAE primitive is adaptive; seed may not apply depending on backend.
        iqae_res = iqae.run(epsilon=float(epsilon), alpha=float(alpha))

        p_hat = float(iqae_res.estimation)
        ci = [float(x) for x in iqae_res.confidence_interval]
        ci_low, ci_high = float(ci[0]), float(ci[1])

        cost, cost_meta = _iqae_cost_from_result(iqae_res, cost_mode=self.cost_mode)

        # Optional hard cap for sweeps; if exceeded, record it (you can enforce externally too).
        cap_hit = False
        if max_total_queries is not None and cost > int(max_total_queries):
            cap_hit = True

        meta: Dict[str, Any] = {
            "method": "classiq_iqae",
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "seed": seed,
            "max_total_queries": max_total_queries,
            "cap_hit": cap_hit,
        }
        meta.update(cost_meta)

        return ProbEstimate(
            p_hat=p_hat,
            ci_low=ci_low,
            ci_high=ci_high,
            cost=int(cost),
            meta=meta,
        )
