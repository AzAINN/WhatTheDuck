# estimators_classical.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from structures import TailQuery, ProbEstimate, ClassicalProbEstimator


def _normal_approx_z(confidence: float) -> float:
    """
    Two-sided z for confidence level (e.g., 0.99 -> z ~ 2.575).
    Uses an accurate approximation via erfinv if available; otherwise fallback.
    """
    # Convert confidence to two-sided tail
    # confidence = 1 - delta => delta/2 per tail
    delta = 1.0 - float(confidence)
    p = 1.0 - delta / 2.0
    # Inverse normal CDF approximation
    # Use scipy if you have it, but keep this file dependency-free:
    # Abramowitz-Stegun approximation
    # (good enough for CI bounds used in search decisions)
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924

    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857

    c1 = -0.00778489400243029
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878

    d1 = 0.00778469570904146
    d2 = 0.32246712907004
    d3 = 2.445134137143
    d4 = 3.75440866190742

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    ) / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)


def _wilson_ci(k: int, n: int, confidence: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z = _normal_approx_z(confidence)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


class DiscreteMonteCarloEstimator(ClassicalProbEstimator):
    """
    Estimates p = P(sample_index < query.index) by sampling from the discrete distribution `probs`.

    Cost accounting:
      - cost = number of samples used (<= budget)
    """

    def __init__(self, probs: list[float]):
        self._probs = np.array(probs, dtype=float)
        self._support = np.arange(len(probs), dtype=int)

    def estimate_tail_prob(
        self,
        query: TailQuery,
        *,
        budget: int,
        confidence: float = 0.99,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> ProbEstimate:
        if query.index is None:
            raise ValueError("DiscreteMonteCarloEstimator requires query.index")
        n = int(budget)
        if n <= 0:
            raise ValueError("budget must be positive")

        rng = np.random.default_rng(seed)
        samples = rng.choice(self._support, size=n, p=self._probs)

        # Tail event consistent with the notebook: CDF(index) = P(bin < index)
        successes = int(np.sum(samples < int(query.index)))
        p_hat = successes / n
        ci_low, ci_high = _wilson_ci(successes, n, confidence)

        return ProbEstimate(
            p_hat=float(p_hat),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            cost=int(n),
            meta={
                "method": "discrete_mc",
                "seed": seed,
                "successes": successes,
                "n": n,
                "confidence": confidence,
            },
        )
