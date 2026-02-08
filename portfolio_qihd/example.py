"""CLI example: solve a cardinality-constrained mean–variance portfolio with QIHD."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .data import compute_mu_cov, synthetic_factor_model, caps_from_constant
from .miqp_builder import PortfolioSpec
from .solver import solve_portfolio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=int, default=50, help="Number of assets M")
    p.add_argument("--factors", type=int, default=5, help="Number of latent factors")
    p.add_argument("--samples", type=int, default=4096, help="Return samples for covariance")
    p.add_argument("--k", type=int, default=10, help="Max active assets (cardinality)")
    p.add_argument("--lambda-risk", type=float, default=5.0, help="Risk aversion λ")
    p.add_argument("--cap", type=float, default=0.10, help="Per-asset weight cap u_i")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument("--n-shots", type=int, default=500, help="QIHD trajectories")
    p.add_argument("--n-steps", type=int, default=5000, help="QIHD steps")
    p.add_argument("--dt", type=float, default=0.2, help="QIHD step size")
    p.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"], help="Execution device")
    p.add_argument("--refine", action="store_true", help="Enable PDQP refinement")
    p.add_argument("--output", type=Path, default=Path("portfolio_qihd/outputs/weights.npy"), help="Where to save weights")
    return p.parse_args()


def main(args: argparse.Namespace | None = None):
    args = args or parse_args()

    returns = synthetic_factor_model(
        n_samples=args.samples,
        n_assets=args.assets,
        n_factors=args.factors,
        seed=args.seed,
    )
    mu, cov = compute_mu_cov(returns)
    caps = caps_from_constant(args.cap, args.assets)

    spec = PortfolioSpec(
        mu=mu,
        cov=cov,
        upper=caps,
        max_positions=args.k,
        lambda_risk=args.lambda_risk,
    )

    result = solve_portfolio(
        spec,
        backend_kwargs=dict(
            n_shots=args.n_shots,
            n_steps=args.n_steps,
            dt=args.dt,
            device=args.device,
        ),
        if_refine=args.refine,
    )

    selected_idx = np.nonzero(result.selection > 0.5)[0]
    total_weight = float(result.weights.sum())

    print(f"Selected {len(selected_idx)} / {args.assets} assets (K={args.k})")
    print("Indices:", selected_idx.tolist())
    print("Weights (first 10):", np.round(result.weights, 4)[:10].tolist())
    print(f"Sum weights: {total_weight:.4f}")
    print(f"Objective λ w^T Σ w − μ^T w = {result.objective:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result.weights)
    print(f"Saved weights to {args.output}")

    return result


if __name__ == "__main__":
    main()
