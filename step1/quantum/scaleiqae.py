#!/usr/bin/env python3
"""
Optimized IQAE Scaling Demonstration
Incorporates stability fixes for machine precision and logarithmic epsilon sweeping.
Outputs results to the 'results2' directory.
"""

import argparse
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Plotting and Analysis
try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    plt = None
    pd = None

# Classiq Setup
CLASSIQ_AVAILABLE = True
try:
    from classiq import (
        QBit, QNum, QArray, Const, Constraints, Preferences,
        qfunc, qperm, inplace_prepare_state, show,
    )
    from classiq.applications.iqae.iqae import IQAE
except ImportError:
    CLASSIQ_AVAILABLE = False

@dataclass(frozen=True)
class PnLDistribution:
    pnl_grid: np.ndarray
    probs: np.ndarray
    num_qubits: int

    @property
    def N(self) -> int:
        return int(self.probs.shape[0])

def build_gaussian_dist(num_qubits: int, mu: float, sigma: float) -> PnLDistribution:
    """Discretize a Gaussian P&L distribution."""
    N = 2 ** num_qubits
    pnl_grid = np.linspace(mu - 4*sigma, mu + 4*sigma, N)
    probs = (1.0/(sigma * math.sqrt(2*math.pi))) * np.exp(-0.5*((pnl_grid-mu)/sigma)**2)
    return PnLDistribution(pnl_grid=pnl_grid, probs=probs/probs.sum(), num_qubits=num_qubits)

# -----------------------------
# Quantum Oracles
# -----------------------------
GLOBAL_INDEX = 0
RUNTIME_PROBS: List[float] = []

if CLASSIQ_AVAILABLE:
    @qfunc(synthesize_separately=True)
    def state_preparation(asset: QArray[QBit], ind: QBit):
        inplace_prepare_state(RUNTIME_PROBS, bound=0, target=asset)
        tail_oracle(asset=asset, ind=ind)

    @qperm
    def tail_oracle(asset: Const[QNum], ind: QBit):
        # Tail event: index <= GLOBAL_INDEX
        ind ^= asset <= GLOBAL_INDEX

def get_cost(res: Any) -> float:
    """Calculates total oracle queries: Σ (2k+1) * shots."""
    cost = 0.0
    # Search for iteration data across common Classiq result schemas
    iters = getattr(res, "iterations_data", getattr(res, "iters", []))
    for it in iters:
        k = getattr(it, "grover_iterations", getattr(it, "k", 0))
        shots = getattr(it, "num_shots", getattr(it, "shots", 0))
        if hasattr(it, "sample_results"):
            shots = getattr(it.sample_results, "num_shots", shots)
        cost += (2 * int(k) + 1) * int(shots)
    return cost

# -----------------------------
# Main Scaling Logic
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubits", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="results2")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not CLASSIQ_AVAILABLE:
        print("Classiq SDK not found. Please install to run.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup Problem: 5% Tail Prob on a 6-qubit grid
    dist = build_gaussian_dist(args.num_qubits, 0.0, 1.0)
    thr_idx = int(np.searchsorted(np.cumsum(dist.probs), 0.05, side="left"))
    p_true = dist.probs[:thr_idx+1].sum()

    global GLOBAL_INDEX, RUNTIME_PROBS
    GLOBAL_INDEX = thr_idx
    RUNTIME_PROBS = dist.probs.tolist()

    # Generate Geometric Epsilon Sweep (Halving sequence)
    eps_list = [0.1, 0.05, 0.025, 0.0125]
    data = []

    print(f"Scaling Demo Start: True Tail Prob = {p_true:.6f}")
    
    for eps in eps_list:
        print(f"  Processing Epsilon: {eps}...")
        
        # Stability Fix: Dynamically set machine precision based on target epsilon
        dynamic_precision = max(args.num_qubits, int(np.ceil(-np.log2(eps))) + 2)
        
        iqae = IQAE(
            state_prep_op=state_preparation,
            problem_vars_size=args.num_qubits,
            preferences=Preferences(machine_precision=dynamic_precision)
        )

        for r in range(args.repeats):
            res = iqae.run(epsilon=eps, alpha=0.05)
            err = abs(float(res.estimation) - p_true)
            cost = get_cost(res)
            data.append({"eps": eps, "error": err, "cost": cost, "run": r})

    # Save CSV
    if pd is not None:
        df = pd.DataFrame(data)
        df.to_csv(out_dir / "scaling_data.csv", index=False)
        
        if plt is not None:
            # Stability Fix: Use Median for plotting to smooth out simulation noise
            agg = df.groupby("eps").median().reset_index()
            agg = agg.sort_values("eps", ascending=False)

            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Error vs Cost (Standard Quantum Speedup Proof)
            ax[0].loglog(agg["cost"], agg["error"], 'o-', linewidth=2, label="Measured (Median)")
            
            # Theoretical Reference: O(1/N)
            ref_x = np.geomspace(agg["cost"].min(), agg["cost"].max(), 100)
            ref_y = (agg["error"].iloc[0] * agg["cost"].iloc[0]) / ref_x
            ax[0].loglog(ref_x, ref_y, 'k--', alpha=0.6, label="Theory O(1/N)")
            
            ax[0].set_title("Scaling: Error vs Oracle Queries")
            ax[0].set_xlabel("Queries (Cost)")
            ax[0].set_ylabel("Absolute Error")
            ax[0].legend()
            ax[0].grid(True, which="both", linestyle="--", alpha=0.5)

            # Plot 2: Cost vs 1/Epsilon (Resource Proof)
            ax[1].loglog(1/agg["eps"], agg["cost"], 's-', color='orange', linewidth=2, label="Measured Cost")
            
            # Theoretical Reference: O(1/eps)
            ref_y2 = (agg["cost"].iloc[0] / (1/agg["eps"].iloc[0])) * (1/np.geomspace(agg["eps"].max(), agg["eps"].min(), 100))
            ax[1].loglog(1/np.geomspace(agg["eps"].max(), agg["eps"].min(), 100), ref_y2, 'k--', alpha=0.6, label="Theory O(1/ε)")
            
            ax[1].set_title("Scaling: Cost vs Target Precision")
            ax[1].set_xlabel("Precision (1/ε)")
            ax[1].set_ylabel("Total Queries")
            ax[1].legend()
            ax[1].grid(True, which="both", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(out_dir / "optimized_scaling_proof.png", dpi=200)
            print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()