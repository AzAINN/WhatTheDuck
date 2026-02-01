"""
ESTIMATION-ONLY ERROR (GRID-TRUTH BASELINE)
==========================================
Compute estimation-only VaR error for MC and IQAE against a fixed discretized grid truth.

Method A: exact grid truth via discretized CDF (Gaussian only, fast, deterministic).
Optional: high-sample MC-on-grid truth.
"""

import os
import csv
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

from classiq import *
from classiq.applications.iqae.iqae import IQAE

# =========================
# CONFIG
# =========================

MU = 0.7
SIGMA = 0.13
ALPHA = 0.07
NUM_QUBITS = 8  # fixed grid resolution (2^n bins)

MC_BUDGETS = [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
IQAE_EPSILONS = [0.1, 0.07, 0.05, 0.03, 0.02, 0.01]
IQAE_RUNS = 10
MC_RUNS = 20

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# GRID TRUTH
# =========================

def discretized_grid(mu, sigma, num_qubits):
    low = mu - 3 * sigma
    high = mu + 3 * sigma
    grid = np.linspace(low, high, 2**num_qubits)
    probs = scipy.stats.norm.pdf(grid, loc=mu, scale=sigma)
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)
    return grid, probs, cdf


def grid_truth_var(mu, sigma, alpha, num_qubits):
    grid, probs, cdf = discretized_grid(mu, sigma, num_qubits)
    idx = int(np.argmax(cdf >= alpha))
    return float(grid[idx]), grid, probs, cdf


# =========================
# CLASSICAL MC (GRID-ALIGNED)
# =========================

def mc_var_on_grid(mu, sigma, alpha, num_qubits, n, rng):
    grid, probs, cdf = discretized_grid(mu, sigma, num_qubits)
    samples = rng.normal(loc=mu, scale=sigma, size=int(n))
    # Snap samples to nearest grid index to align with discretized world
    idx = np.clip(np.searchsorted(grid, samples, side="left"), 0, len(grid) - 1)
    # Empirical grid CDF
    counts = np.bincount(idx, minlength=len(grid))
    cdf_emp = np.cumsum(counts) / float(n)
    var_idx = int(np.argmax(cdf_emp >= alpha))
    return float(grid[var_idx])


# =========================
# IQAE (GRID-ALIGNED)
# =========================

PROBS = []
GLOBAL_INDEX: int = 0

@qfunc(synthesize_separately=True)
def state_preparation(asset: QArray[QBit], ind: QBit):
    load_distribution(asset=asset)
    payoff(asset=asset, ind=ind)


@qfunc
def load_distribution(asset: QNum):
    inplace_prepare_state(PROBS, bound=0, target=asset)


@qperm
def payoff(asset: Const[QNum], ind: QBit):
    ind ^= asset < GLOBAL_INDEX


def create_iqae_instance(num_qubits: int) -> IQAE:
    return IQAE(
        state_prep_op=state_preparation,
        problem_vars_size=num_qubits,
        constraints=Constraints(max_width=28),
        preferences=Preferences(machine_precision=num_qubits),
    )


def iqae_var_estimate(mu, sigma, alpha, num_qubits, epsilon, alpha_failure=0.01):
    grid, probs, cdf = discretized_grid(mu, sigma, num_qubits)
    global PROBS, GLOBAL_INDEX
    PROBS = probs.tolist()
    GLOBAL_INDEX = int(np.argmax(cdf >= alpha))

    iqae = create_iqae_instance(num_qubits)
    res = iqae.run(epsilon=epsilon, alpha=alpha_failure)
    a_hat = res.estimation
    # Map a_hat to grid VaR
    idx = int(np.argmax(cdf >= a_hat))
    var_est = float(grid[idx])
    return var_est


# =========================
# PLOTS
# =========================

def plot_with_band(ax, x, values, label):
    vals = np.array(values)
    med = np.median(vals, axis=0)
    p25 = np.percentile(vals, 25, axis=0)
    p75 = np.percentile(vals, 75, axis=0)
    ax.plot(x, med, marker="o", linewidth=2.0, label=label)
    ax.fill_between(x, p25, p75, alpha=0.15)


def main():
    var_grid_truth, grid, probs, cdf = grid_truth_var(MU, SIGMA, ALPHA, NUM_QUBITS)

    # MC estimation-only error
    rng = np.random.default_rng(1234)
    mc_errors = []
    for _ in range(MC_RUNS):
        run_errs = []
        for n in MC_BUDGETS:
            var_mc = mc_var_on_grid(MU, SIGMA, ALPHA, NUM_QUBITS, n, rng)
            run_errs.append(abs(var_mc - var_grid_truth))
        mc_errors.append(run_errs)

    # IQAE estimation-only error
    iqae_errors = []
    for _ in range(IQAE_RUNS):
        run_errs = []
        for eps in IQAE_EPSILONS:
            var_iqae = iqae_var_estimate(MU, SIGMA, ALPHA, NUM_QUBITS, eps)
            run_errs.append(abs(var_iqae - var_grid_truth))
        iqae_errors.append(run_errs)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_with_band(ax, MC_BUDGETS, mc_errors, "MC (estimation-only error)")
    plot_with_band(ax, range(len(IQAE_EPSILONS)), iqae_errors, "IQAE (estimation-only error)")

    ax.set_title("Estimation-Only VaR Error (Grid-Truth Baseline)")
    ax.set_xlabel("Budget (MC: samples, IQAE: epsilon index)")
    ax.set_ylabel("|VaR_est − VaR_grid*|")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    out_path = os.path.join(OUTPUT_DIR, "16_estimation_only_error.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("✓ Saved:", out_path)


if __name__ == "__main__":
    main()
