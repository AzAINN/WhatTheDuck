"""
IQAE SENSITIVITY SWEEP
======================
Runs IQAE sweeps across:
- confidence levels (VaR alpha: 0.05, 0.01)
- discretization resolutions (num_qubits)
- estimation precision (epsilon list)

Outputs: results/iqae_sensitivity_sweep.csv
"""

import csv
import os
import numpy as np
import scipy

from classiq import *
from classiq.applications.iqae.iqae import IQAE

# =========================
# CONFIG
# =========================

MU = 0.7
SIGMA = 0.13

VAR_ALPHAS = [0.05, 0.01]  # 95% and 99% confidence
NUM_QUBITS_LIST = [6, 7, 8]

EPSILONS = [
    0.3, 0.25, 0.2, 0.17, 0.14, 0.12, 0.1, 0.085,
    0.07, 0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025,
    0.02, 0.017, 0.015, 0.012, 0.01
]

RUNS = 3
ALPHA_FAILURE = 0.01  # IQAE failure probability (confidence 99%)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "iqae_sensitivity_sweep.csv")

# =========================
# GLOBALS used by qfunc
# =========================

PROBS = []
GLOBAL_INDEX: int = 0
NUM_QUBITS = 7


def get_normal_probabilities(mu_normal, sigma_normal, num_points):
    low = mu_normal - 3 * sigma_normal
    high = mu_normal + 3 * sigma_normal
    x = np.linspace(low, high, num_points)
    return x, scipy.stats.norm.pdf(x, loc=mu_normal, scale=sigma_normal)


def compute_var_index(alpha: float, probs: np.ndarray) -> int:
    cdf = np.cumsum(probs)
    return int(np.argmax(cdf >= alpha))


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


def run_iqae_once(iqae: IQAE, epsilon: float, alpha: float) -> dict:
    iqae_res = iqae.run(epsilon=epsilon, alpha=alpha)
    measured_payoff = iqae_res.estimation
    ci_low, ci_high = iqae_res.confidence_interval

    iterations_data = getattr(iqae_res, "iterations_data", []) or []
    shots_total = 0
    grover_calls = 0
    for it in iterations_data:
        k = getattr(it, "grover_iterations", None)
        shots = None
        if hasattr(it, "sample_results") and it.sample_results is not None:
            shots = getattr(it.sample_results, "num_shots", None)
        if hasattr(it, "num_shots") and shots is None:
            shots = it.num_shots
        shots_total += shots or 0
        if k is not None and shots is not None:
            grover_calls += k * shots

    if shots_total == 0 and hasattr(iqae_res, "sample_results"):
        shots_total = getattr(iqae_res.sample_results, "num_shots", 0)

    return {
        "estimation": measured_payoff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "shots_total": shots_total,
        "grover_calls": grover_calls,
    }


def main():
    fieldnames = [
        "var_alpha",
        "num_qubits",
        "epsilon",
        "alpha_failure",
        "a_true",
        "a_hat_mean",
        "a_hat_std",
        "abs_error_mean",
        "abs_error_std",
        "ci_low",
        "ci_high",
        "shots_total_mean",
        "grover_calls_mean",
        "runs",
    ]

    rows = []
    for num_qubits in NUM_QUBITS_LIST:
        global NUM_QUBITS
        NUM_QUBITS = num_qubits

        grid_points, probs = get_normal_probabilities(MU, SIGMA, 2**num_qubits)
        probs = probs / np.sum(probs)

        global PROBS
        PROBS = probs.tolist()

        for var_alpha in VAR_ALPHAS:
            global GLOBAL_INDEX
            GLOBAL_INDEX = compute_var_index(var_alpha, probs)
            a_true = float(np.sum(probs[:GLOBAL_INDEX]))

            iqae = create_iqae_instance(num_qubits)

            for eps in EPSILONS:
                estimates = []
                abs_errors = []
                shots_list = []
                grover_list = []
                ci_low = None
                ci_high = None

                for _ in range(RUNS):
                    res = run_iqae_once(iqae, epsilon=eps, alpha=ALPHA_FAILURE)
                    estimates.append(res["estimation"])
                    abs_errors.append(abs(res["estimation"] - a_true))
                    shots_list.append(res["shots_total"])
                    grover_list.append(res["grover_calls"])
                    ci_low = res["ci_low"]
                    ci_high = res["ci_high"]

                rows.append(
                    {
                        "var_alpha": var_alpha,
                        "num_qubits": num_qubits,
                        "epsilon": eps,
                        "alpha_failure": ALPHA_FAILURE,
                        "a_true": a_true,
                        "a_hat_mean": float(np.mean(estimates)),
                        "a_hat_std": float(np.std(estimates)) if RUNS > 1 else 0.0,
                        "abs_error_mean": float(np.mean(abs_errors)),
                        "abs_error_std": float(np.std(abs_errors)) if RUNS > 1 else 0.0,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "shots_total_mean": float(np.mean(shots_list)),
                        "grover_calls_mean": float(np.mean(grover_list)),
                        "runs": RUNS,
                    }
                )

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("✓ Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
