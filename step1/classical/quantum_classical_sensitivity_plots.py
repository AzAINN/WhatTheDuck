"""
QUANTUM + CLASSICAL SENSITIVITY & SCALING PLOTS
================================================
Generates a suite of plots to compare estimation scaling and sensitivity:
1) Probability CI half-width vs budget (MC vs IQAE)
2) VaR error vs budget (MC vs IQAE-mapped)
3) Sensitivity 2x2 panel:
   - Confidence level (95% vs 99%) [MC-only unless IQAE data exists]
   - Discretization resolution (num_qubits) vs discretization error
   - Estimation precision (epsilon) vs IQAE CI half-width + budget
   - Stopping criteria proxy: CI half-width vs budget with envelope
4) Error decomposition: estimation error vs discretization error
"""

import os
import csv
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# =========================
# CONFIG (match value_at_risk.py)
# =========================

MU = 0.7
SIGMA = 0.13
ALPHA = 0.07
NUM_QUBITS = 7
Z_95 = 1.96

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_iqae_results(path: str):
    budgets = []
    a_hat = []
    a_hat_std = []
    prob_err = []
    ci_low = []
    ci_high = []
    a_true = None

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if a_true is None:
                try:
                    a_true = float(row.get("a_true", "nan"))
                except ValueError:
                    a_true = None
            a_hat_mean = float(row["a_hat_mean"])
            a_hat_sigma = float(row.get("a_hat_std", 0.0) or 0.0)
            prob_err_mean = float(row.get("abs_error_mean", 0.0) or 0.0)
            ci_l = float(row.get("ci_low", 0.0) or 0.0)
            ci_h = float(row.get("ci_high", 0.0) or 0.0)
            grover_calls = float(row.get("grover_calls_mean", 0) or 0)
            shots = float(row.get("shots_total_mean", 0) or 0)
            budget = grover_calls if grover_calls > 0 else shots
            if budget <= 0:
                continue
            budgets.append(budget)
            a_hat.append(a_hat_mean)
            a_hat_std.append(a_hat_sigma)
            prob_err.append(prob_err_mean)
            ci_low.append(ci_l)
            ci_high.append(ci_h)

    if not budgets:
        return None

    order = np.argsort(budgets)
    return {
        "budget": np.array(budgets)[order],
        "a_hat": np.array(a_hat)[order],
        "a_hat_std": np.array(a_hat_std)[order],
        "prob_err": np.array(prob_err)[order],
        "ci_low": np.array(ci_low)[order],
        "ci_high": np.array(ci_high)[order],
        "a_true": a_true,
    }


def load_classical_convergence(path: str) -> Dict[int, Dict[str, float]]:
    by_n_err: Dict[int, List[float]] = {}
    by_n_var: Dict[int, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["sample_size"])
            err = float(row["abs_error"])
            var_est = float(row.get("var_estimate", 0.0))
            by_n_err.setdefault(n, []).append(err)
            by_n_var.setdefault(n, []).append(var_est)

    stats: Dict[int, Dict[str, float]] = {}
    for n, errs in by_n_err.items():
        arr = np.array(errs, dtype=float)
        var_arr = np.array(by_n_var.get(n, []), dtype=float)
        stats[n] = {
            "mean_err": float(np.mean(arr)),
            "p10_err": float(np.percentile(arr, 10)),
            "p90_err": float(np.percentile(arr, 90)),
            "var_mean": float(np.mean(var_arr)) if var_arr.size else 0.0,
        }
    return stats


def discretized_var(mu: float, sigma: float, alpha: float, num_qubits: int):
    low = mu - 3 * sigma
    high = mu + 3 * sigma
    grid = np.linspace(low, high, 2**num_qubits)
    probs = norm.pdf(grid, loc=mu, scale=sigma)
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)
    idx = int(np.argmax(cdf >= alpha))
    return float(grid[idx])


def probability_ci_halfwidth_mc(p_true: float, n: np.ndarray, z: float = Z_95) -> np.ndarray:
    return z * np.sqrt(p_true * (1 - p_true) / n)


def monotone_envelope(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    last = np.inf
    for i in range(len(out)):
        if not np.isfinite(out[i]):
            continue
        last = min(last, out[i])
        out[i] = last
    return out


def plot_prob_ci_halfwidth_vs_budget(iqae, output_path: str):
    p_true = float(iqae["a_true"])
    mc_n = np.array(sorted({100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200}))
    mc_half = probability_ci_halfwidth_mc(p_true, mc_n)
    iqae_half = 0.5 * (iqae["ci_high"] - iqae["ci_low"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mc_n, mc_half, marker="o", linewidth=2.0, label="Classical MC (CI half-width)")
    ax.plot(iqae["budget"], iqae_half, marker="o", linewidth=2.0, label="IQAE (CI half-width)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Probability CI Half-Width vs Budget")
    ax.set_xlabel("Queries / Samples (log)")
    ax.set_ylabel("CI Half-Width (log)")
    ax.legend(loc="best")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_discretization_error_vs_qubits(output_path: str):
    qubits_list = list(range(4, 11))
    grid_sizes = [2**q for q in qubits_list]
    var_true = float(norm.ppf(ALPHA, loc=MU, scale=SIGMA))
    disc_err = [abs(discretized_var(MU, SIGMA, ALPHA, q) - var_true) for q in qubits_list]

    # Jitter mu/sigma slightly to create an envelope
    rng = np.random.default_rng(1234)
    jitter_mu = MU * 0.01
    jitter_sigma = SIGMA * 0.05
    disc_min = []
    disc_max = []
    disc_med = []
    for q in qubits_list:
        errors = []
        for _ in range(40):
            mu_j = MU + rng.normal(0.0, jitter_mu)
            sigma_j = max(1e-6, SIGMA + rng.normal(0.0, jitter_sigma))
            v_disc = discretized_var(mu_j, sigma_j, ALPHA, q)
            v_cont = float(norm.ppf(ALPHA, loc=mu_j, scale=sigma_j))
            errors.append(abs(v_disc - v_cont))
        errors = np.array(errors)
        disc_min.append(float(np.min(errors)))
        disc_max.append(float(np.max(errors)))
        disc_med.append(float(np.median(errors)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid_sizes, disc_med, marker="o", linewidth=2.0, label="Median")
    ax.fill_between(grid_sizes, disc_min, disc_max, alpha=0.15, label="Min/Max envelope")
    ax.set_title("Discretization Error vs Grid Resolution")
    ax.set_xlabel("Grid size (number of bins)")
    ax.set_ylabel("|VaR_continuous − VaR_discretized|")
    ax.legend(loc="best")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_var_error_vs_budget(iqae, mc_stats, output_path: str):
    mc_n = np.array(sorted(mc_stats.keys()))
    mc_err = np.array([mc_stats[n]["mean_err"] for n in mc_n])

    # Map IQAE probability estimate to discretized VaR grid for comparable VaR error
    var_discrete = discretized_var(MU, SIGMA, ALPHA, NUM_QUBITS)
    low = MU - 3 * SIGMA
    high = MU + 3 * SIGMA
    grid = np.linspace(low, high, 2**NUM_QUBITS)
    probs = norm.pdf(grid, loc=MU, scale=SIGMA)
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)

    def a_hat_to_var(a_hat: float) -> float:
        idx = int(np.argmax(cdf >= a_hat))
        return float(grid[idx])

    iqae_var = np.array([a_hat_to_var(a) for a in iqae["a_hat"]])
    iqae_err = np.abs(iqae_var - var_discrete)
    iqae_err = np.maximum(iqae_err, 1e-6)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mc_n, mc_err, marker="o", linewidth=2.0, label="Classical MC (VaR error)")
    ax.plot(iqae["budget"], iqae_err, marker="o", linewidth=2.0, label="IQAE (VaR error)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("VaR Error vs Budget")
    ax.set_xlabel("Queries / Samples (log)")
    ax.set_ylabel("VaR Error (log)")
    ax.legend(loc="best")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_sensitivity_panel(iqae, output_path: str):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs = axs.flatten()

    # Panel 1: Confidence level (MC-only)
    p95 = 1 - 0.95
    p99 = 1 - 0.99
    n = np.array(sorted({200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200}))
    hw95 = probability_ci_halfwidth_mc(p95, n)
    hw99 = probability_ci_halfwidth_mc(p99, n)
    axs[0].plot(n, hw95, marker="o", label="MC CI half-width (95%)")
    axs[0].plot(n, hw99, marker="o", label="MC CI half-width (99%)")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_title("Confidence Level Sensitivity")
    axs[0].set_xlabel("Samples (log)")
    axs[0].set_ylabel("CI Half-Width (log)")
    axs[0].legend(loc="best")

    # Panel 2: Discretization resolution
    qubits_list = [4, 5, 6, 7, 8, 9, 10]
    var_true = float(norm.ppf(ALPHA, loc=MU, scale=SIGMA))
    disc_err = [abs(discretized_var(MU, SIGMA, ALPHA, q) - var_true) for q in qubits_list]
    axs[1].plot(qubits_list, disc_err, marker="o")
    axs[1].set_title("Discretization Resolution")
    axs[1].set_xlabel("Num Qubits")
    axs[1].set_ylabel("Discretization Error")

    # Panel 3: Estimation precision (IQAE epsilon sweep proxy)
    iqae_half = 0.5 * (iqae["ci_high"] - iqae["ci_low"])
    axs[2].plot(iqae["budget"], iqae_half, marker="o")
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_title("Estimation Precision (IQAE)")
    axs[2].set_xlabel("Budget (log)")
    axs[2].set_ylabel("CI Half-Width (log)")

    # Panel 4: Stopping criteria proxy (envelope)
    env = monotone_envelope(iqae_half)
    axs[3].plot(iqae["budget"], iqae_half, marker="o", label="Observed")
    axs[3].plot(iqae["budget"], env, linewidth=2.0, label="Monotone envelope")
    axs[3].set_xscale("log")
    axs[3].set_yscale("log")
    axs[3].set_title("Stopping Criteria Proxy")
    axs[3].set_xlabel("Budget (log)")
    axs[3].set_ylabel("CI Half-Width (log)")
    axs[3].legend(loc="best")

    for ax in axs:
        ax.grid(True, which="major", linestyle="-", alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_error_decomposition(iqae, mc_stats, output_path: str):
    # Discretization error (grid vs analytic VaR)
    var_true = float(norm.ppf(ALPHA, loc=MU, scale=SIGMA))
    var_disc = discretized_var(MU, SIGMA, ALPHA, NUM_QUBITS)
    disc_err = abs(var_disc - var_true)

    # Estimation error at a representative budget
    iqae_half = 0.5 * (iqae["ci_high"] - iqae["ci_low"])
    iqae_rep = float(np.median(iqae_half))
    mc_n = np.array(sorted(mc_stats.keys()))
    mc_err = np.array([mc_stats[n]["mean_err"] for n in mc_n])
    mc_rep = float(np.median(mc_err))

    labels = ["Discretization", "MC Estimation", "IQAE Estimation"]
    values = [disc_err, mc_rep, iqae_rep]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title("Error Decomposition")
    ax.set_ylabel("Error Magnitude")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    iqae_path = os.path.join(RESULTS_DIR, "iqae_epsilon_sweep.csv")
    classical_path = os.path.join(RESULTS_DIR, "classical_var_convergence.csv")

    if not os.path.exists(iqae_path):
        raise FileNotFoundError(f"Missing IQAE results: {iqae_path}")
    if not os.path.exists(classical_path):
        raise FileNotFoundError(f"Missing classical convergence: {classical_path}")

    iqae = load_iqae_results(iqae_path)
    if iqae is None or iqae["a_true"] is None or not np.isfinite(iqae["a_true"]):
        raise ValueError("Missing a_true in IQAE results; re-run value_at_risk.py.")

    mc_stats = load_classical_convergence(classical_path)

    plot_discretization_error_vs_qubits(
        os.path.join(OUTPUT_DIR, "13_discretization_error_vs_qubits.png"),
    )
    plot_prob_ci_halfwidth_vs_budget(
        iqae,
        os.path.join(OUTPUT_DIR, "14_prob_ci_halfwidth_vs_budget.png"),
    )
    plot_var_error_vs_budget(
        iqae,
        mc_stats,
        os.path.join(OUTPUT_DIR, "15_var_error_vs_budget.png"),
    )

    print("✓ Saved plots to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
