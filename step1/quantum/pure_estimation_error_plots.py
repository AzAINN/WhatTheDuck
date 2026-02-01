"""
PURE ESTIMATION ERROR PLOTS (PROBABILITY-SPACE)
=============================================
Implements:
Plot 1 (main): probability estimation error vs budget (MC vs IQAE)
Plot 2 (support): frontier error vs budget (MC vs IQAE)
Plot 3 (optional): queries to reach target error (step) + speedup

Truth is defined on the same discretized model (grid truth).
"""

import os
import glob
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# =========================
# CONFIG
# =========================

CONF_TARGET = 0.95
ALPHA_VAR = 1 - CONF_TARGET
NUM_QUBITS = 7
EPS_FLOOR = 1e-6

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_DIR = RESULTS_DIR
IQAE_CSV = os.path.join(RESULTS_DIR, "var_sweep_01.csv")
MC_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "graphs", "data", "monte_carlo_naive.csv")


def discretized_grid(mu, sigma, num_qubits):
    low = mu - 3 * sigma
    high = mu + 3 * sigma
    grid = np.linspace(low, high, 2**num_qubits)
    probs = norm.pdf(grid, loc=mu, scale=sigma)
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)
    return grid, probs, cdf


def grid_truth_var(mu, sigma, alpha, num_qubits):
    grid, probs, cdf = discretized_grid(mu, sigma, num_qubits)
    idx = int(np.argmax(cdf >= alpha))
    return float(grid[idx]), float(cdf[idx]), grid, cdf


def load_mc_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int(row["N"])
                var_pred = float(row["VaR_prediction"])
                var_theory = float(row["VaR_theoretical"])
                err = abs(var_pred - var_theory)
                rows.append((n, err))
            except (ValueError, KeyError):
                continue
    return rows


def load_iqae_rows(conf_target):
    rows = []
    with open(IQAE_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                conf = float(row["confidence_level"])
                if round(conf, 2) != round(conf_target, 2):
                    continue
                if row.get("VaR_predicted") in (None, "", "None"):
                    continue
                if row.get("VaR_theoretical") in (None, "", "None"):
                    continue
                shots = float(row.get("shots", 0) or 0)
                grover = float(row.get("grover_calls", 0) or 0)
                budget = grover if grover > 0 else shots
                if budget <= 0:
                    continue
                rows.append(
                    {
                        "budget": budget,
                        "var_predicted": float(row["VaR_predicted"]),
                        "var_theoretical": float(row["VaR_theoretical"]),
                        "mu": float(row.get("mu", "nan")),
                        "sigma": float(row.get("sigma", "nan")),
                    }
                )
            except (ValueError, KeyError):
                continue
    return rows


def iqae_var_errors(rows):
    buckets = defaultdict(list)
    raw_points = []
    for r in rows:
        err = abs(r["var_predicted"] - r["var_theoretical"])
        buckets[r["budget"]].append(err)
        raw_points.append((r["budget"], err))

    budgets = np.array(sorted(buckets.keys()))
    med = np.array([np.median(buckets[b]) for b in budgets])
    p25 = np.array([np.percentile(buckets[b], 25) for b in budgets])
    p75 = np.array([np.percentile(buckets[b], 75) for b in budgets])
    return budgets, med, p25, p75, raw_points


def frontier_curve(budgets, errors):
    order = np.argsort(budgets)
    b = np.array(budgets)[order]
    e = np.array(errors)[order]
    best = np.minimum.accumulate(e)
    return b, best


def plot_main(mc_b, mc_med, mc_p25, mc_p75, iq_b, iq_med, iq_p25, iq_p75):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mc_b, mc_med, marker="o", linewidth=2.0, label="MC (median)")
    ax.fill_between(mc_b, mc_p25, mc_p75, alpha=0.15)
    ax.plot(iq_b, iq_med, marker="o", linewidth=2.0, label="IQAE (median)")
    ax.fill_between(iq_b, iq_p25, iq_p75, alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Estimation-Only VaR Error vs Budget (conf=0.95)")
    ax.set_xlabel("Queries / Samples (log)")
    ax.set_ylabel("|VaR_est − VaR_grid*| (log)")
    ax.legend(loc="best")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pure_prob_error_vs_budget.png"), bbox_inches="tight")
    plt.close()


def plot_frontier(mc_raw, iq_raw):
    mc_b, mc_frontier = frontier_curve([b for b, _ in mc_raw], [e for _, e in mc_raw])
    iq_b, iq_frontier = frontier_curve([b for b, _ in iq_raw], [e for _, e in iq_raw])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter([b for b, _ in mc_raw], [e for _, e in mc_raw], alpha=0.2, s=14, label="MC raw")
    ax.scatter([b for b, _ in iq_raw], [e for _, e in iq_raw], alpha=0.2, s=14, label="IQAE raw")
    ax.plot(mc_b, mc_frontier, linewidth=2.2, label="MC frontier")
    ax.plot(iq_b, iq_frontier, linewidth=2.2, label="IQAE frontier")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Frontier: Best VaR Error vs Budget")
    ax.set_xlabel("Queries / Samples (log)")
    ax.set_ylabel("|VaR_est − VaR_grid*| (log)")
    ax.legend(loc="best")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "frontier_prob_error.png"), bbox_inches="tight")
    plt.close()

    return mc_b, mc_frontier, iq_b, iq_frontier


def plot_budget_to_target(mc_b, mc_frontier, iq_b, iq_frontier):
    targets = np.array([1e-2, 5e-3, 2e-3, 1e-3, 7e-4, 5e-4], dtype=float)

    def min_budget(b, e, t):
        idx = np.where(e <= t)[0]
        if len(idx) == 0:
            return np.nan
        return float(np.min(b[idx]))

    mc_req = np.array([min_budget(mc_b, mc_frontier, t) for t in targets])
    iq_req = np.array([min_budget(iq_b, iq_frontier, t) for t in targets])

    def envelope(arr):
        out = arr.copy()
        last = np.inf
        for i in range(len(out)):
            if not np.isfinite(out[i]):
                continue
            last = min(last, out[i])
            out[i] = last
        return out

    mc_req = envelope(mc_req)
    iq_req = envelope(iq_req)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(targets, mc_req, where="post", linewidth=2.0, label="MC (frontier)")
    ax.scatter(targets, mc_req, s=24)
    ax.step(targets, iq_req, where="post", linewidth=2.0, label="IQAE (frontier)")
    ax.scatter(targets, iq_req, s=24)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Budget to Reach Target VaR Error (frontier)")
    ax.set_xlabel("Target error τ (log)")
    ax.set_ylabel("Required queries (log)")
    ax.legend(loc="best")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "budget_to_target_frontier_prob.png"), bbox_inches="tight")
    plt.close()

    # Speedup
    speedup = mc_req / iq_req
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(targets, speedup, marker="o", linewidth=2.0, label="Speedup = MC / IQAE")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="No speedup")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Speedup vs Target VaR Error")
    ax.set_xlabel("Target error τ (log)")
    ax.set_ylabel("Speedup (log)")
    ax.legend(loc="best")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "speedup_frontier_prob.png"), bbox_inches="tight")
    plt.close()


def main():
    rows = load_iqae_rows(CONF_TARGET)
    if not rows:
        raise RuntimeError("No IQAE rows found for confidence 0.95 in var_sweep*.csv")

    # Use mu/sigma from IQAE rows
    mu = rows[0]["mu"] if np.isfinite(rows[0]["mu"]) else MU
    sigma = rows[0]["sigma"] if np.isfinite(rows[0]["sigma"]) else SIGMA

    # MC estimation-only VaR error from CSV
    mc_raw = load_mc_rows(MC_CSV)
    if not mc_raw:
        raise RuntimeError("No MC rows found in monte_carlo_naive.csv")
    mc_buckets = defaultdict(list)
    for b, e in mc_raw:
        mc_buckets[b].append(e)
    mc_b = np.array(sorted(mc_buckets.keys()))
    mc_med = np.array([np.median(mc_buckets[b]) for b in mc_b])
    mc_p25 = np.array([np.percentile(mc_buckets[b], 25) for b in mc_b])
    mc_p75 = np.array([np.percentile(mc_buckets[b], 75) for b in mc_b])

    # IQAE estimation-only VaR error from var_sweep_01.csv
    iq_b, iq_med, iq_p25, iq_p75, iq_raw = iqae_var_errors(rows)

    # Apply floor for log display
    mc_med = np.maximum(mc_med, EPS_FLOOR)
    mc_p25 = np.maximum(mc_p25, EPS_FLOOR)
    mc_p75 = np.maximum(mc_p75, EPS_FLOOR)
    iq_med = np.maximum(iq_med, EPS_FLOOR)
    iq_p25 = np.maximum(iq_p25, EPS_FLOOR)
    iq_p75 = np.maximum(iq_p75, EPS_FLOOR)

    plot_main(mc_b, mc_med, mc_p25, mc_p75, iq_b, iq_med, iq_p25, iq_p75)
    mc_b_f, mc_frontier, iq_b_f, iq_frontier = plot_frontier(mc_raw, iq_raw)
    plot_budget_to_target(mc_b_f, mc_frontier, iq_b_f, iq_frontier)

    print("✓ Saved plots to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
