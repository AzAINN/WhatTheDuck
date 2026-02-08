"""
Plot IQAE epsilon sweep results produced by value_at_risk.py.

Reads: results/iqae_epsilon_sweep.csv
Outputs (in results/):
  - iqae_abs_error_vs_epsilon.png
  - iqae_shots_vs_epsilon.png
  - iqae_grover_calls_vs_epsilon.png
  - iqae_summary.png (2x2 grid)

Assumes CSV columns written by iqae_epsilon_sweep in value_at_risk.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("results/iqae_epsilon_sweep.csv")
OUT_DIR = Path("results")


def load():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run the IQAE sweep first.")
    return pd.read_csv(CSV_PATH)


def plot_abs_error(df):
    fig, ax = plt.subplots()
    ax.plot(df["epsilon"], df["abs_error_mean"], "o-", label="|a_hat - a_true|")
    ax.fill_between(
        df["epsilon"],
        df["abs_error_mean"] - df["abs_error_std"],
        df["abs_error_mean"] + df["abs_error_std"],
        color="tab:blue",
        alpha=0.2,
        label="±1 std",
    )
    ax.set_xlabel("epsilon")
    ax.set_ylabel("absolute error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "iqae_abs_error_vs_epsilon.png", dpi=200)
    return fig


def plot_shots(df):
    fig, ax = plt.subplots()
    ax.plot(df["epsilon"], df["shots_total_mean"], "s-", color="tab:orange", label="shots total")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("shots")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "iqae_shots_vs_epsilon.png", dpi=200)
    return fig


def plot_grover(df):
    fig, ax = plt.subplots()
    # Plot total samples if available (GPU + quantum), else just grover calls
    if "total_samples_mean" in df.columns and df["total_samples_mean"].iloc[0] > 0:
        ax.plot(df["epsilon"], df["total_samples_mean"], "d-", color="tab:green",
                label="total samples (GPU MC + grover calls)")
        ylabel = "total samples"
    else:
        ax.plot(df["epsilon"], df["grover_calls_mean"], "d-", color="tab:green",
                label="grover calls")
        ylabel = "grover calls"
    ax.set_xlabel("epsilon")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "iqae_grover_calls_vs_epsilon.png", dpi=200)
    return fig


def summary(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.plot(df["epsilon"], df["abs_error_mean"], "o-", label="abs error")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.6); ax.legend()
    ax.set_xlabel("epsilon"); ax.set_ylabel("abs error")

    ax = axes[0, 1]
    ax.plot(df["epsilon"], df["shots_total_mean"], "s-", color="tab:orange", label="shots")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.6); ax.legend()
    ax.set_xlabel("epsilon"); ax.set_ylabel("shots")

    ax = axes[1, 0]
    # Use total_samples if available
    if "total_samples_mean" in df.columns and df["total_samples_mean"].iloc[0] > 0:
        ax.plot(df["epsilon"], df["total_samples_mean"], "d-", color="tab:green",
                label="total samples")
        ylabel = "total samples"
    else:
        ax.plot(df["epsilon"], df["grover_calls_mean"], "d-", color="tab:green",
                label="grover calls")
        ylabel = "grover calls"
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.6); ax.legend()
    ax.set_xlabel("epsilon"); ax.set_ylabel(ylabel)

    ax = axes[1, 1]
    ax.plot(df["epsilon"], df["a_hat_mean"], "^-", color="tab:red", label="a_hat")
    ax.axhline(df["a_true"].iloc[0], color="k", linestyle="--", label="a_true")
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.6); ax.legend()
    ax.set_xlabel("epsilon"); ax.set_ylabel("amplitude")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "iqae_summary.png", dpi=200)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    plot_abs_error(df)
    plot_shots(df)
    plot_grover(df)
    summary(df)
    print("Saved plots to results/:",
          "iqae_abs_error_vs_epsilon.png,",
          "iqae_shots_vs_epsilon.png,",
          "iqae_grover_calls_vs_epsilon.png,",
          "iqae_summary.png")


if __name__ == "__main__":
    main()
