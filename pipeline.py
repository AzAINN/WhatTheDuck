"""
End-to-End Pipeline: Portfolio Optimization -> VaR (GPU MC + Quantum IQAE)

Pipeline steps:
  1. Define a realistic 20-stock universe with sector-based factor model
  2. Run mean-variance portfolio optimization (cardinality-constrained)
  3. GPU Monte Carlo per-asset + portfolio loss distributions
  4. Publication-quality per-asset and portfolio plots
  5. Structured text + CSV output of all metrics
  6. (Optional) Feed portfolio losses into quantum IQAE for VaR estimation
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.stats
import scipy.optimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Publication-quality plot settings (matching enhanced_monte_carlo_graphs.py)
# ---------------------------------------------------------------------------
COLOR_PRIMARY = "#1e3a8a"
COLOR_SECONDARY = "#3b82f6"
COLOR_ACCENT = "#f59e0b"
COLOR_DANGER = "#dc2626"
COLOR_GRID = "#e5e7eb"
COLOR_TEXT = "#1f2937"
COLOR_BOUND_UPPER = "#6366f1"
COLOR_BOUND_LOWER = "#8b5cf6"

SECTOR_COLORS = {
    "Tech": "#3b82f6",
    "Finance": "#10b981",
    "Healthcare": "#f59e0b",
    "Energy": "#ef4444",
    "Consumer": "#8b5cf6",
    "Industrial": "#f97316",
    "Utilities": "#06b6d4",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})


# ---------------------------------------------------------------------------
# Stock universe
# ---------------------------------------------------------------------------
STOCK_UNIVERSE = [
    # (ticker, sector, expected_annual_return, annual_volatility)
    ("AAPL",  "Tech",        0.12, 0.28),
    ("MSFT",  "Tech",        0.11, 0.25),
    ("GOOGL", "Tech",        0.10, 0.30),
    ("AMZN",  "Tech",        0.13, 0.35),
    ("NVDA",  "Tech",        0.18, 0.45),
    ("JPM",   "Finance",     0.09, 0.22),
    ("BAC",   "Finance",     0.07, 0.26),
    ("GS",    "Finance",     0.10, 0.28),
    ("JNJ",   "Healthcare",  0.06, 0.18),
    ("PFE",   "Healthcare",  0.05, 0.24),
    ("UNH",   "Healthcare",  0.08, 0.22),
    ("XOM",   "Energy",      0.08, 0.30),
    ("CVX",   "Energy",      0.07, 0.28),
    ("PG",    "Consumer",    0.05, 0.16),
    ("KO",    "Consumer",    0.04, 0.15),
    ("WMT",   "Consumer",    0.06, 0.18),
    ("CAT",   "Industrial",  0.09, 0.26),
    ("BA",    "Industrial",  0.08, 0.35),
    ("NEE",   "Utilities",   0.06, 0.20),
    ("DUK",   "Utilities",   0.04, 0.16),
]

# Map sector names to integer IDs for factor loading assignment
_SECTOR_NAMES = sorted(set(s for _, s, _, _ in STOCK_UNIVERSE))
_SECTOR_ID = {name: i for i, name in enumerate(_SECTOR_NAMES)}


# ===================================================================
# Step 1: Build stock universe & generate synthetic returns
# ===================================================================
def build_stock_universe(n_days=252, df=4, seed=42):
    """
    Generate synthetic daily returns for STOCK_UNIVERSE using a
    sector-based factor model with Student-t innovations.

    Factor structure (3 factors):
      - Factor 0: market (loads on all assets)
      - Factor 1: sector rotation
      - Factor 2: volatility regime

    Correlations: within-sector ~0.6, cross-sector ~0.3.

    Returns:
        tickers:  list[str]
        sectors:  list[str]
        returns:  (n_days, M) array of daily returns
        mu:       (M,) annualized expected returns (from universe definition)
        cov:      (M, M) sample covariance of daily returns
    """
    rng = np.random.default_rng(seed)
    M = len(STOCK_UNIVERSE)
    n_factors = 3

    tickers = [t for t, _, _, _ in STOCK_UNIVERSE]
    sectors = [s for _, s, _, _ in STOCK_UNIVERSE]
    annual_mu = np.array([r for _, _, r, _ in STOCK_UNIVERSE])
    annual_vol = np.array([v for _, _, _, v in STOCK_UNIVERSE])

    # Daily parameters
    daily_mu = annual_mu / 252
    daily_vol = annual_vol / np.sqrt(252)

    # Student-t(df) has variance df/(df-2) for df>2, so unit-variance
    # t-samples need scaling by 1/sqrt(df/(df-2)).
    t_var = df / (df - 2) if df > 2 else 2.0
    t_scale = 1.0 / np.sqrt(t_var)

    # Build factor loadings B: (M, 3)
    # Loadings are calibrated so that Var(r_i) ≈ daily_vol[i]^2.
    # Systematic fraction ~60%, idiosyncratic ~40%.
    sys_frac = 0.6
    idio_frac = 1.0 - sys_frac

    B = np.zeros((M, n_factors))
    for i, (_, sector, _, _) in enumerate(STOCK_UNIVERSE):
        sid = _SECTOR_ID[sector]
        sys_vol = daily_vol[i] * np.sqrt(sys_frac)
        # Factor 0: market — all assets
        B[i, 0] = 0.8 * sys_vol
        # Factor 1: sector rotation — sign alternates by sector
        B[i, 1] = (0.4 if sid % 2 == 0 else -0.4) * sys_vol
        # Factor 2: volatility regime
        B[i, 2] = 0.2 * sys_vol

    # Sample factors from Student-t(df) for fat tails, scaled to unit variance
    chi2 = rng.chisquare(df, size=(n_days, 1))
    G = rng.standard_normal((n_days, n_factors))
    factors = G * np.sqrt(df / chi2) * t_scale  # (n_days, n_factors)

    # Idiosyncratic noise (also Student-t, unit-variance-scaled)
    chi2_idio = rng.chisquare(df, size=(n_days, 1))
    noise = rng.standard_normal((n_days, M)) * np.sqrt(df / chi2_idio) * t_scale
    idio_scale = daily_vol[np.newaxis, :] * np.sqrt(idio_frac)

    # Returns = systematic + idiosyncratic + drift
    returns = factors @ B.T + idio_scale * noise + daily_mu[np.newaxis, :]

    # Sample covariance from daily returns
    cov_daily = np.cov(returns, rowvar=False)
    # Annualize (cov_annual = cov_daily * 252) so mu and cov are on the
    # same scale for the optimizer
    cov_annual = cov_daily * 252

    return tickers, sectors, returns, annual_mu, cov_annual, cov_daily


# ===================================================================
# Step 2: Portfolio optimization
# ===================================================================
def optimize_portfolio(mu, cov, max_positions=8, lambda_risk=5.0, cap=0.15):
    """
    Run mean-variance optimization with cardinality constraint.

    Tries portfolio_qihd QIHD solver first; falls back to scipy.optimize
    sequential least squares programming (SLSQP) if unavailable.

    Returns:
        weights:  (M,) optimal weights (zero for unselected assets)
        selected: list[int] indices of selected assets
    """
    M = len(mu)

    # --- Try QIHD solver ---
    try:
        from portfolio_qihd import PortfolioSpec, solve_portfolio, caps_from_constant

        upper = caps_from_constant(cap, M)
        spec = PortfolioSpec(
            mu=mu, cov=cov, upper=upper,
            max_positions=max_positions, lambda_risk=lambda_risk,
        )
        result = solve_portfolio(spec)
        w = np.clip(result.weights, 0, None)
        w /= w.sum() if w.sum() > 0 else 1.0
        selected = list(np.where(w > 1e-6)[0])
        print(f"  Solver: QIHD (OpenPhiSolve)")
        return w, selected

    except ImportError:
        pass

    # --- Scipy fallback: relaxed QP (no binary variables) ---
    # Solve: min  lambda_risk * w^T cov w  -  mu^T w
    # s.t.   sum(w) = 1,  0 <= w_i <= cap
    # Then zero out smallest weights to enforce cardinality.
    print(f"  Solver: scipy SLSQP fallback")

    def objective(w):
        return lambda_risk * w @ cov @ w - mu @ w

    def grad(w):
        return 2 * lambda_risk * cov @ w - mu

    bounds = [(0.0, cap)] * M
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                    "jac": lambda w: np.ones(M)}]

    w0 = np.full(M, 1.0 / M)
    res = scipy.optimize.minimize(
        objective, w0, jac=grad, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, None)

    # Enforce cardinality: keep top-K by weight
    if np.count_nonzero(w > 1e-6) > max_positions:
        ranked = np.argsort(w)[::-1]
        keep = set(ranked[:max_positions])
        for i in range(M):
            if i not in keep:
                w[i] = 0.0
        # Re-normalize
        if w.sum() > 0:
            w /= w.sum()
        # Re-apply cap
        w = np.minimum(w, cap)
        if w.sum() > 0:
            w /= w.sum()

    selected = list(np.where(w > 1e-6)[0])
    return w, selected


# ===================================================================
# Step 3: GPU Monte Carlo — per-asset + portfolio
# ===================================================================
def gpu_monte_carlo_var(cov, weights, tickers, sectors,
                        n_scenarios=100_000, df=4, alpha=0.05, seed=42):
    """
    GPU Monte Carlo sampling for SELECTED assets.

    Samples from multivariate Student-t(df, 0, cov_selected) using
    JAX if available, numpy otherwise.

    Returns dict:
        portfolio_losses:   (N,) portfolio loss scenarios
        asset_returns:      (N, M_selected) per-asset return scenarios
        portfolio_var:      float — portfolio VaR at alpha
        portfolio_cvar:     float — portfolio CVaR at alpha
        per_asset_metrics:  list[dict] per selected asset
        device:             str
    """
    selected = np.where(weights > 1e-6)[0]
    M_sel = len(selected)
    w_sel = weights[selected]
    cov_sel = cov[np.ix_(selected, selected)]

    device = "numpy-cpu"
    asset_returns = None

    try:
        import jax
        import jax.numpy as jnp

        backend = jax.default_backend()
        device = backend if backend != "cpu" else "jax-cpu"

        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)

        # Cholesky of selected covariance
        L = np.linalg.cholesky(cov_sel + 1e-8 * np.eye(M_sel))
        L_j = jnp.array(L, dtype=jnp.float32)

        # Gaussian samples
        Z = jax.random.normal(k1, (n_scenarios, M_sel))

        # Student-t mixing: chi2 via gamma
        chi2 = 2.0 * jax.random.gamma(k2, df / 2.0, shape=(n_scenarios, 1))
        T = Z * jnp.sqrt(df / chi2)

        # Correlated returns
        R = T @ L_j.T  # (N, M_sel)
        asset_returns = np.asarray(R, dtype=np.float64)

    except ImportError:
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(cov_sel + 1e-8 * np.eye(M_sel))
        Z = rng.standard_normal((n_scenarios, M_sel))
        U = rng.chisquare(df, size=(n_scenarios, 1))
        T = Z * np.sqrt(df / U)
        asset_returns = T @ L.T

    # Portfolio P&L and losses
    portfolio_pnl = asset_returns @ w_sel
    portfolio_losses = -portfolio_pnl

    # Portfolio VaR / CVaR
    var_threshold = np.percentile(portfolio_losses, (1 - alpha) * 100)
    portfolio_var = float(var_threshold)
    tail = portfolio_losses[portfolio_losses >= var_threshold]
    portfolio_cvar = float(tail.mean()) if len(tail) > 0 else portfolio_var

    # Per-asset metrics
    per_asset_metrics = []
    for j, idx in enumerate(selected):
        r = asset_returns[:, j]
        losses_j = -r
        var_j = float(np.percentile(losses_j, (1 - alpha) * 100))
        tail_j = losses_j[losses_j >= var_j]
        cvar_j = float(tail_j.mean()) if len(tail_j) > 0 else var_j
        per_asset_metrics.append({
            "index": int(idx),
            "ticker": tickers[idx],
            "sector": sectors[idx],
            "weight": float(w_sel[j]),
            "var": var_j,
            "cvar": cvar_j,
            "mean": float(r.mean()),
            "std": float(r.std()),
        })

    return {
        "portfolio_losses": portfolio_losses,
        "asset_returns": asset_returns,
        "selected_indices": selected,
        "portfolio_var": portfolio_var,
        "portfolio_cvar": portfolio_cvar,
        "per_asset_metrics": per_asset_metrics,
        "device": device,
        "n_scenarios": n_scenarios,
    }


# ===================================================================
# Step 4: Plotting
# ===================================================================
def _style_ax(ax):
    """Apply publication-quality styling to an axes."""
    ax.set_facecolor("#fafafa")
    ax.grid(True, which="major", linestyle="-", alpha=0.3, color=COLOR_GRID, zorder=1)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15, color=COLOR_GRID, zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_GRID)
        spine.set_linewidth(1.5)


def _style_legend(ax):
    """Apply publication-quality styling to a legend."""
    leg = ax.legend(loc="best", frameon=True, fancybox=True,
                    shadow=True, framealpha=0.95, edgecolor=COLOR_GRID)
    if leg:
        leg.get_frame().set_facecolor("white")


def plot_asset_report(ticker, sector, returns_1d, var_val, cvar_val,
                      weight, output_path):
    """
    Single-asset distribution plot: histogram + VaR/CVaR lines.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _style_ax(ax)

    color = SECTOR_COLORS.get(sector, COLOR_SECONDARY)

    ax.hist(returns_1d, bins=150, density=True, alpha=0.6,
            color=color, edgecolor="white", linewidth=0.3,
            label=f"{ticker} daily returns", zorder=3)

    ax.axvline(x=-var_val, color=COLOR_DANGER, linestyle="--", linewidth=2.5,
               alpha=0.9, label=f"VaR(5%) = {var_val:.4f}", zorder=5)
    ax.axvline(x=-cvar_val, color=COLOR_ACCENT, linestyle="-.", linewidth=2.5,
               alpha=0.9, label=f"CVaR(5%) = {cvar_val:.4f}", zorder=5)

    ax.set_xlabel("Daily Return", fontweight="500", color=COLOR_TEXT)
    ax.set_ylabel("Density", fontweight="500", color=COLOR_TEXT)
    ax.set_title(f"{ticker} ({sector})  —  Weight: {weight:.1%}",
                 fontweight="600", color=COLOR_TEXT, pad=15)
    _style_legend(ax)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_portfolio_report(mc_results, weights, tickers, sectors, output_dir):
    """
    Generate all report plots:
      1. Per-asset distribution panels (grid)
      2. Portfolio loss distribution with VaR/CVaR
      3. Asset allocation pie chart
      4. Risk contribution bar chart
      5. Combined summary figure
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = mc_results["per_asset_metrics"]
    selected = mc_results["selected_indices"]
    n_sel = len(selected)

    # --- 1. Per-asset panels ---
    for j, m in enumerate(metrics):
        r = mc_results["asset_returns"][:, j]
        plot_asset_report(
            m["ticker"], m["sector"], r, m["var"], m["cvar"], m["weight"],
            os.path.join(output_dir, f"asset_{m['ticker']}.png"),
        )

    # --- 2. Portfolio loss distribution ---
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _style_ax(ax)

    losses = mc_results["portfolio_losses"]
    ax.hist(losses, bins=200, density=True, alpha=0.6,
            color=COLOR_PRIMARY, edgecolor="white", linewidth=0.3,
            label="Portfolio loss distribution", zorder=3)
    ax.axvline(x=mc_results["portfolio_var"], color=COLOR_DANGER,
               linestyle="--", linewidth=2.5, alpha=0.9,
               label=f"VaR(5%) = {mc_results['portfolio_var']:.4f}", zorder=5)
    ax.axvline(x=mc_results["portfolio_cvar"], color=COLOR_ACCENT,
               linestyle="-.", linewidth=2.5, alpha=0.9,
               label=f"CVaR(5%) = {mc_results['portfolio_cvar']:.4f}", zorder=5)
    ax.set_xlabel("Portfolio Loss", fontweight="500", color=COLOR_TEXT)
    ax.set_ylabel("Density", fontweight="500", color=COLOR_TEXT)
    ax.set_title("Portfolio Loss Distribution (Monte Carlo)",
                 fontweight="600", color=COLOR_TEXT, pad=15)
    _style_legend(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "portfolio_distribution.png"),
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    # --- 3. Allocation pie chart ---
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    sel_tickers = [tickers[i] for i in selected]
    sel_weights = weights[selected]
    sel_sectors = [sectors[i] for i in selected]
    colors = [SECTOR_COLORS.get(s, COLOR_SECONDARY) for s in sel_sectors]

    wedges, texts, autotexts = ax.pie(
        sel_weights, labels=sel_tickers, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.80,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("600")
    ax.set_title("Portfolio Allocation", fontweight="600",
                 color=COLOR_TEXT, pad=20, fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "allocation.png"),
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    # --- 4. Risk contribution bar chart ---
    # Marginal risk contribution: w_i * (Sigma @ w)_i / portfolio_vol
    cov_sel = np.cov(mc_results["asset_returns"], rowvar=False)
    w_sel = sel_weights
    sigma_w = cov_sel @ w_sel
    port_var = w_sel @ sigma_w
    port_vol = np.sqrt(port_var) if port_var > 0 else 1e-10
    risk_contrib = w_sel * sigma_w / port_vol
    risk_pct = risk_contrib / risk_contrib.sum() * 100 if risk_contrib.sum() > 0 else risk_contrib

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _style_ax(ax)
    bars = ax.bar(sel_tickers, risk_pct, color=colors, edgecolor="white",
                  linewidth=1.5, zorder=3, alpha=0.85)
    ax.set_xlabel("Asset", fontweight="500", color=COLOR_TEXT)
    ax.set_ylabel("Risk Contribution (%)", fontweight="500", color=COLOR_TEXT)
    ax.set_title("Marginal Risk Contributions",
                 fontweight="600", color=COLOR_TEXT, pad=15)
    for bar, pct in zip(bars, risk_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9,
                fontweight="500", color=COLOR_TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "risk_contributions.png"),
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    # --- 5. Combined summary figure ---
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # (A) Portfolio loss distribution
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1)
    ax1.hist(losses, bins=150, density=True, alpha=0.6,
             color=COLOR_PRIMARY, edgecolor="white", linewidth=0.3, zorder=3)
    ax1.axvline(x=mc_results["portfolio_var"], color=COLOR_DANGER,
                linestyle="--", linewidth=2, zorder=5)
    ax1.axvline(x=mc_results["portfolio_cvar"], color=COLOR_ACCENT,
                linestyle="-.", linewidth=2, zorder=5)
    ax1.set_xlabel("Loss", fontweight="500", color=COLOR_TEXT)
    ax1.set_ylabel("Density", fontweight="500", color=COLOR_TEXT)
    ax1.set_title("(A) Portfolio Loss Distribution",
                  fontweight="600", color=COLOR_TEXT, loc="left", pad=15)

    # (B) Allocation pie
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie(sel_weights, labels=sel_tickers, autopct="%1.0f%%",
            colors=colors, startangle=90, pctdistance=0.80,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops={"fontsize": 9})
    ax2.set_title("(B) Asset Allocation",
                  fontweight="600", color=COLOR_TEXT, loc="left", pad=15)

    # (C) Risk contributions
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3)
    ax3.bar(sel_tickers, risk_pct, color=colors, edgecolor="white",
            linewidth=1.5, zorder=3, alpha=0.85)
    ax3.set_xlabel("Asset", fontweight="500", color=COLOR_TEXT)
    ax3.set_ylabel("Risk %", fontweight="500", color=COLOR_TEXT)
    ax3.set_title("(C) Risk Contributions",
                  fontweight="600", color=COLOR_TEXT, loc="left", pad=15)
    ax3.tick_params(axis="x", rotation=45)

    # (D) Per-asset VaR comparison
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4)
    var_vals = [m["var"] for m in metrics]
    cvar_vals = [m["cvar"] for m in metrics]
    tick_labels = [m["ticker"] for m in metrics]
    x_pos = np.arange(n_sel)
    width = 0.35
    ax4.bar(x_pos - width / 2, var_vals, width, color=COLOR_DANGER,
            alpha=0.7, label="VaR(5%)", edgecolor="white", linewidth=1.5, zorder=3)
    ax4.bar(x_pos + width / 2, cvar_vals, width, color=COLOR_ACCENT,
            alpha=0.7, label="CVaR(5%)", edgecolor="white", linewidth=1.5, zorder=3)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax4.set_xlabel("Asset", fontweight="500", color=COLOR_TEXT)
    ax4.set_ylabel("Loss", fontweight="500", color=COLOR_TEXT)
    ax4.set_title("(D) Per-Asset VaR / CVaR",
                  fontweight="600", color=COLOR_TEXT, loc="left", pad=15)
    _style_legend(ax4)

    fig.suptitle("Portfolio Risk Report", fontweight="700", fontsize=18,
                 color=COLOR_TEXT, y=0.98)
    fig.savefig(os.path.join(output_dir, "summary.png"),
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"  Plots saved to {output_dir}/")


# ===================================================================
# Step 5: Structured output
# ===================================================================
def print_report(mc_results, weights, tickers, sectors, output_dir):
    """Print formatted table and save CSV."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = mc_results["per_asset_metrics"]

    print()
    print("=" * 85)
    print("PORTFOLIO OPTIMIZATION + RISK RESULTS")
    print("=" * 85)
    header = (f"{'Ticker':<8s}{'Sector':<14s}{'Weight':>8s}"
              f"{'VaR(5%)':>10s}{'CVaR(5%)':>10s}"
              f"{'Mean':>10s}{'Std':>10s}")
    print(header)
    print("-" * 85)

    rows = []
    for m in metrics:
        line = (f"{m['ticker']:<8s}{m['sector']:<14s}{m['weight']:>8.4f}"
                f"{m['var']:>10.4f}{m['cvar']:>10.4f}"
                f"{m['mean']:>10.6f}{m['std']:>10.6f}")
        print(line)
        rows.append(m)

    print("-" * 85)
    pvar = mc_results["portfolio_var"]
    pcvar = mc_results["portfolio_cvar"]
    pnl = mc_results["portfolio_losses"]
    line = (f"{'PORTFOLIO':<8s}{'':<14s}{'1.0000':>8s}"
            f"{pvar:>10.4f}{pcvar:>10.4f}"
            f"{-pnl.mean():>10.6f}{pnl.std():>10.6f}")
    print(line)
    print("=" * 85)

    # Save CSV
    csv_path = os.path.join(output_dir, "report.csv")
    fieldnames = ["ticker", "sector", "weight", "var", "cvar", "mean", "std"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            writer.writerow({k: m[k] for k in fieldnames})
        writer.writerow({
            "ticker": "PORTFOLIO", "sector": "",
            "weight": 1.0, "var": pvar, "cvar": pcvar,
            "mean": float(-pnl.mean()), "std": float(pnl.std()),
        })
    print(f"\n  Report saved to {csv_path}")


# ===================================================================
# Step 6: Quantum IQAE bridge
# ===================================================================
def run_quantum_var(portfolio_losses, num_qubits=7, epsilon=0.05,
                    alpha=0.05, mc_confidence=0.99):
    """
    Feed portfolio losses -> PMF -> warm start -> IQAE bisection.

    Returns dict with quantum VaR estimate and oracle query count.
    """
    try:
        from quantum_VaR.VaR_Quantum_highD import (
            build_pmf_from_samples,
            mc_warm_start_from_losses,
            classical_var,
            quantum_value_at_risk,
        )
    except ImportError as e:
        print(f"\n  Quantum pipeline unavailable: {e}")
        print("  (requires classiq SDK)")
        return None

    print("\n--- Quantum IQAE Pipeline ---")

    # Discretize portfolio losses
    grid, pmf, lo, hi = build_pmf_from_samples(portfolio_losses, num_qubits)
    print(f"  Discretized onto {2**num_qubits} bins over [{lo:.4f}, {hi:.4f}]")

    # Classical reference
    ref_idx, ref_val = classical_var(grid, pmf, alpha)
    print(f"  Classical CDF VaR: {ref_val:.6f}  (index {ref_idx})")

    # Warm-start bracket from MC samples
    ws = mc_warm_start_from_losses(portfolio_losses, grid, alpha,
                                   confidence=mc_confidence)
    bracket = (ws["lo"], ws["hi"])
    print(f"  Warm-start bracket: [{bracket[0]}, {bracket[1]}]  "
          f"(width={ws['bracket_width']})")

    # Quantum VaR
    tolerance = alpha / 10
    q_idx, q_var, q_est, q_ci, q_queries, q_steps = quantum_value_at_risk(
        grid, pmf, alpha,
        tolerance=tolerance,
        epsilon=epsilon,
        bracket=bracket,
    )

    print(f"\n  Quantum VaR: {q_var:.6f}  (index {q_idx})")
    if q_est is not None:
        print(f"  Tail P estimate: {q_est:.6f}")
        print(f"  IQAE CI: [{q_ci[0]:.6f}, {q_ci[1]:.6f}]")
    print(f"  Bisection steps: {q_steps}")
    print(f"  Oracle queries: {q_queries}")
    print(f"  Classical ref: {ref_val:.6f}  |  Delta: {abs(q_var - ref_val):.6f}")

    return {
        "quantum_var": q_var,
        "quantum_var_index": q_idx,
        "classical_var": ref_val,
        "classical_var_index": ref_idx,
        "tail_estimate": q_est,
        "ci": q_ci,
        "oracle_queries": q_queries,
        "bisection_steps": q_steps,
        "bracket": bracket,
        "grid_range": (lo, hi),
        "num_bins": 2 ** num_qubits,
    }


# ===================================================================
# CLI
# ===================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="End-to-end portfolio optimization + VaR pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Full MC pipeline (no quantum)
  python pipeline.py --no-quantum

  # Specific portfolio configuration
  python pipeline.py --max-positions 5 --lambda-risk 10 --no-quantum

  # Full pipeline with quantum IQAE
  python pipeline.py --quantum --epsilon 0.05

  # Large MC run
  python pipeline.py --n-scenarios 500000 --no-quantum
""",
    )
    # Portfolio
    p.add_argument("--max-positions", type=int, default=8,
                   help="cardinality constraint K (default: 8)")
    p.add_argument("--lambda-risk", type=float, default=5.0,
                   help="risk aversion parameter (default: 5.0)")
    p.add_argument("--cap", type=float, default=0.15,
                   help="max weight per asset (default: 0.15)")
    # Monte Carlo
    p.add_argument("--n-scenarios", type=int, default=100_000,
                   help="MC simulation scenarios (default: 100000)")
    p.add_argument("--df", type=int, default=4,
                   help="Student-t degrees of freedom (default: 4)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="VaR confidence level (default: 0.05)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed (default: 42)")
    # Quantum
    p.add_argument("--no-quantum", action="store_true",
                   help="skip IQAE, GPU MC only")
    p.add_argument("--quantum", action="store_true",
                   help="run IQAE after MC")
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="IQAE precision (default: 0.05)")
    # Output
    p.add_argument("--output-dir", type=str, default="results/pipeline",
                   help="output directory (default: results/pipeline)")
    return p


# ===================================================================
# Main
# ===================================================================
def main(args=None):
    if args is None:
        args = build_parser().parse_args()

    output_dir = args.output_dir

    # --- Step 1: Universe & returns ---
    print("=" * 65)
    print("STEP 1: Building stock universe & synthetic returns")
    print("=" * 65)
    tickers, sectors, returns, mu, cov = build_stock_universe(
        n_days=252, df=args.df, seed=args.seed,
    )
    M = len(tickers)
    print(f"  Universe: {M} stocks across {len(set(sectors))} sectors")
    print(f"  Return matrix: {returns.shape}")
    print(f"  Top expected returns:")
    order = np.argsort(mu)[::-1]
    for i in order[:5]:
        print(f"    {tickers[i]:>5s} ({sectors[i]:<12s})  "
              f"mu={mu[i]:.2%}  vol={np.sqrt(cov[i,i]*252):.2%}")

    # --- Step 2: Optimize ---
    print()
    print("=" * 65)
    print("STEP 2: Portfolio optimization")
    print("=" * 65)
    weights, selected = optimize_portfolio(
        mu, cov,
        max_positions=args.max_positions,
        lambda_risk=args.lambda_risk,
        cap=args.cap,
    )
    print(f"  Selected {len(selected)}/{M} assets:")
    for i in selected:
        print(f"    {tickers[i]:>5s} ({sectors[i]:<12s})  w={weights[i]:.4f}")
    print(f"  Total weight: {weights.sum():.4f}")

    # --- Step 3: GPU Monte Carlo ---
    print()
    print("=" * 65)
    print("STEP 3: GPU Monte Carlo VaR")
    print("=" * 65)
    mc = gpu_monte_carlo_var(
        cov, weights, tickers, sectors,
        n_scenarios=args.n_scenarios,
        df=args.df,
        alpha=args.alpha,
        seed=args.seed,
    )
    print(f"  Device: {mc['device']}")
    print(f"  Scenarios: {mc['n_scenarios']:,}")
    print(f"  Portfolio VaR(5%):  {mc['portfolio_var']:.6f}")
    print(f"  Portfolio CVaR(5%): {mc['portfolio_cvar']:.6f}")

    # --- Step 4: Plots ---
    print()
    print("=" * 65)
    print("STEP 4: Generating plots")
    print("=" * 65)
    plot_portfolio_report(mc, weights, tickers, sectors, output_dir)

    # --- Step 5: Report ---
    print_report(mc, weights, tickers, sectors, output_dir)

    # --- Step 6: Quantum (optional) ---
    quantum_result = None
    run_q = args.quantum and not args.no_quantum

    if run_q:
        print()
        print("=" * 65)
        print("STEP 6: Quantum IQAE VaR")
        print("=" * 65)
        quantum_result = run_quantum_var(
            mc["portfolio_losses"],
            num_qubits=7,
            epsilon=args.epsilon,
            alpha=args.alpha,
        )

        if quantum_result is not None:
            # Append quantum result to CSV
            csv_path = os.path.join(output_dir, "report.csv")
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow(["# Quantum IQAE Results"])
                writer.writerow(["quantum_var", quantum_result["quantum_var"]])
                writer.writerow(["classical_var_ref", quantum_result["classical_var"]])
                writer.writerow(["oracle_queries", quantum_result["oracle_queries"]])
                writer.writerow(["bisection_steps", quantum_result["bisection_steps"]])
                writer.writerow(["epsilon", args.epsilon])
    elif not args.no_quantum:
        print("\n  (Quantum IQAE not requested. Use --quantum to enable.)")

    # --- Final summary ---
    print()
    print("=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Output directory: {output_dir}")
    print(f"  MC device: {mc['device']}")
    print(f"  Portfolio VaR(5%):  {mc['portfolio_var']:.6f}")
    print(f"  Portfolio CVaR(5%): {mc['portfolio_cvar']:.6f}")
    if quantum_result:
        print(f"  Quantum VaR:        {quantum_result['quantum_var']:.6f}")
        print(f"  Oracle queries:     {quantum_result['oracle_queries']}")
    print()

    return {
        "tickers": tickers,
        "sectors": sectors,
        "weights": weights,
        "selected": selected,
        "mc": mc,
        "quantum": quantum_result,
    }


if __name__ == "__main__":
    main()
