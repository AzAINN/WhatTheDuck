"""Data helpers for portfolio MIQP examples."""

from __future__ import annotations

import numpy as np


def compute_mu_cov(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sample mean and covariance from returns matrix.

    Args:
        returns: shape (n_samples, n_assets)
    """
    if returns.ndim != 2:
        raise ValueError("returns must be 2-D (samples, assets)")
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)
    return mu, cov


def synthetic_factor_model(
    *,
    n_samples: int = 4096,
    n_assets: int = 50,
    n_factors: int = 5,
    factor_var: float = 0.05,
    idio_var: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate returns from a simple linear factor model: r = F @ B^T + ε.
    """
    rng = np.random.default_rng(seed)
    F = rng.normal(scale=np.sqrt(factor_var), size=(n_samples, n_factors))
    B = rng.normal(scale=1.0 / np.sqrt(n_factors), size=(n_assets, n_factors))
    eps = rng.normal(scale=np.sqrt(idio_var), size=(n_samples, n_assets))
    returns = F @ B.T + eps
    return returns


def caps_from_constant(max_weight: float, n_assets: int) -> np.ndarray:
    """
    Constant per-asset cap u_i. Typical values: 0.05–0.10 for large universes.
    """
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    return np.full(n_assets, max_weight, dtype=float)
