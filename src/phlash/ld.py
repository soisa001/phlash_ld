"""Linkage disequilibrium utilities.

This module provides lightweight helpers for estimating empirical LD
summaries from the windowed heterozygosity matrices already used by
phlash and a simple likelihood based on the current demographic model.

The implementation attempts to use the :mod:`moments.LD` framework when
available, but gracefully falls back to a JAX-friendly exponential decay
model when the optional dependency is missing. This keeps the LD
component usable on CPU or GPU without introducing a hard dependency.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from loguru import logger

try:  # pragma: no cover - optional dependency
    import moments.LD as moments_ld
except Exception:  # pragma: no cover - optional dependency
    moments_ld = None


def estimate_ld_stats(
    het_matrix: np.ndarray, window_size: int, max_lag: int = 10
) -> dict[str, Any]:
    """Estimate empirical LD statistics from a heterozygosity matrix.

    The heterozygosity matrix stores the number of heterozygous sites per
    window. We approximate LD by the correlation of heterozygosity counts
    across windows separated by a given lag. Missing windows (encoded as
    ``-1``) are ignored.

    Args:
        het_matrix: Array shaped ``(N, L)`` with per-window heterozygosity
            counts.
        window_size: Width of each window in base pairs.
        max_lag: Maximum window separation (in number of windows) to
            consider when computing correlations.

    Returns:
        Dictionary containing lags, corresponding genomic distances, the
        empirical correlations, and an estimate of their variance.
    """

    if het_matrix is None:
        return {
            "lags": np.array([], dtype=np.int32),
            "distances": np.array([], dtype=np.float64),
            "empirical": np.array([], dtype=np.float64),
            "variance": np.array([], dtype=np.float64),
        }

    if het_matrix.ndim != 2:
        raise ValueError("het_matrix must be 2D")

    emp = []
    var = []
    lags = np.arange(1, max_lag + 1)
    # missing data mask
    mask = het_matrix >= 0
    for lag in lags:
        x = het_matrix[:, :-lag]
        y = het_matrix[:, lag:]
        m = mask[:, :-lag] & mask[:, lag:]
        if not np.any(m):
            emp.append(0.0)
            var.append(1.0)
            continue
        # centre the counts per column while respecting missing data
        x_mean = np.where(m, x, 0).sum(axis=0) / np.maximum(m.sum(axis=0), 1)
        y_mean = np.where(m, y, 0).sum(axis=0) / np.maximum(m.sum(axis=0), 1)
        x_c = np.where(m, x - x_mean, 0)
        y_c = np.where(m, y - y_mean, 0)
        denom = np.sqrt(
            np.maximum(np.where(m, x_c**2, 0).sum(), 1e-9)
            * np.maximum(np.where(m, y_c**2, 0).sum(), 1e-9)
        )
        corr = np.where(m, x_c * y_c, 0).sum() / denom
        # number of effective pairs drives variance of the mean correlation
        eff_n = np.maximum(m.sum(), 1)
        emp.append(float(corr))
        var.append(float(max(1e-9, (1 - corr**2) / eff_n)))

    lags = np.asarray(lags)
    distances = lags * window_size
    return {
        "lags": lags,
        "distances": distances,
        "empirical": np.asarray(emp),
        "variance": np.asarray(var),
    }


def _moments_ld_expectation(dm, ld_data):
    """Compute LD expectation using :mod:`moments` when available."""

    if moments_ld is None:
        return None
    try:  # pragma: no cover - dependent on optional package
        # Construct a two-locus demographic model assuming a single
        # population size history.
        grid_pts = max(20, 2 * dm.M)
        rho = float(dm.rho)
        theta = float(dm.theta)
        # Two-locus equilibrium spectrum under piecewise-constant size
        # history. moments expects sizes scaled by 2N0.
        tl_model = moments_ld.Numerics.steady_state(
            ld_data["lags"], Npop=1, rho=rho, theta=theta, pop_ids=["pop"]
        )
        return tl_model
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("Failed to compute moments LD expectation: {}", e)
        return None


def ld_log_likelihood(dm, ld_data: dict | None) -> jnp.ndarray:
    """Log-likelihood contribution of linkage disequilibrium.

    Args:
        dm: Demographic model for which the LD expectation should be
            computed.
        ld_data: Output of :func:`estimate_ld_stats`.

    Returns:
        Scalar log-likelihood of the LD component.
    """

    if not ld_data or len(ld_data.get("lags", [])) == 0:
        return jnp.array(0.0)

    empirical = jnp.asarray(ld_data["empirical"], dtype=jnp.float64)
    variance = jnp.asarray(ld_data["variance"], dtype=jnp.float64)
    lags = jnp.asarray(ld_data["lags"], dtype=jnp.float64)

    # moments-based expectation if possible; otherwise fall back to an
    # exponential decay derived from the recombination rate.
    expected = _moments_ld_expectation(dm, ld_data)
    if expected is None:
        expected = jnp.exp(-dm.rho * lags)
    else:
        expected = jnp.asarray(expected, dtype=jnp.float64)
        if expected.shape != empirical.shape:
            expected = jnp.broadcast_to(expected, empirical.shape)

    resid = empirical - expected
    ll = -0.5 * jnp.sum(resid * resid / jnp.maximum(variance, 1e-9))
    return ll

