"""Basic tests for the LD term and its integration in phlash."""

import jax
import jax.numpy as jnp
import numpy as np

from phlash.kernel import get_kernel
from phlash.ld import (
    DEFAULT_BP_BINS,
    DEFAULT_RECOMB_RATE,
    bp_to_recomb_bins,
    compute_empirical_ld_from_het_matrix,
)
from phlash.model import log_density
from phlash.params import MCMCParams

jax.config.update("jax_enable_x64", True)


def test_ld_module_imports():
    """Ensure defaults are available and conversions work."""
    r_bins = bp_to_recomb_bins(DEFAULT_BP_BINS, DEFAULT_RECOMB_RATE)
    np.testing.assert_allclose(r_bins, DEFAULT_BP_BINS * DEFAULT_RECOMB_RATE)


def test_compute_empirical_ld_from_het_matrix():
    rng = np.random.default_rng(42)
    het_matrix = (rng.random((6, 400)) < 0.05).astype(np.int8)
    bp_bins = DEFAULT_BP_BINS[:6]

    ld_stats = compute_empirical_ld_from_het_matrix(
        het_matrix, window_size=50, bp_bins=bp_bins
    )

    assert ld_stats is not None
    assert ld_stats.shape == (len(bp_bins) - 1,)
    assert ld_stats.sum() >= 0


def test_log_density_with_and_without_ld_term():
    """Simulate log_density evaluation toggling the LD contribution."""
    rng = np.random.default_rng(0)
    het_matrix = (rng.random((8, 300)) < 0.05).astype(np.int8)
    kern = get_kernel(M=16, data=het_matrix, double_precision=True)

    # Simple AFS with the correct dimension (n-1 == 15 when M=16)
    afs = jnp.ones(15)

    mcp = MCMCParams.from_linear(
        pattern="16*1",
        t1=1e-3,
        tM=1.0,
        c=jnp.ones(16),
        theta=1e-2,
        rho=1e-2,
    )

    ld_bins = jnp.array(DEFAULT_BP_BINS[:5])
    ld_stats = compute_empirical_ld_from_het_matrix(
        het_matrix, window_size=1, bp_bins=np.asarray(ld_bins)
    )
    assert ld_stats is not None
    if ld_stats.sum() == 0:
        ld_stats = ld_stats + 1e-3
    ld_stats = jnp.array(ld_stats)

    base_kwargs = dict(
        inds=jnp.arange(het_matrix.shape[0]),
        warmup=jnp.full((het_matrix.shape[0], 1), -1, dtype=jnp.int8),
        kern=kern,
        afs=afs,
        afs_transform=None,
        ld_stats=ld_stats,
        ld_bp_bins=ld_bins,
        ld_recomb_rate=DEFAULT_RECOMB_RATE,
    )

    weights_no_ld = jnp.array([1.0, het_matrix.shape[0], 1.0, 0.0])
    weights_with_ld = weights_no_ld.at[3].set(1.0)

    ll_no_ld = log_density(mcp, c=weights_no_ld, **base_kwargs)
    ll_with_ld = log_density(mcp, c=weights_with_ld, **base_kwargs)

    assert jnp.isfinite(ll_no_ld)
    assert jnp.isfinite(ll_with_ld)
    assert ll_with_ld != ll_no_ld
