import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import xlogy
from jaxtyping import Array, Float, Float64, Int8, Int64
from typing import Optional

import phlash.hmm
from phlash.params import MCMCParams, PSMCParams


def log_prior(mcp: MCMCParams) -> float:
    ret = sum(
        jax.scipy.stats.norm.logpdf(a, loc=mu, scale=sigma).sum()
        for (a, mu, sigma) in [
            (jnp.log(mcp.rho_over_theta), 0.0, 1.0),
        ]
    )
    ret -= mcp.alpha * jnp.sum(jnp.diff(mcp.log_c) ** 2)
    x, _ = jax.flatten_util.ravel_pytree(mcp)
    ret -= mcp.beta * x.dot(x)
    return ret


def log_density(
    mcp: MCMCParams,
    c: Float64[Array, "4"],
    inds: Int64[Array, "batch"],
    warmup: Int8[Array, "c ell"],
    kern: "phlash.gpu.PSMCKernel",
    afs: Int64[Array, "n"],
    afs_transform: Float[Array, "m n"] = None,
    ld_stats: Optional[Float[Array, "k"]] = None,
    ld_bp_bins: Optional[Float[Array, "k1"]] = None,
    ld_recomb_rate: float = 1e-8,
) -> float:
    r"""
    Computes the log density of a statistical model by combining the contributions from
    the prior, the hidden Markov model (HMM), the allele frequency spectrum (AFS)
    model, and optionally the linkage disequilibrium (LD) model, weighted by given 
    coefficients.

    Args:
        mcp: The Markov Chain Monte Carlo parameters used to specify the model.
        c: Weights for each component of the density - prior, HMM model, AFS model,
           and LD model. Should be length 4: [c_prior, c_hmm, c_afs, c_ld].
        inds: Mini-batch indices for selecting subsets of the data.
        warmup: Warmup data for HMM initialization.
        kern: An instantiated PSMC Kernel used in the computation.
        afs: The allele frequency spectrum data.
        afs_transform: Optional transform matrix for AFS (e.g., folding/binning).
        ld_stats: Optional observed LD statistics (sigmaD2 values per distance bin).
        ld_bp_bins: Optional physical distance bins (bp) for LD computation.
        ld_recomb_rate: Recombination rate per bp (default 1e-8).

    Returns:
        The log density, or negative infinity where the result is not finite.
    """
    dm = mcp.to_dm()
    pp = PSMCParams.from_dm(dm)
    pis = vmap(lambda pp, d: phlash.hmm.psmc_ll(pp, d)[0], (None, 0))(
        pp, warmup
    )  # (I, M)
    pps = vmap(lambda pi: pp._replace(pi=pi))(pis)
    
    # l1: Prior term
    l1 = log_prior(mcp)
    
    # l2: HMM likelihood term
    l2 = vmap(kern.loglik, (0, 0))(pps, inds).sum()
    
    # l3: AFS likelihood term
    if afs is not None:
        n = len(afs) + 1
        if afs_transform is None:
            T = jnp.eye(n - 1)
        else:
            T = afs_transform
        assert T.ndim == 2
        assert T.shape[1] == n - 1
        etbl = dm.eta.etbl(n)
        esfs = etbl / etbl.sum()
        l3 = xlogy(T @ afs, T @ esfs).sum()
    else:
        l3 = 0.0
    
    # l4: LD likelihood term
    if ld_stats is not None and ld_bp_bins is not None:
        from phlash.ld import compute_expected_ld_jax

        ld_stats_sum = ld_stats.sum()
        if ld_stats_sum <= 0:
            l4 = 0.0
        else:
            # Compute expected LD from demographic model
            expected_ld = compute_expected_ld_jax(
                dm.eta,
                dm.theta,
                n_samples=20,
                bp_bins=ld_bp_bins,
                recomb_rate=ld_recomb_rate,
            )

            expected_ld_sum = expected_ld.sum()
            if expected_ld_sum <= 0:
                l4 = 0.0
            else:
                # Normalize for likelihood computation
                expected_ld_norm = expected_ld / expected_ld_sum
                expected_ld_norm = jnp.maximum(expected_ld_norm, 1e-20)

                # Normalize observed LD for comparison
                ld_stats_norm = ld_stats / ld_stats_sum

                # xlogy likelihood: sum of observed * log(expected)
                l4 = xlogy(ld_stats_norm, expected_ld_norm).sum()
    else:
        l4 = 0.0
    
    ll = jnp.array([l1, l2, l3, l4])
    ret = jnp.dot(c, ll)
    return jnp.where(jnp.isfinite(ret), ret, -jnp.inf)

