"""Linkage Disequilibrium computation for phlash.

This module provides functions to compute empirical LD statistics from genomic data
and expected LD from demographic models using the moments library.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy
from typing import Optional
from loguru import logger

# Default physical distance bins (bp) for LD decay: 0 to 20kb
DEFAULT_BP_BINS = np.array([0, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])

# Default recombination rate per bp
DEFAULT_RECOMB_RATE = 1e-8


def bp_to_recomb_bins(bp_bins: np.ndarray, recomb_rate: float = DEFAULT_RECOMB_RATE) -> np.ndarray:
    """Convert physical distance bins (bp) to recombination distance bins.
    
    Args:
        bp_bins: Array of physical distance bin edges in base pairs.
        recomb_rate: Recombination rate per bp per generation.
        
    Returns:
        Array of recombination distance bin edges.
    """
    return bp_bins * recomb_rate


def compute_empirical_ld_from_ts(
    ts,
    nodes: list[tuple[int, int]] = None,
    bp_bins: np.ndarray = DEFAULT_BP_BINS,
    recomb_rate: float = DEFAULT_RECOMB_RATE,
) -> Optional[np.ndarray]:
    """Compute empirical LD statistics from a tree sequence using moments.LD.
    
    Args:
        ts: tskit.TreeSequence object.
        nodes: List of (node1, node2) tuples representing diploid individuals.
               If None, use all individuals in the tree sequence.
        bp_bins: Physical distance bins in base pairs.
        recomb_rate: Recombination rate per bp.
        
    Returns:
        Array of sigmaD2 statistics for each recombination bin, or None if 
        LD computation fails or is not possible.
    """
    try:
        import moments.LD
    except ImportError:
        logger.warning("moments package not installed. LD computation disabled.")
        return None
    
    try:
        import tempfile
        import os
        
        # Get nodes to analyze
        if nodes is None:
            nodes = [tuple(i.nodes) for i in ts.individuals()]
        
        nodes_flat = list({x for t in nodes for x in t})
        n_samples = len(nodes_flat)
        
        if n_samples < 4:
            logger.debug("Too few samples for LD computation (need at least 4 haplotypes)")
            return None
            
        # Convert bp bins to recombination distance
        r_bins = bp_to_recomb_bins(bp_bins, recomb_rate)
        
        # Write tree sequence to temporary VCF for moments.LD parsing
        with tempfile.TemporaryDirectory() as tmpdir:
            vcf_path = os.path.join(tmpdir, "data.vcf")
            pop_path = os.path.join(tmpdir, "pops.txt")
            map_path = os.path.join(tmpdir, "recomb.map")
            
            # Write VCF
            with open(vcf_path, "w") as f:
                ts.write_vcf(f, individuals=list(range(len(nodes))))
            
            # Write population file (single population)
            with open(pop_path, "w") as f:
                f.write("sample\tpop\n")
                for i in range(len(nodes)):
                    f.write(f"tsk_{i}\tpop0\n")
            
            # Write recombination map (uniform rate)
            L = int(ts.sequence_length)
            map_cm = L * recomb_rate * 100  # convert to cM
            with open(map_path, "w") as f:
                f.write("Pos\tMap(cM)\n")
                f.write(f"0\t0\n")
                f.write(f"{L}\t{map_cm}\n")
            
            # Compute LD statistics using moments
            ld_stats = moments.LD.Parsing.compute_ld_statistics(
                vcf_path,
                rec_map_file=map_path,
                pop_file=pop_path,
                pops=["pop0"],
                r_bins=r_bins,
                report=False,
            )
            
            # Get mean statistics
            means = moments.LD.Parsing.means_from_region_data(
                {0: ld_stats}, 
                ld_stats["stats"], 
                norm_idx=0
            )
            
            # Extract sigmaD2 for single population (DD_0_0 statistic)
            # The means object contains normalized LD statistics
            if hasattr(means, '__iter__') and len(means) > 0:
                # Get the DD statistic (D^2) for population 0
                sigma_d2 = np.array([m.get("DD_0_0", 0.0) if isinstance(m, dict) else 0.0 
                                     for m in means[:-1]])
                return sigma_d2
            
    except Exception as e:
        logger.debug(f"LD computation from tree sequence failed: {e}")
        
    return None


def compute_empirical_ld_from_het_matrix(
    het_matrix: np.ndarray,
    window_size: int = 100,
    bp_bins: np.ndarray = DEFAULT_BP_BINS,
) -> Optional[np.ndarray]:
    """Compute empirical LD statistics from heterozygosity matrix.
    
    This is a simplified LD computation that works directly on the binned
    heterozygosity data without requiring moments.LD.
    
    Args:
        het_matrix: Binary matrix of shape (N, L) where N is number of diploid
                    samples and L is number of windows.
        window_size: Size of each window in base pairs.
        bp_bins: Physical distance bins in base pairs.
        
    Returns:
        Array of LD decay values (correlation of heterozygosity at distance).
    """
    if het_matrix is None:
        return None
        
    N, L = het_matrix.shape
    if L < 10:
        return None
    
    # Convert heterozygosity to allele indicator (0/1)
    # Note: het_matrix contains counts of het sites per window, convert to binary
    alleles = (het_matrix > 0).astype(float)
    
    # Mask missing data
    mask = het_matrix >= 0
    alleles = np.where(mask, alleles, np.nan)
    
    # Compute LD decay as correlation at different distances
    ld_values = []
    bin_edges = bp_bins // window_size  # Convert bp to window indices
    
    for i in range(len(bin_edges) - 1):
        d_min = max(1, int(bin_edges[i]))
        d_max = int(bin_edges[i + 1])
        
        if d_max >= L:
            d_max = L - 1
            
        if d_min >= d_max:
            ld_values.append(0.0)
            continue
            
        # Compute average correlation at this distance range
        correlations = []
        for d in range(d_min, min(d_max + 1, L)):
            a1 = alleles[:, :-d]
            a2 = alleles[:, d:]
            
            # Compute D^2 (correlation squared)
            valid = ~np.isnan(a1) & ~np.isnan(a2)
            if valid.sum() > 10:
                p1 = np.nanmean(a1)
                p2 = np.nanmean(a2)
                if p1 * (1-p1) > 0 and p2 * (1-p2) > 0:
                    cov = np.nanmean(a1 * a2) - p1 * p2
                    d2 = cov ** 2
                    # sigmaD2 = D^2 / (p1(1-p1)p2(1-p2))
                    sigma_d2 = d2 / (p1 * (1-p1) * p2 * (1-p2))
                    correlations.append(sigma_d2)
        
        if correlations:
            ld_values.append(np.mean(correlations))
        else:
            ld_values.append(0.0)
    
    return np.array(ld_values)


def compute_expected_ld(
    eta,  # SizeHistory
    theta: float,
    n_samples: int = 20,
    bp_bins: np.ndarray = DEFAULT_BP_BINS,
    recomb_rate: float = DEFAULT_RECOMB_RATE,
) -> np.ndarray:
    """Compute expected LD from a demographic model using moments.Demes.LD.
    
    Args:
        eta: SizeHistory object representing the demographic model.
        theta: Mutation rate parameter.
        n_samples: Number of haploid samples to simulate.
        bp_bins: Physical distance bins in base pairs.
        recomb_rate: Recombination rate per bp.
        
    Returns:
        Array of expected sigmaD2 values for each bin.
    """
    try:
        import moments.LD
        import moments.Demes
    except ImportError:
        logger.warning("moments package not installed.")
        # Return small positive values so downstream likelihoods remain well-defined
        return np.ones(len(bp_bins) - 1) * 1e-10
    
    try:
        # Convert SizeHistory to demes Graph
        g = eta.to_demes()
        
        # Get reference Ne for rho scaling
        # Use the first epoch's size as reference
        Ne_ref = 1.0 / (2.0 * float(eta.c[0]))
        
        # Convert bp bins to recombination distance (rho = 4*Ne*r*L)
        r_bins = bp_to_recomb_bins(bp_bins, recomb_rate)
        rho = 4 * Ne_ref * r_bins
        
        # Compute expected LD using moments
        y = moments.Demes.LD(
            g,
            sampled_demes=["pop"],  # Single population named "pop" from to_demes()
            rho=rho,
        )
        
        # Average bin edges to get midpoint estimates
        y = moments.LD.LDstats(
            [(y_l + y_r) / 2 for y_l, y_r in zip(y[:-2], y[1:-1])] + [y[-1]],
            num_pops=y.num_pops,
            pop_ids=y.pop_ids,
        )
        
        # Convert to sigmaD2
        y = moments.LD.Inference.sigmaD2(y)
        
        # Extract DD_0_0 (single population D^2)
        sigma_d2 = np.array([y[i].get("DD_0_0", 1e-10) for i in range(len(y))])
        
        # Ensure positive values for log likelihood
        sigma_d2 = np.maximum(sigma_d2, 1e-10)
        
        return sigma_d2
        
    except Exception as e:
        logger.debug(f"Expected LD computation failed: {e}")
        return np.ones(len(bp_bins) - 1) * 1e-10


def compute_expected_ld_jax(
    eta,  # SizeHistory
    theta: float,
    n_samples: int = 20,
    bp_bins: np.ndarray = DEFAULT_BP_BINS,
    recomb_rate: float = DEFAULT_RECOMB_RATE,
) -> jnp.ndarray:
    """JAX-compatible wrapper for computing expected LD.
    
    Uses jax.pure_callback to call the numpy-based moments computation.
    Note: This is not differentiable through the moments computation, but
    the xlogy likelihood remains differentiable.
    
    Args:
        eta: SizeHistory object.
        theta: Mutation rate.
        n_samples: Number of samples.
        bp_bins: Physical distance bins.
        recomb_rate: Recombination rate per bp.
        
    Returns:
        JAX array of expected sigmaD2 values.
    """
    n_bins = len(bp_bins) - 1
    
    def _compute_ld_numpy(t, c, theta_val):
        """Pure numpy function to compute LD."""
        from phlash.size_history import SizeHistory
        eta_np = SizeHistory(t=np.array(t), c=np.array(c))
        return compute_expected_ld(
            eta_np, 
            float(theta_val), 
            n_samples, 
            bp_bins, 
            recomb_rate
        ).astype(np.float64)
    
    # Use pure_callback for numpy interop
    result = jax.pure_callback(
        _compute_ld_numpy,
        jax.ShapeDtypeStruct((n_bins,), jnp.float64),
        eta.t,
        eta.c,
        theta,
    )
    
    return result


def ld_log_likelihood(
    observed_ld: jnp.ndarray,
    expected_ld: jnp.ndarray,
) -> float:
    """Compute log likelihood of observed LD given expected LD.
    
    Uses xlogy for numerical stability, similar to AFS likelihood.
    
    Args:
        observed_ld: Observed sigmaD2 values (or counts).
        expected_ld: Expected sigmaD2 values from demographic model.
        
    Returns:
        Log likelihood value.
    """
    # Normalize expected LD to probability-like values
    expected_ld = expected_ld / expected_ld.sum()
    expected_ld = jnp.maximum(expected_ld, 1e-20)
    
    # Use xlogy: x * log(y), handles x=0 correctly
    return xlogy(observed_ld, expected_ld).sum()
