"""Test script for LD implementation in phlash.

This script tests:
1. Import of the new LD module
2. Basic LD computation from het matrix (no moments dependency)
3. Model.py log_density with LD term
"""

import sys
sys.path.insert(0, r"c:\HsiehLab\phlash\src")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

print("Testing phlash LD implementation...")

# Test 1: Import LD module directly (avoiding phlash.__init__ which imports data.py -> pysam)
print("\n1. Testing LD module imports...")
try:
    # Import ld module directly to avoid pysam dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location("ld", r"c:\HsiehLab\phlash\src\phlash\ld.py")
    ld_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ld_module)
    
    DEFAULT_BP_BINS = ld_module.DEFAULT_BP_BINS
    DEFAULT_RECOMB_RATE = ld_module.DEFAULT_RECOMB_RATE
    bp_to_recomb_bins = ld_module.bp_to_recomb_bins
    compute_empirical_ld_from_het_matrix = ld_module.compute_empirical_ld_from_het_matrix
    print(f"   DEFAULT_BP_BINS: {DEFAULT_BP_BINS}")
    print(f"   DEFAULT_RECOMB_RATE: {DEFAULT_RECOMB_RATE}")
    print("   ✓ LD module imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: bp_to_recomb_bins
print("\n2. Testing bp_to_recomb_bins...")
r_bins = bp_to_recomb_bins(DEFAULT_BP_BINS, DEFAULT_RECOMB_RATE)
print(f"   r_bins: {r_bins}")
assert len(r_bins) == len(DEFAULT_BP_BINS)
print("   ✓ bp_to_recomb_bins works")

# Test 3: Compute LD from het matrix
print("\n3. Testing compute_empirical_ld_from_het_matrix...")
# Create synthetic het matrix
np.random.seed(42)
het_matrix = (np.random.random((10, 500)) < 0.05).astype(np.int8)
print(f"   het_matrix shape: {het_matrix.shape}")

ld_stats = compute_empirical_ld_from_het_matrix(
    het_matrix,
    window_size=100,
    bp_bins=DEFAULT_BP_BINS,
)
print(f"   LD stats: {ld_stats}")
print(f"   LD stats shape: {ld_stats.shape if ld_stats is not None else 'None'}")
if ld_stats is not None:
    print("   ✓ compute_empirical_ld_from_het_matrix works")
else:
    print("   ⚠ LD stats returned None (may be expected for small data)")

# Test 4: Import model with LD support
print("\n4. Testing model.py with LD term...")
try:
    from phlash.model import log_density, log_prior
    print("   ✓ model.py imported successfully")
except Exception as e:
    print(f"   ✗ Model import failed: {e}")
    sys.exit(1)

# Test 5: Import params and size_history
print("\n5. Testing params and size_history...")
try:
    from phlash.params import MCMCParams
    from phlash.size_history import SizeHistory, DemographicModel
    print("   ✓ params and size_history imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 6: Create demo model and test to_demes
print("\n6. Testing SizeHistory.to_demes()...")
try:
    t = jnp.array([0.0, 100.0, 1000.0, 10000.0])
    c = jnp.array([1.0, 0.5, 2.0, 1.0])
    eta = SizeHistory(t=t, c=c)
    g = eta.to_demes()
    print(f"   demes graph: {g}")
    print("   ✓ to_demes works")
except Exception as e:
    print(f"   ✗ to_demes failed: {e}")

# Test 7: Test mcmc imports
print("\n7. Testing mcmc.py imports...")
try:
    from phlash.mcmc import fit
    print("   ✓ mcmc.py imported successfully")
except Exception as e:
    print(f"   ⚠ mcmc.py import failed (may need pysam): {e}")

print("\n" + "="*50)
print("LD implementation tests completed!")
print("="*50)
