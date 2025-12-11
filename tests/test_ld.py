import jax.numpy as jnp

from phlash.ld import estimate_ld_stats, ld_log_likelihood
from phlash.params import MCMCParams


def test_estimate_ld_stats_basic():
    het = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=jnp.int8)
    stats = estimate_ld_stats(het, window_size=100, max_lag=2)
    assert stats["lags"].tolist() == [1, 2]
    assert stats["distances"].tolist() == [100, 200]
    assert len(stats["empirical"]) == 2
    assert len(stats["variance"]) == 2


def test_ld_log_likelihood_matches_expectation():
    mcp = MCMCParams.from_linear(
        pattern="1*1",
        t1=0.1,
        tM=1.0,
        c=jnp.array([1.0], dtype=jnp.float64),
        theta=0.01,
        rho=0.02,
    )
    dm = mcp.to_dm()
    ld_data = {
        "lags": jnp.array([1.0, 2.0]),
        "empirical": jnp.exp(-dm.rho * jnp.array([1.0, 2.0])),
        "variance": jnp.array([1e-3, 1e-3]),
    }
    # When empirical matches model expectation, contribution should be near zero
    ll = ld_log_likelihood(dm, ld_data)
    assert jnp.isfinite(ll)
    assert jnp.abs(ll) < 1e-3

