import jax
import jax.numpy as jnp
import pytest

from probjax.nn.pallas_kernels import compute_mamba_scan, ssd, ssd_linear_scan


def test_mamba_scan_requires_accelerator():
    """Test that mamba scan raises an error on CPU."""
    batch = 1
    seq_len = 16
    inner_dim = 128
    state_dim = 8
    seq_tile_size = 8
    dim_tile_size = 128

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, seq_len, inner_dim), dtype=jnp.float32)
    a = jax.random.normal(key, (state_dim, inner_dim), dtype=jnp.float32)
    b = jax.random.normal(key, (batch, seq_len, state_dim), dtype=jnp.float32)
    c = jax.random.normal(key, (batch, seq_len, state_dim), dtype=jnp.float32)
    delta = jax.random.normal(key, (batch, seq_len, inner_dim), dtype=jnp.float32)
    d = jax.random.normal(key, (1, inner_dim), dtype=jnp.float32)

    if jax.default_backend() == "cpu":
        with pytest.raises(Exception):
            compute_mamba_scan(
                x,
                a,
                b,
                c,
                delta,
                d,
                seq_tile_size=seq_tile_size,
                dim_tile_size=dim_tile_size,
            )
    else:
        out = compute_mamba_scan(
            x,
            a,
            b,
            c,
            delta,
            d,
            seq_tile_size=seq_tile_size,
            dim_tile_size=dim_tile_size,
        )
        assert out.shape == (batch, seq_len, inner_dim)


def test_ssd_requires_accelerator():
    """Test that SSD raises an error on CPU."""
    batch = 1
    num_groups = 1
    num_heads = 1
    seq_len = 512
    dk = 128
    dv = 128

    key = jax.random.PRNGKey(1)
    q = jax.random.normal(key, (batch, num_groups, seq_len, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (batch, num_groups, seq_len, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (batch, num_heads, seq_len, dv), dtype=jnp.float32)
    log_alpha = jax.random.normal(key, (batch, num_heads, seq_len), dtype=jnp.float32)

    if jax.default_backend() == "cpu":
        with pytest.raises(Exception):
            ssd(q, k, v, log_alpha)
    else:
        out = ssd(q, k, v, log_alpha)
        ref, _ = ssd_linear_scan(q, k, v, log_alpha)
        assert out.shape == ref.shape
        assert jnp.allclose(out, ref, atol=1e-2, rtol=1e-2)
