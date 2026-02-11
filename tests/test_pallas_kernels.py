import jax
import jax.numpy as jnp
import pytest

from probjax.nn.pallas_kernels.mambda import compute_mamba_scan
from probjax.nn.pallas_kernels.ssd import ssd, ssd_linear_scan


def _skip_if_cpu():
    if jax.default_backend() == "cpu":
        pytest.skip("pallas kernels require accelerator backend")


def _mamba_reference(x, a, b, c, delta, d):
    # Reference scan based on mambda forward kernel.
    def body(h, inputs):
        x_t, b_t, c_t, delta_t = inputs
        delta_t = delta_t[None, :]
        a_bar = jnp.exp(delta_t * a)
        b_bar = delta_t * b_t[:, None] * x_t[None, :]
        h_next = a_bar * h + b_bar
        y_t = c_t[None, :] @ h_next + x_t[None, :] * d
        return h_next, jnp.squeeze(y_t, axis=0)

    h0 = jnp.zeros((a.shape[0], a.shape[1]), dtype=jnp.float32)
    _, y = jax.lax.scan(body, h0, (x, b, c, delta))
    return y


def test_mamba_scan_matches_reference_small():
    _skip_if_cpu()
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

    out = compute_mamba_scan(
        x, a, b, c, delta, d, seq_tile_size=seq_tile_size, dim_tile_size=dim_tile_size
    )
    ref = jax.vmap(_mamba_reference, in_axes=(0, None, 0, 0, 0, None))(
        x, a, b, c, delta, d
    )

    assert out.shape == ref.shape
    assert jnp.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_ssd_matches_linear_scan_small():
    _skip_if_cpu()
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
    log_alpha = jax.random.normal(
        key, (batch, num_heads, seq_len), dtype=jnp.float32
    )

    out = ssd(q, k, v, log_alpha)
    ref, _ = ssd_linear_scan(q, k, v, log_alpha)

    assert out.shape == ref.shape
    assert jnp.allclose(out, ref, atol=1e-2, rtol=1e-2)
