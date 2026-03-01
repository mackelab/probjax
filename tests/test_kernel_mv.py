import jax
import jax.numpy as jnp
import pytest

from probjax.nn.pallas_kernels.kernel_mv import (
    kde_density,
    kernel_mv,
    kernel_mv_naive,
    rbf_kde_density,
    rbf_kernel_mv,
    rbf_kernel_mv_naive,
)


def _scaled_rbf_kernel_fn(q_block, k_block, params):
    amp, lengthscale = params[0], params[1]
    inv_l2 = 1.0 / (lengthscale * lengthscale)
    q_norm2 = jnp.sum(q_block * q_block, axis=-1, keepdims=True)
    k_norm2 = jnp.sum(k_block * k_block, axis=-1)[None, :]
    qk = q_block @ k_block.T
    r2 = jnp.maximum(q_norm2 + k_norm2 - 2.0 * qk, 0.0)
    base = jnp.exp(-0.5 * inv_l2 * r2)
    return amp * base


def _scaled_rbf_kernel_vjp_fn(q_block, k_block, params, ct):
    amp, lengthscale = params[0], params[1]
    inv_l2 = 1.0 / (lengthscale * lengthscale)
    inv_l3 = inv_l2 / lengthscale
    q_norm2 = jnp.sum(q_block * q_block, axis=-1, keepdims=True)
    k_norm2 = jnp.sum(k_block * k_block, axis=-1)[None, :]
    qk = q_block @ k_block.T
    r2 = jnp.maximum(q_norm2 + k_norm2 - 2.0 * qk, 0.0)
    base = jnp.exp(-0.5 * inv_l2 * r2)
    w = amp * base
    coeff = ct * w

    row_sum = jnp.sum(coeff, axis=-1)
    col_sum = jnp.sum(coeff, axis=0)
    dq = inv_l2 * (coeff @ k_block - row_sum[:, None] * q_block)
    dk = inv_l2 * (coeff.T @ q_block - col_sum[:, None] * k_block)
    damp = jnp.sum(ct * base)
    dlengthscale = inv_l3 * jnp.sum(coeff * r2)
    dparams = jnp.asarray([damp, dlengthscale], dtype=params.dtype)
    return dq, dk, dparams


@pytest.mark.parametrize(
    "batch_size, n, m, d, o",
    [
        (1, 13, 17, 7, 5),
        (2, 32, 24, 16, 8),
        (2, 9, 11, 3, 4),
    ],
)
def test_rbf_kernel_mv_matches_naive_forward(batch_size, n, m, d, o):
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(key_q, (batch_size, n, d), dtype=jnp.float32)
    k = jax.random.normal(key_k, (batch_size, m, d), dtype=jnp.float32)
    v = jax.random.normal(key_v, (batch_size, m, o), dtype=jnp.float32)
    lengthscale = jnp.asarray(1.3, dtype=jnp.float32)

    out_ref = rbf_kernel_mv_naive(q, k, v, lengthscale)
    out = rbf_kernel_mv(
        q,
        k,
        v,
        lengthscale,
        block_q=8,
        block_k=8,
        block_o=8,
        interpret=True,
    )

    assert out.shape == out_ref.shape == (batch_size, n, o)
    assert jnp.allclose(out, out_ref, atol=1e-4, rtol=1e-4)


def test_rbf_kernel_mv_reverse_grads_match_naive():
    batch_size, n, m, d, o = 2, 11, 15, 6, 7
    key_q, key_k, key_v, key_t = jax.random.split(jax.random.PRNGKey(1), 4)
    q = jax.random.normal(key_q, (batch_size, n, d), dtype=jnp.float32)
    k = jax.random.normal(key_k, (batch_size, m, d), dtype=jnp.float32)
    v = jax.random.normal(key_v, (batch_size, m, o), dtype=jnp.float32)
    target = jax.random.normal(key_t, (batch_size, n, o), dtype=jnp.float32)
    lengthscale = jnp.asarray(1.1, dtype=jnp.float32)

    def loss_kernel(q_, k_, v_, ls_):
        out = rbf_kernel_mv(
            q_,
            k_,
            v_,
            ls_,
            block_q=8,
            block_k=8,
            block_o=8,
            interpret=True,
        )
        return jnp.sum((out - target) ** 2)

    def loss_naive(q_, k_, v_, ls_):
        out = rbf_kernel_mv_naive(q_, k_, v_, ls_)
        return jnp.sum((out - target) ** 2)

    grads_kernel = jax.grad(loss_kernel, argnums=(0, 1, 2, 3))(q, k, v, lengthscale)
    grads_ref = jax.grad(loss_naive, argnums=(0, 1, 2, 3))(q, k, v, lengthscale)

    assert jnp.allclose(grads_kernel[0], grads_ref[0], atol=3e-3, rtol=2e-3)
    assert jnp.allclose(grads_kernel[1], grads_ref[1], atol=3e-3, rtol=2e-3)
    assert jnp.allclose(grads_kernel[2], grads_ref[2], atol=3e-3, rtol=2e-3)
    assert jnp.allclose(grads_kernel[3], grads_ref[3], atol=3e-3, rtol=2e-3)


def test_generic_kernel_mv_with_vector_params_matches_naive_and_grads():
    batch_size, n, m, d, o = 2, 10, 13, 5, 6
    key_q, key_k, key_v, key_t = jax.random.split(jax.random.PRNGKey(2), 4)
    q = jax.random.normal(key_q, (batch_size, n, d), dtype=jnp.float32)
    k = jax.random.normal(key_k, (batch_size, m, d), dtype=jnp.float32)
    v = jax.random.normal(key_v, (batch_size, m, o), dtype=jnp.float32)
    target = jax.random.normal(key_t, (batch_size, n, o), dtype=jnp.float32)
    params = jnp.asarray([0.7, 1.2], dtype=jnp.float32)

    out_ref = kernel_mv_naive(q, k, v, params, _scaled_rbf_kernel_fn)
    out = kernel_mv(
        q,
        k,
        v,
        params,
        _scaled_rbf_kernel_fn,
        _scaled_rbf_kernel_vjp_fn,
        block_q=8,
        block_k=8,
        block_o=8,
        interpret=True,
    )
    assert jnp.allclose(out, out_ref, atol=1e-4, rtol=1e-4)

    def loss_kernel(q_, k_, v_, p_):
        out_ = kernel_mv(
            q_,
            k_,
            v_,
            p_,
            _scaled_rbf_kernel_fn,
            _scaled_rbf_kernel_vjp_fn,
            block_q=8,
            block_k=8,
            block_o=8,
            interpret=True,
        )
        return jnp.sum((out_ - target) ** 2)

    def loss_naive(q_, k_, v_, p_):
        out_ = kernel_mv_naive(q_, k_, v_, p_, _scaled_rbf_kernel_fn)
        return jnp.sum((out_ - target) ** 2)

    grads_kernel = jax.grad(loss_kernel, argnums=(0, 1, 2, 3))(q, k, v, params)
    grads_ref = jax.grad(loss_naive, argnums=(0, 1, 2, 3))(q, k, v, params)

    assert jnp.allclose(grads_kernel[0], grads_ref[0], atol=4e-3, rtol=3e-3)
    assert jnp.allclose(grads_kernel[1], grads_ref[1], atol=4e-3, rtol=3e-3)
    assert jnp.allclose(grads_kernel[2], grads_ref[2], atol=4e-3, rtol=3e-3)
    assert jnp.allclose(grads_kernel[3], grads_ref[3], atol=4e-3, rtol=3e-3)


def test_rbf_kde_density_matches_naive_density():
    batch_size, n, m, d = 2, 9, 12, 4
    key_q, key_x, key_w = jax.random.split(jax.random.PRNGKey(7), 3)
    query = jax.random.normal(key_q, (batch_size, n, d), dtype=jnp.float32)
    data = jax.random.normal(key_x, (batch_size, m, d), dtype=jnp.float32)
    weights = jax.random.uniform(key_w, (batch_size, m), dtype=jnp.float32)
    lengthscale = jnp.asarray(0.9, dtype=jnp.float32)

    out = rbf_kde_density(
        query,
        data,
        lengthscale,
        weights=weights,
        normalized=True,
        block_q=8,
        block_k=8,
        interpret=True,
    )

    w = weights / jnp.sum(weights, axis=-1, keepdims=True)
    ref = rbf_kernel_mv_naive(query, data, w[..., None], lengthscale)[..., 0]
    normalizer = 1.0 / ((jnp.sqrt(2.0 * jnp.pi) * lengthscale) ** d)
    ref = ref * normalizer

    assert out.shape == (batch_size, n)
    assert jnp.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_generic_kde_density_log_output_finite():
    n, m, d = 10, 14, 3
    key_q, key_x = jax.random.split(jax.random.PRNGKey(8), 2)
    query = jax.random.normal(key_q, (n, d), dtype=jnp.float32)
    data = jax.random.normal(key_x, (m, d), dtype=jnp.float32)
    params = jnp.asarray([1.3, 0.7], dtype=jnp.float32)

    log_d = kde_density(
        query,
        data,
        params,
        _scaled_rbf_kernel_fn,
        _scaled_rbf_kernel_vjp_fn,
        log_density=True,
        block_q=8,
        block_k=8,
        interpret=True,
    )

    assert log_d.shape == (n,)
    assert jnp.all(jnp.isfinite(log_d))
