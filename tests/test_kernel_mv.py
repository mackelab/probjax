import importlib

import jax
import jax.numpy as jnp
import pytest

from probjax.nn.pallas_kernels import (
    compute_mamba_scan,
    kde_density,
    kernel_mv,
    kernel_mv_naive,
    rbf_kde_density,
    rbf_kernel_mv,
    rbf_kernel_mv_naive,
    ssd,
    ssd_linear_scan,
)

mamba_kernel_mod = importlib.import_module("probjax.nn.pallas_kernels.kernels.mamba")
ssd_kernel_mod = importlib.import_module("probjax.nn.pallas_kernels.kernels.ssd")


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


def test_mamba_scan_requires_supported_backend_config():
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

    if jax.default_backend() != "tpu":
        with pytest.raises(RuntimeError):
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


def test_ssd_requires_supported_backend_config():
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

    if jax.default_backend() != "tpu":
        with pytest.raises(RuntimeError):
            ssd(q, k, v, log_alpha)
    else:
        out = ssd(q, k, v, log_alpha)
        ref, _ = ssd_linear_scan(q, k, v, log_alpha)
        assert out.shape == ref.shape
        assert jnp.allclose(out, ref, atol=1e-2, rtol=1e-2)


def test_mamba_scan_gpu_supported_config_uses_pallas_scan(monkeypatch):
    batch = 1
    seq_len = 16
    inner_dim = 128
    state_dim = 16
    seq_tile_size = 8
    dim_tile_size = 128

    key = jax.random.PRNGKey(6)
    x = jax.random.normal(key, (batch, seq_len, inner_dim), dtype=jnp.float32)
    a = jax.random.normal(key, (state_dim, inner_dim), dtype=jnp.float32)
    b = jax.random.normal(key, (batch, seq_len, state_dim), dtype=jnp.float32)
    c = jax.random.normal(key, (batch, seq_len, state_dim), dtype=jnp.float32)
    delta = jax.random.normal(key, (batch, seq_len, inner_dim), dtype=jnp.float32)
    d = jax.random.normal(key, (1, inner_dim), dtype=jnp.float32)

    monkeypatch.setattr(mamba_kernel_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        mamba_kernel_mod, "_pallas_backend", lambda prefer_mosaic_gpu=None: "triton"
    )
    monkeypatch.setattr(
        mamba_kernel_mod, "_gpu_supports_mamba_pallas_for_shape", lambda **_: True
    )

    def _unexpected_reference(*args, **kwargs):
        raise AssertionError("reference fallback should be disabled")

    def _fake_scan(*args, **kwargs):
        x_arg = args[0]
        return jnp.full_like(x_arg, 7.0)

    monkeypatch.setattr(
        mamba_kernel_mod, "_mamba_scan_reference", _unexpected_reference
    )
    monkeypatch.setattr(mamba_kernel_mod, "_make_mamba_scan", lambda *args: _fake_scan)

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

    assert out.shape == x.shape
    assert jnp.all(out == 7.0)


def test_mamba_scan_gpu_unsupported_config_raises_without_reference_fallback(
    monkeypatch,
):
    batch = 1
    seq_len = 16
    inner_dim = 128
    state_dim = 16
    seq_tile_size = 8
    dim_tile_size = 128

    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (batch, seq_len, inner_dim), dtype=jnp.float32)
    a = jax.random.normal(key, (state_dim, inner_dim), dtype=jnp.float32)
    b = jax.random.normal(key, (batch, seq_len, state_dim), dtype=jnp.float32)
    c = jax.random.normal(key, (batch, seq_len, state_dim), dtype=jnp.float32)
    delta = jax.random.normal(key, (batch, seq_len, inner_dim), dtype=jnp.float32)
    d = jax.random.normal(key, (1, inner_dim), dtype=jnp.float32)

    monkeypatch.setattr(mamba_kernel_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        mamba_kernel_mod, "_pallas_backend", lambda prefer_mosaic_gpu=None: "triton"
    )
    monkeypatch.setattr(
        mamba_kernel_mod, "_gpu_supports_mamba_pallas_for_shape", lambda **_: False
    )

    def _unexpected_reference(*args, **kwargs):
        raise AssertionError("reference fallback should be disabled")

    monkeypatch.setattr(
        mamba_kernel_mod, "_mamba_scan_reference", _unexpected_reference
    )

    with pytest.raises(RuntimeError, match="Reference fallback on GPU is disabled"):
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


def test_mamba_scan_gpu_kernel_failure_raises_without_reference_fallback(monkeypatch):
    batch = 1
    seq_len = 16
    inner_dim = 128
    state_dim = 16
    seq_tile_size = 8
    dim_tile_size = 128

    key = jax.random.PRNGKey(3)
    x = jax.random.normal(key, (batch, seq_len, inner_dim), dtype=jnp.float32)
    a = jax.random.normal(key, (state_dim, inner_dim), dtype=jnp.float32)
    b = jax.random.normal(key, (batch, seq_len, state_dim), dtype=jnp.float32)
    c = jax.random.normal(key, (batch, seq_len, state_dim), dtype=jnp.float32)
    delta = jax.random.normal(key, (batch, seq_len, inner_dim), dtype=jnp.float32)
    d = jax.random.normal(key, (1, inner_dim), dtype=jnp.float32)

    monkeypatch.setattr(mamba_kernel_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        mamba_kernel_mod, "_pallas_backend", lambda prefer_mosaic_gpu=None: "triton"
    )
    monkeypatch.setattr(
        mamba_kernel_mod, "_gpu_supports_mamba_pallas_for_shape", lambda **_: True
    )

    def _unexpected_reference(*args, **kwargs):
        raise AssertionError("reference fallback should be disabled")

    def _failing_scan(*args, **kwargs):
        raise RuntimeError("pallas compile failed")

    monkeypatch.setattr(
        mamba_kernel_mod, "_mamba_scan_reference", _unexpected_reference
    )
    monkeypatch.setattr(
        mamba_kernel_mod, "_make_mamba_scan", lambda *args: _failing_scan
    )

    with pytest.raises(RuntimeError, match="pallas compile failed"):
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


def test_ssd_gpu_supported_config_uses_pallas_kernel(monkeypatch):
    batch = 1
    num_groups = 1
    num_heads = 2
    seq_len = 128
    dk = 64
    dv = 64

    key = jax.random.PRNGKey(7)
    q = jax.random.normal(key, (batch, num_groups, seq_len, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (batch, num_groups, seq_len, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (batch, num_heads, seq_len, dv), dtype=jnp.float32)
    log_alpha = jax.random.normal(key, (batch, num_heads, seq_len), dtype=jnp.float32)

    monkeypatch.setattr(ssd_kernel_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        ssd_kernel_mod, "_pallas_backend", lambda prefer_mosaic_gpu=None: "triton"
    )
    monkeypatch.setattr(
        ssd_kernel_mod, "_gpu_supports_ssd_pallas_for_shape", lambda **_: True
    )

    def _unexpected_linear_scan(*args, **kwargs):
        raise AssertionError("linear scan fallback should be disabled")

    def _fake_ssd(q_arg, k_arg, v_arg, log_alpha_arg, h0_arg):
        del q_arg, k_arg, log_alpha_arg, h0_arg
        return jnp.full_like(v_arg, 5.0)

    monkeypatch.setattr(ssd_kernel_mod, "ssd_linear_scan", _unexpected_linear_scan)
    monkeypatch.setattr(ssd_kernel_mod, "_make_ssd", lambda: _fake_ssd)

    out = ssd(q, k, v, log_alpha)

    assert out.shape == v.shape
    assert jnp.all(out == 5.0)


def test_ssd_gpu_unsupported_config_raises_without_linear_scan_fallback(monkeypatch):
    batch = 1
    num_groups = 1
    num_heads = 1
    seq_len = 128
    dk = 64
    dv = 64

    key = jax.random.PRNGKey(4)
    q = jax.random.normal(key, (batch, num_groups, seq_len, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (batch, num_groups, seq_len, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (batch, num_heads, seq_len, dv), dtype=jnp.float32)
    log_alpha = jax.random.normal(key, (batch, num_heads, seq_len), dtype=jnp.float32)

    monkeypatch.setattr(ssd_kernel_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        ssd_kernel_mod, "_pallas_backend", lambda prefer_mosaic_gpu=None: "triton"
    )
    monkeypatch.setattr(
        ssd_kernel_mod, "_gpu_supports_ssd_pallas_for_shape", lambda **_: False
    )

    def _unexpected_linear_scan(*args, **kwargs):
        raise AssertionError("linear scan fallback should be disabled")

    monkeypatch.setattr(ssd_kernel_mod, "ssd_linear_scan", _unexpected_linear_scan)

    with pytest.raises(RuntimeError, match="Reference fallback on GPU is disabled"):
        ssd(q, k, v, log_alpha)


def test_ssd_gpu_kernel_failure_raises_without_linear_scan_fallback(monkeypatch):
    batch = 1
    num_groups = 1
    num_heads = 1
    seq_len = 128
    dk = 64
    dv = 64

    key = jax.random.PRNGKey(5)
    q = jax.random.normal(key, (batch, num_groups, seq_len, dk), dtype=jnp.float32)
    k = jax.random.normal(key, (batch, num_groups, seq_len, dk), dtype=jnp.float32)
    v = jax.random.normal(key, (batch, num_heads, seq_len, dv), dtype=jnp.float32)
    log_alpha = jax.random.normal(key, (batch, num_heads, seq_len), dtype=jnp.float32)

    monkeypatch.setattr(ssd_kernel_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        ssd_kernel_mod, "_pallas_backend", lambda prefer_mosaic_gpu=None: "triton"
    )
    monkeypatch.setattr(
        ssd_kernel_mod, "_gpu_supports_ssd_pallas_for_shape", lambda **_: True
    )

    def _unexpected_linear_scan(*args, **kwargs):
        raise AssertionError("linear scan fallback should be disabled")

    def _failing_ssd(*args, **kwargs):
        raise RuntimeError("pallas launch failed")

    monkeypatch.setattr(ssd_kernel_mod, "ssd_linear_scan", _unexpected_linear_scan)
    monkeypatch.setattr(ssd_kernel_mod, "_make_ssd", lambda: _failing_ssd)

    with pytest.raises(RuntimeError, match="pallas launch failed"):
        ssd(q, k, v, log_alpha)
