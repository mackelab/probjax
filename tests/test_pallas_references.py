"""Primitive-based kernels against independent mathematical references.

CPU covers interpreted attention and kernel products. Accelerator tests retain
SSD, Mamba and multi-device sharding coverage without a second implementation.
"""

import jax
import jax.numpy as jnp
import pytest

import probjax.nn.pallas_kernels as pk
from probjax.nn.pallas_kernels.kernels.attention import BlockSizes


def _mha_reference(q, k, v, **kwargs):
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * kwargs.get("sm_scale", 1.0)
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, axis=-1), v)


def _rbf_reference(q, k, v, lengthscale, **kwargs):
    return pk.rbf_kernel_mv_naive(q, k, v, lengthscale)


def _ssd_reference(q, k, v, log_alpha):
    from tests.test_ssd_ad import ssd_reference

    h0 = jnp.zeros((v.shape[0], v.shape[1], q.shape[-1], v.shape[-1]), v.dtype)
    return ssd_reference(q, k, v, log_alpha, h0)


def _mamba_reference(*args):
    from probjax.nn.pallas_kernels.kernels.mamba import _mamba_scan_reference

    return _mamba_scan_reference(*args)


def _qkv(key, batch=2, seq=64, heads=4, head_dim=16):
    kq, kk, kv = jax.random.split(key, 3)
    shape = (batch, seq, heads, head_dim)
    return (
        jax.random.normal(kq, shape),
        jax.random.normal(kk, shape),
        jax.random.normal(kv, shape),
    )


def _block_sizes(seq):
    size = min(seq, 32)
    return BlockSizes(
        block_q=size,
        block_k=size,
        block_q_dkv=size,
        block_kv_dkv=size,
        block_q_dq=size,
        block_kv_dq=size,
    )


@pytest.fixture
def mha_inputs():
    return _qkv(jax.random.key(0))


def test_mha_forward_matches_reference(mha_inputs):
    q, k, v = mha_inputs
    kwargs = dict(block_sizes=_block_sizes(q.shape[1]), interpret=True)
    new = pk.mha(q, k, v, **kwargs)
    reference = _mha_reference(q, k, v, **kwargs)
    assert jnp.allclose(new, reference, atol=1e-5)


def test_mha_grad_matches_reference(mha_inputs):
    q, k, v = mha_inputs
    kwargs = dict(block_sizes=_block_sizes(q.shape[1]), interpret=True)

    def loss_new(q, k, v):
        return jnp.sum(pk.mha(q, k, v, **kwargs) ** 2)

    def loss_reference(q, k, v):
        return jnp.sum(_mha_reference(q, k, v, **kwargs) ** 2)

    grads_new = jax.grad(loss_new, argnums=(0, 1, 2))(q, k, v)
    grads_reference = jax.grad(loss_reference, argnums=(0, 1, 2))(q, k, v)
    for g_new, g_reference in zip(grads_new, grads_reference):
        assert jnp.allclose(g_new, g_reference, atol=1e-4)


def test_mha_jvp_matches_reference(mha_inputs):
    q, k, v = mha_inputs
    kwargs = dict(
        block_sizes=_block_sizes(q.shape[1]), interpret=True, diff_mode="forward"
    )
    tangents = _qkv(jax.random.key(1))

    def f_new(q, k, v):
        return pk.mha(q, k, v, **kwargs)

    def f_reference(q, k, v):
        return _mha_reference(q, k, v, **kwargs)

    out_new, t_new = jax.jvp(f_new, (q, k, v), tangents)
    out_reference, t_reference = jax.jvp(f_reference, (q, k, v), tangents)
    assert jnp.allclose(out_new, out_reference, atol=1e-5)
    assert jnp.all(jnp.isfinite(t_new))
    assert jnp.allclose(t_new, t_reference, atol=1e-4, rtol=1e-4)


def test_mha_vmap_matches_reference(mha_inputs):
    # Compare primitive batching with independently evaluated reference slices.
    q, k, v = mha_inputs
    kwargs = dict(block_sizes=_block_sizes(q.shape[1]), interpret=True)
    qs = jnp.stack([q, q + 0.1])
    ks = jnp.stack([k, k - 0.1])
    vs = jnp.stack([v, v * 1.1])

    new = jax.vmap(lambda a, b, c: pk.mha(a, b, c, **kwargs))(qs, ks, vs)
    reference = jnp.stack([
        _mha_reference(qs[i], ks[i], vs[i], **kwargs) for i in range(2)
    ])
    assert jnp.allclose(new, reference, atol=1e-5)


def test_kernel_mv_matches_reference():
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (2, 32, 4))
    k = jax.random.normal(kk, (2, 48, 4))
    v = jax.random.normal(kv, (2, 48, 8))
    lengthscale = jnp.asarray([1.0])

    new = pk.rbf_kernel_mv(q, k, v, lengthscale, interpret=True)
    reference = _rbf_reference(q, k, v, lengthscale, interpret=True)
    assert jnp.allclose(new, reference, atol=1e-5)

    g_new = jax.grad(
        lambda q: jnp.sum(pk.rbf_kernel_mv(q, k, v, lengthscale, interpret=True))
    )(q)
    g_reference = jax.grad(
        lambda q: jnp.sum(_rbf_reference(q, k, v, lengthscale, interpret=True))
    )(q)
    assert jnp.allclose(g_new, g_reference, atol=1e-4)


# --------------------------------------------------------------------------
# Accelerator-only comparisons (run these on the GPU machine)
# --------------------------------------------------------------------------


def _requires_accelerator():
    if jax.default_backend() == "cpu":
        pytest.skip("requires an accelerator backend")


def test_ssd_matches_reference_gpu():
    _requires_accelerator()
    key = jax.random.key(0)
    ks = jax.random.split(key, 4)
    q = jax.random.normal(ks[0], (2, 2, 256, 64))
    k = jax.random.normal(ks[1], (2, 2, 256, 64))
    v = jax.random.normal(ks[2], (2, 4, 256, 64))
    log_alpha = -jnp.abs(jax.random.normal(ks[3], (2, 4, 256)))

    new = pk.ssd(q, k, v, log_alpha)
    reference = _ssd_reference(q, k, v, log_alpha)
    assert jnp.allclose(new, reference, atol=1e-4)

    g_new = jax.grad(lambda v: jnp.sum(pk.ssd(q, k, v, log_alpha)))(v)
    g_reference = jax.grad(lambda v: jnp.sum(_ssd_reference(q, k, v, log_alpha)))(v)
    assert jnp.allclose(g_new, g_reference, atol=1e-3)


def test_mamba_matches_reference_gpu():
    _requires_accelerator()
    key = jax.random.key(0)
    ks = jax.random.split(key, 6)
    batch, seq, dim, state = 2, 128, 128, 16
    x = jax.random.normal(ks[0], (batch, seq, dim))
    a = -jnp.abs(jax.random.normal(ks[1], (state, dim)))
    b = jax.random.normal(ks[2], (batch, seq, state))
    c = jax.random.normal(ks[3], (batch, seq, state))
    delta = jax.nn.softplus(jax.random.normal(ks[4], (batch, seq, dim)))
    d = jax.random.normal(ks[5], (1, dim))

    new = pk.compute_mamba_scan(x, a, b, c, delta, d, seq_tile_size=32, dim_tile_size=128)
    reference = _mamba_reference(x, a, b, c, delta, d)
    assert jnp.allclose(new, reference, atol=1e-4)


@pytest.mark.mesh
@pytest.mark.parametrize("axis", [0, 2])
def test_mha_sharded_no_allgather_mesh(axis):
    if jax.device_count() < 2:
        pytest.skip("requires >= 2 devices")
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = jax.make_mesh((2,), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
    q, k, v = _qkv(jax.random.key(0))
    axes = [None] * 4
    axes[axis] = "data"
    sharding = NamedSharding(mesh, P(*axes))
    qs, ks_, vs = (jax.device_put(x, sharding) for x in (q, k, v))
    kwargs = dict(
        block_sizes=_block_sizes(q.shape[1]), interpret=jax.default_backend() == "cpu"
    )

    fn = jax.jit(lambda a, b, c: pk.mha(a, b, c, **kwargs))
    hlo = fn.lower(qs, ks_, vs).compile().as_text()
    assert "all-gather" not in hlo

    new = fn(qs, ks_, vs)
    reference = _mha_reference(q, k, v, **kwargs)
    assert jnp.allclose(jax.device_get(new), reference, atol=1e-4)
    gradient = jax.jit(
        jax.grad(
            lambda a, b, c: jnp.sum(pk.mha(a, b, c, **kwargs) ** 2), argnums=(0, 1, 2)
        )
    )
    actual_grad = gradient(qs, ks_, vs)
    expected_grad = jax.grad(
        lambda a, b, c: jnp.sum(_mha_reference(a, b, c) ** 2), argnums=(0, 1, 2)
    )(q, k, v)
    for actual, expected in zip(actual_grad, expected_grad):
        assert jnp.allclose(actual, expected, atol=2e-4, rtol=1e-4)
    tangent = jax.jit(
        lambda a, b, c: jax.jvp(
            lambda x, y, z: pk.mha(x, y, z, **kwargs), (a, b, c), (a, b, c)
        )[1]
    )(qs, ks_, vs)
    expected_tangent = jax.jvp(_mha_reference, (q, k, v), (q, k, v))[1]
    assert jnp.allclose(tangent, expected_tangent, atol=2e-4, rtol=1e-4)


def test_ssd_jvp_matches_reference_gpu():
    # Forward-mode SSD (composite-of-forwards jvp primitive) vs jax.jvp of
    # the pure-JAX reference, including the fused backward through transpose.
    _requires_accelerator()
    from tests.test_ssd_ad import ssd_reference

    key = jax.random.key(0)
    ks = jax.random.split(key, 10)
    q = jax.random.normal(ks[0], (2, 2, 256, 64))
    k = jax.random.normal(ks[1], (2, 2, 256, 64))
    v = jax.random.normal(ks[2], (2, 4, 256, 64))
    la = -jnp.abs(jax.random.normal(ks[3], (2, 4, 256))) * 0.3
    h0 = jnp.zeros((2, 4, 64, 64))
    primals = (q, k, v, la, h0)
    tangents = tuple(
        jax.random.normal(kx, a.shape) * 0.5 for kx, a in zip(ks[4:], primals)
    )

    import importlib

    ssd_mod = importlib.import_module("probjax.nn.pallas_kernels.kernels.ssd")
    _, t_new = jax.jvp(ssd_mod._ssd_op, primals, tangents)
    _, t_ref = jax.jvp(ssd_reference, primals, tangents)
    assert jnp.allclose(t_new, t_ref, atol=1e-2)

    g_new = jax.grad(lambda v: jnp.sum(ssd_mod._ssd_op(q, k, v, la, h0)))(v)
    g_reference = jax.grad(lambda v: jnp.sum(_ssd_reference(q, k, v, la)))(v)
    assert jnp.allclose(g_new, g_reference, atol=1e-3)


@pytest.mark.parametrize('save_residuals', [False, True])
def test_flash3_public_forward_fallback(monkeypatch, save_residuals):
    import importlib

    flash = importlib.import_module(
        'probjax.nn.pallas_kernels.kernels.flash_attention3'
    )

    def public(q, k, v, *, config, save_residuals):
        assert config == 'config'
        out = q * k + v
        return (out, (jnp.ones_like(out),)) if save_residuals else out

    monkeypatch.setattr(flash, '_attention_forward_impl', None)
    monkeypatch.setattr(flash, '_attention_impl', public)
    actual = flash._run_flash_forward_raw(
        jnp.array(2.0),
        jnp.array(3.0),
        jnp.array(4.0),
        config='config',
        save_residuals=save_residuals,
        use_pipeline_emitter=False,
    )
    if save_residuals:
        out, lse = actual
        assert lse == 1.0
    else:
        out = actual
    assert out == 10.0


def test_flash3_public_backward_fallback(monkeypatch):
    import importlib

    flash = importlib.import_module(
        'probjax.nn.pallas_kernels.kernels.flash_attention3'
    )

    def public(q, k, v, *, config, save_residuals):
        assert config == 'config' and not save_residuals
        return q * k + v

    monkeypatch.setattr(flash, '_attention_backward_impl', None)
    monkeypatch.setattr(flash, '_attention_impl', public)
    grads = flash._run_flash_backward_raw(
        jnp.array(5.0),
        jnp.array(2.0),
        jnp.array(3.0),
        jnp.array(4.0),
        jnp.array(10.0),
        jnp.array(1.0),
        config='config',
    )
    for actual, expected in zip(grads, (15.0, 10.0, 5.0)):
        assert actual == expected


@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('rate', [0.0, 0.2])
def test_multitile_attention_jvp_with_bias_mask_dropout(causal, rate):
    from probjax.nn.pallas_kernels import CausalMask, DenseBias

    q, k, v = _qkv(jax.random.key(30), batch=1, seq=64, heads=2)
    tangents = _qkv(jax.random.key(31), batch=1, seq=64, heads=2)
    bias = jax.random.normal(jax.random.key(32), (1, 2, 64, 64)) * 0.1
    rng = jax.random.key(33)
    dropped = jax.random.bernoulli(rng, rate, (1, 2, 64, 64))

    def reference(q, k, v):
        logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) * 0.25 + bias
        if causal:
            logits = jnp.where(
                jnp.arange(64)[:, None] >= jnp.arange(64)[None, :], logits, -jnp.inf
            )
        weights = jax.nn.softmax(logits, axis=-1)
        weights = jnp.where(dropped, 0.0, weights / (1 - rate))
        return jnp.einsum('bhqk,bkhd->bqhd', weights, v)

    def kernel(q, k, v):
        return pk.mha(
            q,
            k,
            v,
            sm_scale=0.25,
            mask=CausalMask() if causal else None,
            bias=DenseBias(bias),
            rng=rng,
            dropout_rate=rate,
            dropout_impl='materialize',
            block_sizes=_block_sizes(64),
            interpret=True,
            diff_mode='forward',
        )

    actual = jax.jit(
        lambda q, k, v, dq, dk, dv: jax.jvp(kernel, (q, k, v), (dq, dk, dv))
    )(q, k, v, *tangents)
    expected = jax.jvp(reference, (q, k, v), tangents)
    for x, y in zip(actual, expected):
        assert jnp.all(jnp.isfinite(x))
        assert jnp.allclose(x, y, atol=1e-5, rtol=2e-4)


@pytest.mark.parametrize("seq,dim", [(32, 128), (8, 256)])
def test_mamba_gpu_multitile_dispatch_and_gradients(monkeypatch, seq, dim):
    """GPU tile dependencies must not rely on ordering separate programs."""
    import importlib

    mamba = importlib.import_module("probjax.nn.pallas_kernels.kernels.mamba")
    keys = jax.random.split(jax.random.key(31), 6)
    args = (
        jax.random.normal(keys[0], (1, seq, dim)) * 0.2,
        -jax.nn.softplus(jax.random.normal(keys[1], (16, dim))),
        jax.random.normal(keys[2], (1, seq, 16)) * 0.2,
        jax.random.normal(keys[3], (1, seq, 16)) * 0.2,
        jax.nn.softplus(jax.random.normal(keys[4], (1, seq, dim))),
        jax.random.normal(keys[5], (1, dim)),
    )
    reference = mamba._mamba_scan_reference(*args)
    reference_grads = jax.grad(
        lambda *xs: jnp.sum(mamba._mamba_scan_reference(*xs) ** 2),
        argnums=tuple(range(6)),
    )(*args)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(mamba, "_pallas_backend", lambda: "triton")

    def forbidden(*args, **kwargs):
        raise AssertionError("multi-tile GPU call entered the unordered carry kernel")

    monkeypatch.setattr(mamba, "_mamba_scan_op", forbidden)

    def scan(*xs):
        return mamba.compute_mamba_scan(*xs, seq_tile_size=8, dim_tile_size=128)

    assert jnp.allclose(scan(*args), reference, rtol=2e-5, atol=2e-5)
    actual_grads = jax.grad(lambda *xs: jnp.sum(scan(*xs) ** 2), argnums=tuple(range(6)))(*args)
    for actual, expected in zip(actual_grads, reference_grads, strict=True):
        assert jnp.all(jnp.isfinite(actual))
        assert jnp.allclose(actual, expected, rtol=2e-4, atol=2e-4)
