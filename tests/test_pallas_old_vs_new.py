"""A/B tests: primitive-based pallas kernels vs the frozen old_pallas path.

Run on a multi-GPU machine before deleting ``old_pallas``::

    pytest tests/test_pallas_old_vs_new.py -v

CPU runs cover mha (interpret mode) and kernel_mv; ssd/mamba/flash require an
accelerator and are skipped. Sharded comparisons additionally require >= 2
devices. Importing old_pallas swaps the custom_partitioning batching rule;
its __init__ restores the live general rule afterwards, so both paths behave
correctly in one process (old kernels are compared un-vmapped).
"""

import jax
import jax.numpy as jnp
import pytest

import probjax.nn.pallas_kernels as new_pk
from probjax.nn.pallas_kernels import old_pallas
from probjax.nn.pallas_kernels.kernels.attention import BlockSizes


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


def test_mha_forward_matches_old(mha_inputs):
    q, k, v = mha_inputs
    kwargs = dict(block_sizes=_block_sizes(q.shape[1]), interpret=True)
    new = new_pk.mha(q, k, v, **kwargs)
    old = old_pallas.mha(q, k, v, **kwargs)
    assert jnp.allclose(new, old, atol=1e-5)


def test_mha_grad_matches_old(mha_inputs):
    q, k, v = mha_inputs
    kwargs = dict(block_sizes=_block_sizes(q.shape[1]), interpret=True)

    def loss_new(q, k, v):
        return jnp.sum(new_pk.mha(q, k, v, **kwargs) ** 2)

    def loss_old(q, k, v):
        return jnp.sum(old_pallas.mha(q, k, v, **kwargs) ** 2)

    grads_new = jax.grad(loss_new, argnums=(0, 1, 2))(q, k, v)
    grads_old = jax.grad(loss_old, argnums=(0, 1, 2))(q, k, v)
    for g_new, g_old in zip(grads_new, grads_old):
        assert jnp.allclose(g_new, g_old, atol=1e-4)


def test_mha_jvp_matches_old(mha_inputs):
    q, k, v = mha_inputs
    kwargs = dict(
        block_sizes=_block_sizes(q.shape[1]), interpret=True, diff_mode="forward"
    )
    tangents = _qkv(jax.random.key(1))

    def f_new(q, k, v):
        return new_pk.mha(q, k, v, **kwargs)

    def f_old(q, k, v):
        return old_pallas.mha(q, k, v, **kwargs)

    out_new, t_new = jax.jvp(f_new, (q, k, v), tangents)
    out_old, t_old = jax.jvp(f_old, (q, k, v), tangents)
    assert jnp.allclose(out_new, out_old, atol=1e-5)
    # Interpret-mode JVP produces NaNs at some positions for BOTH paths
    # (pre-existing kernel quirk); require identical NaN masks and matching
    # finite values.
    assert jnp.array_equal(jnp.isnan(t_new), jnp.isnan(t_old))
    assert jnp.allclose(
        jnp.nan_to_num(t_new), jnp.nan_to_num(t_old), atol=1e-4
    )


def test_mha_vmap_matches_old_unvmapped(mha_inputs):
    # The new path supports vmap via the general CP rule + primitive batching;
    # compare against the old path applied per-slice.
    q, k, v = mha_inputs
    kwargs = dict(block_sizes=_block_sizes(q.shape[1]), interpret=True)
    qs = jnp.stack([q, q + 0.1])
    ks = jnp.stack([k, k - 0.1])
    vs = jnp.stack([v, v * 1.1])

    new = jax.vmap(lambda a, b, c: new_pk.mha(a, b, c, **kwargs))(qs, ks, vs)
    old = jnp.stack(
        [old_pallas.mha(qs[i], ks[i], vs[i], **kwargs) for i in range(2)]
    )
    assert jnp.allclose(new, old, atol=1e-5)


def test_kernel_mv_matches_old():
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (2, 32, 4))
    k = jax.random.normal(kk, (2, 48, 4))
    v = jax.random.normal(kv, (2, 48, 8))
    lengthscale = jnp.asarray([1.0])

    new = new_pk.rbf_kernel_mv(q, k, v, lengthscale, interpret=True)
    old = old_pallas.rbf_kernel_mv(q, k, v, lengthscale, interpret=True)
    assert jnp.allclose(new, old, atol=1e-5)

    g_new = jax.grad(lambda q: jnp.sum(new_pk.rbf_kernel_mv(q, k, v, lengthscale, interpret=True)))(q)
    g_old = jax.grad(lambda q: jnp.sum(old_pallas.rbf_kernel_mv(q, k, v, lengthscale, interpret=True)))(q)
    assert jnp.allclose(g_new, g_old, atol=1e-4)


# --------------------------------------------------------------------------
# Accelerator-only comparisons (run these on the GPU machine)
# --------------------------------------------------------------------------


def _requires_accelerator():
    if jax.default_backend() == "cpu":
        pytest.skip("requires an accelerator backend")


def test_ssd_matches_old_gpu():
    _requires_accelerator()
    key = jax.random.key(0)
    ks = jax.random.split(key, 4)
    q = jax.random.normal(ks[0], (2, 2, 128, 32))
    k = jax.random.normal(ks[1], (2, 2, 128, 32))
    v = jax.random.normal(ks[2], (2, 4, 128, 32))
    log_alpha = -jnp.abs(jax.random.normal(ks[3], (2, 4, 128)))

    new = new_pk.ssd(q, k, v, log_alpha)
    old = old_pallas.ssd(q, k, v, log_alpha)
    assert jnp.allclose(new, old, atol=1e-4)

    g_new = jax.grad(lambda v: jnp.sum(new_pk.ssd(q, k, v, log_alpha)))(v)
    g_old = jax.grad(lambda v: jnp.sum(old_pallas.ssd(q, k, v, log_alpha)))(v)
    assert jnp.allclose(g_new, g_old, atol=1e-3)


def test_mamba_matches_old_gpu():
    _requires_accelerator()
    key = jax.random.key(0)
    ks = jax.random.split(key, 6)
    batch, seq, dim, state = 2, 128, 64, 16
    x = jax.random.normal(ks[0], (batch, seq, dim))
    a = -jnp.abs(jax.random.normal(ks[1], (state, dim)))
    b = jax.random.normal(ks[2], (batch, seq, state))
    c = jax.random.normal(ks[3], (batch, seq, state))
    delta = jax.nn.softplus(jax.random.normal(ks[4], (batch, seq, dim)))
    d = jax.random.normal(ks[5], (1, dim))

    new = new_pk.compute_mamba_scan(x, a, b, c, delta, d)
    old = old_pallas.compute_mamba_scan(x, a, b, c, delta, d)
    assert jnp.allclose(new, old, atol=1e-4)


def test_mha_sharded_no_allgather_gpu():
    _requires_accelerator()
    if jax.device_count() < 2:
        pytest.skip("requires >= 2 devices")
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = jax.make_mesh(
        (2,), ("data",), axis_types=(jax.sharding.AxisType.Auto,)
    )
    q, k, v = _qkv(jax.random.key(0))
    sharding = NamedSharding(mesh, P("data", None, None, None))
    qs, ks_, vs = (jax.device_put(x, sharding) for x in (q, k, v))
    kwargs = dict(block_sizes=_block_sizes(q.shape[1]))

    fn = jax.jit(lambda a, b, c: new_pk.mha(a, b, c, **kwargs))
    hlo = fn.lower(qs, ks_, vs).compile().as_text()
    assert "all-gather" not in hlo

    new = fn(qs, ks_, vs)
    old = old_pallas.mha(q, k, v, **kwargs)
    assert jnp.allclose(jax.device_get(new), old, atol=1e-4)
