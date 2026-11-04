"""Native GQA/MQA in the attention kernels and the split-KV decode forward.

CPU runs cover the XLA path, the Triton kernels in Pallas interpret mode (the
``h // group`` KV indexing and the dK/dV group reduction) and the split-KV
kernel in interpret mode. GPU-only tests (skipped on CPU) check the compiled
kernels and the automatic dispatch.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.layers.attention import dot_product_attention, flex_attention
from probjax.nn.pallas_kernels import CausalMask
from probjax.nn.pallas_kernels.kernels.attention import BlockSizes, mha, use_splitkv
from probjax.nn.pallas_kernels.kernels.attention_splitkv import (
    splitkv_attention,
    splitkv_forward,
)


def _requires_accelerator():
    if jax.default_backend() == "cpu":
        pytest.skip("requires an accelerator backend")


def _inputs(b, lq, lkv, hq, hkv, d, dtype=jnp.float32, seed=0):
    ks = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(ks[0], (b, lq, hq, d), jnp.float32).astype(dtype)
    k = jax.random.normal(ks[1], (b, lkv, hkv, d), jnp.float32).astype(dtype)
    v = jax.random.normal(ks[2], (b, lkv, hkv, d), jnp.float32).astype(dtype)
    g = jax.random.normal(ks[3], (b, lq, hq, d), jnp.float32)
    return q, k, v, g


def _reference(q, k, v, causal=False):
    """fp32 dense attention with ``jnp.repeat`` GQA semantics."""
    rep = q.shape[2] // k.shape[2]
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    k, v = (jnp.repeat(x, rep, axis=2) for x in (k, v))
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision="highest")
    s = s / math.sqrt(q.shape[-1])
    if causal:
        lq, lk = s.shape[-2:]
        s = jnp.where(jnp.tril(jnp.ones((lq, lk), bool)), s, -1e30)
    return jnp.einsum(
        "bhqk,bkhd->bqhd", jax.nn.softmax(s, axis=-1), v, precision="highest"
    )


def _tol(cpu_tol: float) -> float:
    # Off CPU, fp32 matmuls default to TF32 (relative error ~1e-4..1e-3).
    return cpu_tol if jax.default_backend() == "cpu" else 2e-3


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _value_and_grads(fn, q, k, v, g):
    out, vjp = jax.vjp(fn, q, k, v)
    return (out, *vjp(g.astype(out.dtype)))


def _assert_close_all(got, want, tol):
    for name, a, b in zip(("out", "dq", "dk", "dv"), got, want, strict=True):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert _rel(a, b) < tol, (name, _rel(a, b))


# ---------------------------------------------------------------------------
# CPU-runnable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_flex_gqa_matches_repeat_reference(hkv, causal):
    q, k, v, g = _inputs(2, 40, 40, 4, hkv, 16)
    mask = CausalMask() if causal else None
    got = _value_and_grads(
        lambda q, k, v: flex_attention(q, k, v, mask=mask, enable_gqa=True),
        q,
        k,
        v,
        g,
    )
    want = _value_and_grads(lambda q, k, v: _reference(q, k, v, causal), q, k, v, g)
    tol = _tol(1e-5)
    _assert_close_all(got, want, tol)


@pytest.mark.parametrize("backward_pass_impl", ["triton_fused", "triton_split"])
@pytest.mark.parametrize("hkv", [1, 2])
def test_mha_interpret_gqa_indexes_shared_kv_head(backward_pass_impl, hkv):
    """Pallas interpret mode runs the Triton kernels' index maps on CPU."""
    q, k, v, g = _inputs(1, 32, 32, 4, hkv, 16, seed=hkv)
    blocks = BlockSizes(
        block_q=16,
        block_k=16,
        block_q_dkv=16,
        block_kv_dkv=16,
        block_q_dq=16,
        block_kv_dq=16,
    )

    def fn(q, k, v):
        return mha(
            q,
            k,
            v,
            sm_scale=1 / 4,
            block_sizes=blocks,
            backward_pass_impl=backward_pass_impl,
            interpret=True,
        )

    got = _value_and_grads(fn, q, k, v, g)
    want = _value_and_grads(_reference, q, k, v, g)
    _assert_close_all(got, want, _tol(1e-4))


def test_mha_interpret_gqa_jvp():
    q, k, v, _ = _inputs(1, 32, 32, 4, 2, 16, seed=3)
    tq, tk, tv, _ = _inputs(1, 32, 32, 4, 2, 16, seed=4)
    blocks = BlockSizes(16, 16, 16, 16, 16, 16)
    _, got = jax.jvp(
        lambda q, k, v: mha(q, k, v, sm_scale=0.25, block_sizes=blocks, interpret=True),
        (q, k, v),
        (tq, tk, tv),
    )
    _, want = jax.jvp(_reference, (q, k, v), (tq, tk, tv))
    assert _rel(got, want) < _tol(1e-4)


def test_mha_rejects_bad_head_ratio():
    q, k, v, _ = _inputs(1, 16, 16, 6, 4, 16)
    with pytest.raises(ValueError, match="multiple"):
        mha(q, k, v, interpret=True)


def test_flex_gqa_saves_no_repeated_kv():
    """Residuals keep K/V at Hkv heads (no Hq-head copy)."""
    b, lq, lkv, hq, hkv, d = 1, 24, 40, 8, 2, 16
    q, k, v, _ = _inputs(b, lq, lkv, hq, hkv, d)
    _, vjp_fn = jax.vjp(
        lambda q, k, v: flex_attention(q, k, v, enable_gqa=True), q, k, v
    )
    shapes = [x.shape for x in jax.tree_util.tree_leaves(vjp_fn) if hasattr(x, "shape")]
    assert all(s[1:3] != (lkv, hq) for s in shapes if len(s) == 4), shapes
    jaxpr = jax.make_jaxpr(lambda q, k, v: flex_attention(q, k, v, enable_gqa=True))(
        q, k, v
    )
    for eqn in jaxpr.eqns:
        for out in eqn.outvars:
            shape = getattr(out.aval, "shape", ())
            assert tuple(shape) != (b, lkv, hq, d), eqn.primitive


def test_dot_product_attention_gqa_matches_flax_semantics():
    q, k, v, _ = _inputs(2, 20, 20, 6, 2, 8)
    got = dot_product_attention(q, k, v)
    assert _rel(got, _reference(q, k, v)) < _tol(1e-5)


@pytest.mark.parametrize(
    "b,lq,lkv,hq,hkv,d,splits",
    [
        (1, 1, 200, 4, 4, 16, None),
        (2, 3, 130, 4, 2, 16, 3),
        (1, 4, 96, 8, 1, 32, 2),
        (1, 2, 50, 2, 1, 24, 1),
    ],
)
def test_splitkv_forward_interpret(b, lq, lkv, hq, hkv, d, splits):
    q, k, v, _ = _inputs(b, lq, lkv, hq, hkv, d, seed=lkv)
    out, lse = splitkv_forward(
        q,
        k,
        v,
        sm_scale=1 / math.sqrt(d),
        num_splits=splits,
        block_k=16,
        interpret=True,
        return_lse=True,
    )
    assert out.shape == q.shape
    assert _rel(out, _reference(q, k, v)) < _tol(1e-5)
    kr = jnp.repeat(k, hq // hkv, axis=2)
    s = jnp.einsum("bqhd,bkhd->bhqk", q, kr, precision="highest")
    want_lse = jax.nn.logsumexp(s / math.sqrt(d), axis=-1)
    np.testing.assert_allclose(lse, want_lse, rtol=_tol(1e-5), atol=_tol(1e-5))


def test_splitkv_attention_differentiates_through_fallback():
    q, k, v, g = _inputs(1, 2, 64, 4, 2, 16, seed=7)

    def fn(q, k, v):
        return splitkv_attention(
            q,
            k,
            v,
            sm_scale=0.25,
            fallback=_reference,
            block_k=16,
            num_splits=4,
            interpret=True,
        )

    got = _value_and_grads(fn, q, k, v, g)
    want = _value_and_grads(_reference, q, k, v, g)
    _assert_close_all(got, want, _tol(1e-5))


def test_use_splitkv_only_for_exact_cases():
    q, k, _, _ = _inputs(1, 1, 64, 4, 2, 16)
    assert not use_splitkv(q, k, split_kv=False)
    assert not use_splitkv(q, k, mask=CausalMask(), split_kv=True)
    assert not use_splitkv(q, k, dropout_rate=0.1, split_kv=True)
    assert not use_splitkv(q, k, interpret=True, split_kv=True)
    if jax.default_backend() == "cpu":
        # CPU uses the XLA path, which needs no split-KV.
        assert not use_splitkv(q, k, split_kv=True)


@pytest.mark.parametrize("lq,lkv", [(1, 300), (4, 1000)])
def test_cpu_decode_shapes(lq, lkv):
    q, k, v, g = _inputs(2, lq, lkv, 8, 2, 32, seed=lq)
    got = _value_and_grads(
        lambda q, k, v: flex_attention(q, k, v, enable_gqa=True), q, k, v, g
    )
    want = _value_and_grads(_reference, q, k, v, g)
    tol = _tol(1e-5)
    _assert_close_all(got, want, tol)


# ---------------------------------------------------------------------------
# GPU-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("backward_pass_impl", ["triton_fused", "triton_split"])
@pytest.mark.parametrize("hkv,causal", [(1, False), (2, True), (4, False)])
def test_gpu_gqa_native_matches_repeat(dtype, backward_pass_impl, hkv, causal):
    _requires_accelerator()
    q, k, v, g = _inputs(2, 200, 333, 8, hkv, 64, dtype, seed=hkv)
    mask = CausalMask() if causal else None

    def native(q, k, v):
        return flex_attention(
            q,
            k,
            v,
            mask=mask,
            enable_gqa=True,
            backward_pass_impl=backward_pass_impl,
        )

    def repeated(q, k, v):
        rep = q.shape[2] // k.shape[2]
        k, v = (jnp.repeat(x, rep, axis=2) for x in (k, v))
        return flex_attention(q, k, v, mask=mask, backward_pass_impl=backward_pass_impl)

    got = _value_and_grads(jax.jit(native), q, k, v, g)
    rep = _value_and_grads(jax.jit(repeated), q, k, v, g)
    want = _value_and_grads(lambda q, k, v: _reference(q, k, v, causal), q, k, v, g)
    # Same kernels, same per-head math: native equals the repeat path up to
    # the order of the group sum.
    _assert_close_all(got, rep, 1e-6 if dtype == jnp.float32 else 1e-2)
    _assert_close_all(got, want, 2e-3 if dtype == jnp.float32 else 1e-2)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("lq,lkv,hkv", [(1, 4096, 8), (4, 3000, 2), (16, 2048, 1)])
def test_gpu_splitkv_dispatch(dtype, lq, lkv, hkv):
    _requires_accelerator()
    q, k, v, g = _inputs(1, lq, lkv, 8, hkv, 64, dtype, seed=lq)
    auto = jax.jit(lambda q, k, v: flex_attention(q, k, v, enable_gqa=True))
    regular = jax.jit(
        lambda q, k, v: flex_attention(q, k, v, enable_gqa=True, split_kv=False)
    )
    assert "mha_splitkv_forward" in auto.lower(q, k, v).as_text()
    assert "mha_splitkv_forward" not in regular.lower(q, k, v).as_text()
    tol = 2e-3 if dtype == jnp.float32 else 1e-2
    assert _rel(auto(q, k, v), _reference(q, k, v)) < tol
    # Differentiation runs the regular kernels.
    got = _value_and_grads(
        lambda q, k, v: flex_attention(q, k, v, enable_gqa=True), q, k, v, g
    )
    want = _value_and_grads(regular, q, k, v, g)
    _assert_close_all(got, want, 1e-6 if dtype == jnp.float32 else 1e-2)


def test_gpu_splitkv_jvp_uses_regular_path():
    _requires_accelerator()
    q, k, v, _ = _inputs(1, 2, 2048, 4, 2, 64)
    primal, tangent = jax.jvp(
        lambda q, k, v: mha(q, k, v, sm_scale=0.125), (q, k, v), (q, k, v)
    )
    _, want = jax.jvp(
        lambda q, k, v: mha(q, k, v, sm_scale=0.125, split_kv=False),
        (q, k, v),
        (q, k, v),
    )
    assert _rel(tangent, want) < 1e-6
    assert _rel(primal, _reference(q, k, v)) < 2e-3


def test_gpu_dot_product_attention_routes_gqa_to_kernel():
    _requires_accelerator()
    q, k, v, _ = _inputs(2, 256, 256, 8, 2, 64)
    fn = jax.jit(lambda q, k, v: dot_product_attention(q, k, v))
    text = fn.lower(q, k, v).as_text()
    assert "mha_forward" in text
    assert _rel(fn(q, k, v), _reference(q, k, v)) < 2e-3
