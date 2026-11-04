"""GPU attention fast paths: flex padding, ``dot_product_attention`` dispatch,
and the small-SMEM tile heuristic of the Pallas ``mha`` kernel."""

from __future__ import annotations

import importlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import probjax.nn.layers.attention as attn_layers
from probjax.nn.layers.attention import (
    _flex_padded_lengths,
    _pad_positional_leaf,
    _use_fused_gpu_attention,
    dot_product_attention,
    flex_attention,
)
from probjax.nn.pallas_kernels import (
    BlockSizes,
    CausalMask,
    KVLenMask,
    KeyPaddingMask,
    LocalWindowMask,
)

attn_kernels = importlib.import_module("probjax.nn.pallas_kernels.kernels.attention")


def _requires_accelerator():
    if jax.default_backend() == "cpu":
        pytest.skip("requires an accelerator backend")


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _reference(q, k, v, causal=False, kv_valid=None, window=None, scale=None):
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k, precision="highest") * scale
    qi = jnp.arange(q.shape[1])[:, None]
    ki = jnp.arange(k.shape[1])[None, :]
    m = jnp.ones((q.shape[1], k.shape[1]), bool)
    if causal:
        m &= ki <= qi
    if window is not None:
        m &= (ki >= qi - window[0]) & (ki <= qi + window[1])
    m = m[None, None]
    if kv_valid is not None:
        m = m & kv_valid[:, None, None, :]
    s = jnp.where(m, s, -1e30)
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(s, -1), v, precision="highest")


def _inputs(b, lq, lkv, h, d, dtype=jnp.float32, seed=0):
    ks = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(ks[0], (b, lq, h, d), dtype)
    k = jax.random.normal(ks[1], (b, lkv, h, d), dtype)
    v = jax.random.normal(ks[2], (b, lkv, h, d), dtype)
    g = jax.random.normal(ks[3], (b, lq, h, d), jnp.float32)
    return q, k, v, g


# ---------------------------------------------------------------------------
# flex_attention padding (H2): pad to the tile multiple, not the next pow2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lq,lkv,expected",
    [
        (600, 600, (640, 640)),
        (1500, 1500, (1536, 1536)),
        (3000, 3000, (3072, 3072)),
        (1024, 1024, (1024, 1024)),
        (5, 5, (16, 16)),
        (40, 40, (64, 64)),
        (100, 3000, (128, 3072)),
        # Fused backward would need non-pow2 tiles -> legacy pow2 padding.
        (600, 200, (1024, 256)),
        (130, 1000, (256, 1024)),
    ],
)
def test_flex_padded_lengths(lq, lkv, expected):
    q_pad, kv_pad, bs = _flex_padded_lengths(lq, lkv, 128, 128, 64, 64, 64, 64, "auto")
    assert (q_pad, kv_pad) == expected
    for tile, n in [
        (bs.block_q_dkv, q_pad),
        (bs.block_kv_dkv, kv_pad),
        (bs.block_q_dq, q_pad),
        (bs.block_kv_dq, kv_pad),
    ]:
        tile = min(tile, n)
        assert tile >= 16 and tile & (tile - 1) == 0 and n % tile == 0


def test_pad_positional_leaf():
    x = jnp.ones((2, 600), bool)
    assert _pad_positional_leaf(x, 600, 640, 600, 640).shape == (2, 640)
    assert _pad_positional_leaf(jnp.ones((2, 7)), 600, 640, 600, 640).shape == (2, 7)
    assert _pad_positional_leaf(jnp.array(3), 600, 640, 600, 640).shape == ()


@pytest.mark.parametrize(
    "lq,lkv,causal",
    [(600, 600, True), (1500, 1500, False), (600, 200, False), (40, 40, True)],
)
def test_flex_ragged_lengths_gpu(lq, lkv, causal):
    _requires_accelerator()
    q, k, v, g = _inputs(2, lq, lkv, 2, 64, seed=lq + lkv)
    mask = CausalMask() if causal else None

    def f(q, k, v):
        return flex_attention(q, k, v, mask=mask)

    def r(q, k, v):
        return _reference(q, k, v, causal=causal)

    assert _rel(jax.jit(f)(q, k, v), r(q, k, v)) < 5e-3
    gf = jax.grad(lambda *a: jnp.sum(f(*a) * g), (0, 1, 2))(q, k, v)
    gr = jax.grad(lambda *a: jnp.sum(r(*a) * g), (0, 1, 2))(q, k, v)
    for a, b in zip(gf, gr, strict=True):
        assert _rel(a, b) < 1e-2


def test_flex_ragged_stateful_mask_gpu():
    """Stateful mask data is not padded; the kernel still masks correctly."""
    _requires_accelerator()
    q, k, v, g = _inputs(2, 600, 600, 2, 64, seed=3)
    lens = jnp.array([600, 300])
    valid = jnp.arange(600)[None] < lens[:, None]
    out = jax.jit(lambda q, k, v: flex_attention(q, k, v, mask=KVLenMask(lens)))(
        q, k, v
    )
    assert _rel(out, _reference(q, k, v, kv_valid=valid)) < 5e-3


# ---------------------------------------------------------------------------
# dot_product_attention dispatch (H3)
# ---------------------------------------------------------------------------


def _dispatch_ok(q, mask=None, **kw):
    args = dict(
        mask=mask,
        bias=None,
        dropout_rate=0.0,
        deterministic=True,
        module=None,
        precision=None,
    )
    args.update(kw)
    return _use_fused_gpu_attention(q, q, q, **args)


def test_dispatch_predicate(monkeypatch):
    monkeypatch.setattr(attn_layers.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(attn_kernels, "_query_triton_smem_limit", lambda: 101376)
    q = jnp.zeros((2, 256, 4, 32))
    assert _dispatch_ok(q)
    assert _dispatch_ok(q, mask=CausalMask())
    assert _dispatch_ok(q, mask=LocalWindowMask(8, 8))
    assert _dispatch_ok(jnp.zeros((3, 2, 256, 4, 32)))
    # Exact fallbacks.
    assert not _dispatch_ok(q, mask=jnp.ones((256, 256), bool))
    assert not _dispatch_ok(q, mask=KeyPaddingMask(jnp.array([3, 4])))
    assert not _dispatch_ok(q, bias=jnp.zeros((256, 256)))
    assert not _dispatch_ok(q, dropout_rate=0.1, deterministic=False)
    assert _dispatch_ok(q, dropout_rate=0.1, deterministic=True)
    assert not _dispatch_ok(q, module=object())
    assert not _dispatch_ok(q, precision="highest")
    assert not _dispatch_ok(jnp.zeros((2, 32, 4, 32)))  # too short for the kernel
    assert not _dispatch_ok(jnp.zeros((2, 256, 4, 32), jnp.int32))
    # No NVIDIA device / not a GPU backend.
    monkeypatch.setattr(attn_kernels, "_query_triton_smem_limit", lambda: None)
    assert not _dispatch_ok(q)
    monkeypatch.setattr(attn_layers.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(attn_kernels, "_query_triton_smem_limit", lambda: 101376)
    assert not _dispatch_ok(q)


def test_dot_product_attention_sm_scale_xla():
    """``sm_scale`` used to be silently ignored by the einsum path."""
    q, k, v, _ = _inputs(2, 32, 32, 2, 16)
    out = dot_product_attention(q, k, v, sm_scale=0.3, implementation="xla")
    assert _rel(out, _reference(q, k, v, scale=0.3)) < 1e-3
    default = dot_product_attention(q, k, v, implementation="xla")
    assert _rel(default, _reference(q, k, v)) < 1e-3


def test_dot_product_attention_bad_implementation():
    q = jnp.zeros((1, 16, 1, 16))
    with pytest.raises(ValueError, match="implementation"):
        dot_product_attention(q, q, q, implementation="cudnn")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize(
    "shape,mask,causal,window",
    [
        ((2, 256, 4, 64), None, False, None),
        ((2, 300, 4, 32), CausalMask(), True, None),
        ((2, 3, 200, 2, 64), None, False, None),
        ((2, 256, 4, 64), LocalWindowMask(32, 16), False, (32, 16)),
    ],
)
def test_dot_product_attention_dispatch_gpu(dtype, shape, mask, causal, window):
    _requires_accelerator()
    ks = jax.random.split(jax.random.key(1), 4)
    q, k, v = (jax.random.normal(kk, shape, dtype) for kk in ks[:3])
    g = jax.random.normal(ks[3], shape)
    assert _use_fused_gpu_attention(q, k, v, mask, None, 0.0, True, None, None)

    def loss(impl):
        def fn(q, k, v):
            out = dot_product_attention(q, k, v, mask=mask, implementation=impl)
            return jnp.sum(out.astype(jnp.float32) * g)

        return fn

    flat = [x.reshape((-1,) + x.shape[-3:]).astype(jnp.float32) for x in (q, k, v)]
    ref = _reference(*flat, causal=causal, window=window).reshape(shape)
    out = jax.jit(lambda q, k, v: dot_product_attention(q, k, v, mask=mask))(q, k, v)
    assert out.shape == shape and out.dtype == dtype
    tol = 1e-2 if dtype == jnp.bfloat16 else 2e-3
    assert _rel(out, ref) < tol
    with jax.default_matmul_precision("highest"):
        g_ref = jax.grad(loss("xla"), (0, 1, 2))(
            *(x.astype(jnp.float32) for x in (q, k, v))
        )
    g_new = jax.jit(jax.grad(loss(None), (0, 1, 2)))(q, k, v)
    for a, b in zip(g_new, g_ref, strict=True):
        assert _rel(a, b) < 2 * tol


def test_dot_product_attention_fallbacks_exact_gpu():
    """Dense masks / active dropout / sowing keep the exact Flax results."""
    _requires_accelerator()
    q, k, v, _ = _inputs(2, 256, 256, 2, 32)
    dense = jnp.tril(jnp.ones((256, 256), bool))
    a = dot_product_attention(q, k, v, mask=dense)
    b = dot_product_attention(q, k, v, mask=dense, implementation="xla")
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    rng = jax.random.key(7)
    kw = dict(dropout_rng=rng, dropout_rate=0.2, deterministic=False)
    a = dot_product_attention(q, k, v, **kw)
    b = dot_product_attention(q, k, v, implementation="xla", **kw)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    class Sower(nnx.Module):
        pass

    module = Sower()
    dot_product_attention(q, k, v, module=module)
    assert hasattr(module, "attention_weights")


def test_transformer_default_attention_gpu():
    """The default ``attention_fn`` of probjax transformers takes the fast path."""
    _requires_accelerator()
    from probjax.nn import Transformer

    model = Transformer(
        model_dim=64, num_heads=4, num_layers=2, attn_size=16, rngs=nnx.Rngs(0)
    )
    ref = Transformer(
        model_dim=64,
        num_heads=4,
        num_layers=2,
        attn_size=16,
        attention_fn=lambda *a, **kw: dot_product_attention(
            *a, **kw, implementation="xla"
        ),
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (2, 256, 64))
    assert _rel(model(x), ref(x)) < 5e-3


# ---------------------------------------------------------------------------
# small-SMEM tile heuristic (H1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_bytes", [2, 4])
@pytest.mark.parametrize("head_dim", [16, 32, 64, 128, 256])
def test_smem_adjustment_never_grows(monkeypatch, dtype_bytes, head_dim):
    monkeypatch.setattr(attn_kernels, "_SMEM_LIMIT_CACHE", 101376)
    req = BlockSizes()
    bs, stages = attn_kernels._adjust_block_sizes_for_smem(
        req,
        2,
        q_seq_len=4096,
        kv_seq_len=4096,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )
    for f in (
        "block_q",
        "block_k",
        "block_q_dkv",
        "block_kv_dkv",
        "block_q_dq",
        "block_kv_dq",
    ):
        assert getattr(bs, f) <= getattr(req, f)
        assert getattr(bs, f) >= 16
    assert 1 <= stages <= 2


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("head_dim", [16, 32, 64, 128, 256])
def test_auto_config_compiles_gpu(dtype, head_dim):
    """Under jit the proactive SMEM estimate is all we have: it must fit."""
    _requires_accelerator()
    q, k, v, g = _inputs(1, 512, 512, 2, head_dim, dtype)
    from probjax.nn.pallas_kernels import mha

    scale = 1 / math.sqrt(head_dim)

    def loss(q, k, v):
        return jnp.sum(mha(q, k, v, sm_scale=scale).astype(jnp.float32) * g)

    out = jax.jit(lambda q, k, v: mha(q, k, v, sm_scale=scale))(q, k, v)
    grads = jax.jit(jax.grad(loss, (0, 1, 2)))(q, k, v)
    tol = 1e-2 if dtype == jnp.bfloat16 else 2e-3
    assert _rel(out, _reference(q, k, v)) < tol
    gr = jax.grad(lambda *a: jnp.sum(_reference(*a) * g), (0, 1, 2))(
        *(x.astype(jnp.float32) for x in (q, k, v))
    )
    for a, b in zip(grads, gr, strict=True):
        assert _rel(a, b) < 2 * tol


def test_small_smem_num_warps(monkeypatch):
    monkeypatch.setattr(attn_kernels, "_SMEM_LIMIT_CACHE", 101376)
    warps = attn_kernels._small_smem_num_warps
    assert warps(None, 2, head_dim=128, dtype_bytes=4) == 4
    assert warps(8, 2, head_dim=128, dtype_bytes=4) == 8  # explicit wins
    assert warps(None, 1, head_dim=128, dtype_bytes=4) is None  # non-default
    assert warps(None, 2, head_dim=64, dtype_bytes=2) is None
    monkeypatch.setattr(attn_kernels, "_SMEM_LIMIT_CACHE", 232448)  # H100
    assert warps(None, 2, head_dim=128, dtype_bytes=4) is None


def test_smem_estimates_match_sm86_failures():
    """Regression: estimates never fall below bytes Triton requested on sm_86."""
    fwd = attn_kernels._estimate_fwd_smem_bytes
    bwd = attn_kernels._estimate_bwd_smem_bytes
    # (bq, bk, bd, stages, dtype_bytes) -> requested bytes (RESOURCE_EXHAUSTED)
    for args, requested in [
        ((64, 64, 128, 3, 2), 106496),
        ((128, 64, 128, 3, 2), 114688),
        ((128, 128, 128, 2, 2), 131072),
        ((128, 64, 64, 3, 4), 114688),
        ((64, 128, 64, 2, 4), 114688),
        ((128, 64, 256, 1, 2), 114688),
    ]:
        assert fwd(*args) >= requested
    # (outer, inner, bd, dtype_bytes) -> requested bytes
    for (outer, inner, bd, b), requested in [
        ((64, 128, 128, 2), 114688),
        ((128, 64, 128, 2), 114688),
        ((128, 128, 128, 2), 163840),
        ((64, 64, 128, 4), 131072),
        ((128, 32, 128, 4), 163840),
        ((128, 128, 64, 4), 131072),
    ]:
        assert bwd(inner, outer, outer, inner, bd, b) + 1024 >= requested
    # The tuned fp32 d=128 backward (64 x 32) fits the 99 KB of an RTX 3090.
    assert bwd(32, 64, 64, 32, 128, 4) <= 101376 - attn_kernels._SMEM_RESERVE
