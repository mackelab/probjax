"""XLA (CPU) attention path of ``mha`` / ``mla``.

The reference for mask/bias/dropout semantics is the Pallas kernel itself run in
interpret mode (``mha(..., interpret=True)``), which is what CPU used before the
XLA path existed.  Gradients and forward-mode derivatives are checked against
plain autodiff of the dense strategy, and the dense strategy against an
independent ``jax.nn.dot_product_attention`` reference.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.layers.attention import flex_attention
from probjax.nn.pallas_kernels import (
    BlockDiagonalMask,
    CausalAlibiBias,
    CausalLocalWindowMask,
    CausalMask,
    ConstantBias,
    DenseBias,
    DistanceDecayBias,
    KVLenMask,
    LearnedRelativePositionBias,
    LocalWindowMask,
    MarginalizationMask,
    PerHeadScaleBias,
    SameSegmentMask,
    SeqLenMask,
    SoftCappingBias,
    T5RelativePositionBias,
)
from probjax.nn.pallas_kernels.kernels.attention import BlockSizes, mha
from probjax.nn.pallas_kernels.kernels.attention_xla import (
    use_xla_attention,
    xla_attention,
)
from probjax.nn.pallas_kernels.kernels.latent_attention import _materialize_kv, mla

# The XLA path is what mha/mla dispatch to on CPU; on GPU the Triton kernels
# run instead (and fp32 matmuls default to TF32 there).
pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="tests the CPU (XLA) attention path"
)

B, T, H, D = 2, 32, 2, 16
SCALE = 0.25
SMALL_BLOCKS = BlockSizes(
    block_q=16,
    block_k=16,
    block_q_dkv=16,
    block_kv_dkv=16,
    block_q_dq=16,
    block_kv_dq=16,
)


def _rel(a, b):
    a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
    return float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-30))


def _qkv(seq=T, heads=H, kv_heads=None, dim=D, batch=B, seed=0, kv_seq=None):
    ks = jax.random.split(jax.random.key(seed), 3)
    kv_heads = kv_heads or heads
    kv_seq = kv_seq or seq
    q = jax.random.normal(ks[0], (batch, seq, heads, dim))
    k = jax.random.normal(ks[1], (batch, kv_seq, kv_heads, dim))
    v = jax.random.normal(ks[2], (batch, kv_seq, kv_heads, dim))
    return q, k, v


def _strategies(fn):
    """Run ``fn(strategy, block_q, block_k)`` for dense and two tilings."""
    return {
        "dense": fn("dense", None, None),
        "tiled_even": fn("tiled", 16, 8),
        "tiled_ragged": fn("tiled", 12, 10),
    }


def _ids(batch, seq, n_seg, seed):
    return jnp.sort(jax.random.randint(jax.random.key(seed), (batch, seq), 0, n_seg))


MASKS = {
    "none": lambda: None,
    "causal": CausalMask,
    "local": lambda: LocalWindowMask(left_window=5, right_window=3),
    "causal_local": lambda: CausalLocalWindowMask(left_window=7),
    "kvlen": lambda: KVLenMask(jnp.array([20, 32], jnp.int32)),
    "seqlen": lambda: SeqLenMask(jnp.array([17, 32], jnp.int32)),
    "same_segment": lambda: SameSegmentMask(_ids(B, T, 3, 1), None),
    "block_diag": lambda: BlockDiagonalMask(_ids(B, T, 4, 2), _ids(B, T, 4, 2)),
    "marginalization": lambda: MarginalizationMask(
        jax.random.bernoulli(jax.random.key(3), 0.6, (B, T))
    ),
    "kvlen_and_causal": lambda: (
        KVLenMask(jnp.array([20, 32], jnp.int32)) & CausalMask()
    ),
}

BIASES = {
    "dense": lambda: DenseBias(jax.random.normal(jax.random.key(4), (B, H, T, T))),
    "dense_bcast": lambda: DenseBias(
        jax.random.normal(jax.random.key(5), (1, 1, T, T))
    ),
    "alibi": CausalAlibiBias,
    "decay": lambda: DistanceDecayBias(0.3),
    "const_plus_decay": lambda: ConstantBias(0.2) + DistanceDecayBias(0.1),
    "softcap": lambda: SoftCappingBias(2.0),
}


def test_dispatch_uses_xla_on_cpu():
    q, k, v = _qkv()
    assert use_xla_attention(q, interpret=False)
    assert not use_xla_attention(q, interpret=True)
    jaxpr = str(jax.make_jaxpr(lambda q, k, v: mha(q, k, v, sm_scale=SCALE))(q, k, v))
    assert "pallas_call" not in jaxpr and "_mha_inner" not in jaxpr
    jaxpr = str(
        jax.make_jaxpr(lambda q, k, v: mha(q, k, v, sm_scale=SCALE, interpret=True))(
            q, k, v
        )
    )
    assert "_mha_inner" in jaxpr


@pytest.mark.parametrize("mask_name", list(MASKS))
def test_masks_match_interpret_kernel(mask_name):
    q, k, v = _qkv()
    mask = MASKS[mask_name]()
    ref = mha(
        q, k, v, mask=mask, sm_scale=SCALE, interpret=True, block_sizes=SMALL_BLOCKS
    )
    outs = _strategies(
        lambda s, bq, bk: xla_attention(
            q, k, v, mask=mask, sm_scale=SCALE, strategy=s, block_q=bq, block_k=bk
        )
    )
    for name, out in outs.items():
        assert _rel(out, ref) < 1e-5, name


@pytest.mark.parametrize("bias_name", list(BIASES))
def test_biases_match_interpret_kernel(bias_name):
    q, k, v = _qkv()
    bias = BIASES[bias_name]()
    mask = CausalMask()
    ref = mha(
        q,
        k,
        v,
        mask=mask,
        bias=bias,
        sm_scale=SCALE,
        interpret=True,
        block_sizes=SMALL_BLOCKS,
    )
    outs = _strategies(
        lambda s, bq, bk: xla_attention(
            q,
            k,
            v,
            mask=mask,
            bias=bias,
            sm_scale=SCALE,
            strategy=s,
            block_q=bq,
            block_k=bk,
        )
    )
    for name, out in outs.items():
        assert _rel(out, ref) < 1e-5, name


def test_per_head_scale_bias():
    # The Pallas kernel cannot take this bias (its scales are captured
    # constants), so compare against the formula directly.
    q, k, v = _qkv()
    scales = jnp.array([1.5, 0.5])
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k) * SCALE * scales[None, :, None, None]
    causal = jnp.tril(jnp.ones((T, T), bool))
    p = jax.nn.softmax(jnp.where(causal, s, -jnp.inf), axis=-1)
    ref = jnp.einsum("bhqk,bkhd->bqhd", p, v)
    outs = _strategies(
        lambda st, bq, bk: xla_attention(
            q,
            k,
            v,
            mask=CausalMask(),
            bias=PerHeadScaleBias(scales),
            sm_scale=SCALE,
            strategy=st,
            block_q=bq,
            block_k=bk,
        )
    )
    for name, out in outs.items():
        assert _rel(out, ref) < 1e-5, name


def test_dense_matches_jax_reference():
    q, k, v = _qkv(seq=40, kv_seq=24)
    bias = jax.random.normal(jax.random.key(7), (1, H, 40, 24))
    ref = jax.nn.dot_product_attention(q, k, v, bias=bias, scale=SCALE)
    for strategy, bq, bk in [("dense", None, None), ("tiled", 16, 7)]:
        out = xla_attention(
            q,
            k,
            v,
            bias=DenseBias(bias),
            sm_scale=SCALE,
            strategy=strategy,
            block_q=bq,
            block_k=bk,
        )
        assert _rel(out, ref) < 1e-5, strategy


@pytest.mark.parametrize("impl", ["counter", "materialize"])
def test_dropout_reproduces_kernel_mask(impl):
    q, k, v = _qkv()
    rng = jax.random.key(11)
    kw = dict(sm_scale=SCALE, dropout_rate=0.3, rng=rng, dropout_impl=impl)
    ref = mha(q, k, v, interpret=True, block_sizes=SMALL_BLOCKS, **kw)
    outs = _strategies(
        lambda s, bq, bk: xla_attention(
            q, k, v, strategy=s, block_q=bq, block_k=bk, **kw
        )
    )
    for name, out in outs.items():
        assert _rel(out, ref) < 1e-5, name


def test_fully_masked_rows_are_zero():
    q, k, v = _qkv()
    # Keys >= 0 are all masked for batch 0 -> every row of batch 0 is empty.
    mask = KVLenMask(jnp.array([0, 32], jnp.int32))
    for strategy in ("dense", "tiled"):
        out = xla_attention(
            q, k, v, mask=mask, sm_scale=SCALE, strategy=strategy, block_q=8, block_k=8
        )
        assert jnp.all(out[0] == 0) and jnp.all(jnp.isfinite(out))
        g = jax.grad(
            lambda q, strategy=strategy: jnp.sum(
                xla_attention(
                    q, k, v, mask=mask, sm_scale=SCALE, strategy=strategy, block_k=8
                )
                ** 2
            )
        )(q)
        assert jnp.all(jnp.isfinite(g))


def _loss(strategy, bq=None, bk=None, **kw):
    def f(q, k, v):
        out = xla_attention(
            q, k, v, sm_scale=SCALE, strategy=strategy, block_q=bq, block_k=bk, **kw
        )
        return jnp.sum(jnp.sin(out))

    return f


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(mask=CausalMask()),
        dict(mask=KVLenMask(jnp.array([20, 32], jnp.int32)) & CausalMask()),
        dict(mask=CausalMask(), bias=CausalAlibiBias()),
        dict(mask=CausalMask(), dropout_rate=0.2, rng=jax.random.key(1)),
        dict(
            dropout_rate=0.2,
            rng=jax.random.key(1),
            dropout_impl="materialize",
        ),
    ],
    ids=["plain", "causal", "kvlen_causal", "alibi", "drop_counter", "drop_mat"],
)
def test_tiled_gradients_match_dense(kw):
    q, k, v = _qkv(seq=37)
    ref = jax.grad(_loss("dense", **kw), argnums=(0, 1, 2))(q, k, v)
    got = jax.jit(jax.grad(_loss("tiled", 16, 10, **kw), argnums=(0, 1, 2)))(q, k, v)
    for a, b in zip(got, ref, strict=True):
        assert _rel(a, b) < 1e-5


def test_learnable_bias_gradients():
    """Bias parameters receive gradients through both strategies."""
    q, k, v = _qkv()
    table = jax.random.normal(jax.random.key(8), (H, 16))
    rel_table = jax.random.normal(jax.random.key(9), (H, 2 * 6 + 1))
    dense = jax.random.normal(jax.random.key(10), (1, H, T, T))

    def loss(strategy, params):
        t5, lrp, db = params
        bias = T5RelativePositionBias(
            t5, num_buckets=16, max_distance=20
        ) + LearnedRelativePositionBias(lrp, max_distance=6)
        if db is not None:
            # DenseBias needs its block data, so it cannot sit inside a SumBias.
            bias = DenseBias(db)
        out = xla_attention(
            q,
            k,
            v,
            mask=CausalMask(),
            bias=bias,
            sm_scale=SCALE,
            strategy=strategy,
            block_q=8,
            block_k=8,
        )
        return jnp.sum(jnp.sin(out))

    params_sets = [(table, rel_table, None), (table, rel_table, dense)]

    def ref_loss(params):
        t5, lrp, db = params
        bias = T5RelativePositionBias(
            t5, num_buckets=16, max_distance=20
        ) + LearnedRelativePositionBias(lrp, max_distance=6)
        if db is not None:
            bias = DenseBias(db)
        dense_bias = bias.dense(T, T, batch_size=1, num_heads=H)
        out = jax.nn.dot_product_attention(
            q, k, v, bias=dense_bias, scale=SCALE, is_causal=True
        )
        return jnp.sum(jnp.sin(out))

    for params in params_sets:
        ref = jax.grad(ref_loss)(params)
        for strategy in ("dense", "tiled"):
            got = jax.grad(functools.partial(loss, strategy))(params)
            used = (2,) if params[2] is not None else (0, 1)
            for i in used:
                assert float(jnp.linalg.norm(ref[i])) > 0
                assert _rel(got[i], ref[i]) < 1e-4, (strategy, i)


@pytest.mark.parametrize("strategy", ["dense", "tiled"])
def test_forward_mode_matches_dense(strategy):
    q, k, v = _qkv(seq=24)
    tq, tk, tv = _qkv(seq=24, seed=5)
    kw = dict(mask=CausalMask(), bias=CausalAlibiBias())

    def f(strategy):
        return lambda q, k, v: xla_attention(
            q, k, v, sm_scale=SCALE, strategy=strategy, block_q=8, block_k=8, **kw
        )

    _, ref = jax.jvp(f("dense"), (q, k, v), (tq, tk, tv))
    out, tan = jax.jvp(f(strategy), (q, k, v), (tq, tk, tv))
    assert _rel(tan, ref) < 1e-5
    # Forward mode through the public API (diff_mode="forward" and plain jvp).
    _, tan2 = jax.jvp(
        lambda q: mha(q, k, v, sm_scale=SCALE, diff_mode="forward", **kw), (q,), (tq,)
    )
    _, ref2 = jax.jvp(lambda q: f("dense")(q, k, v), (q,), (tq,))
    assert _rel(tan2, ref2) < 1e-5


def test_hessian_vector_product_tiled():
    q, k, v = _qkv(seq=16)
    f = _loss("tiled", 8, 8, mask=CausalMask())
    g = _loss("dense", mask=CausalMask())
    hvp = jax.jvp(jax.grad(lambda q: f(q, k, v)), (q,), (v,))[1]
    ref = jax.jvp(jax.grad(lambda q: g(q, k, v)), (q,), (v,))[1]
    assert _rel(hvp, ref) < 1e-4


@pytest.mark.parametrize("strategy", ["dense", "tiled"])
def test_gqa_matches_repeated_heads(strategy):
    q, k, v = _qkv(heads=4, kv_heads=2)
    mask = CausalMask()
    bias = CausalAlibiBias()

    def f(q, k, v):
        return xla_attention(
            q,
            k,
            v,
            mask=mask,
            bias=bias,
            sm_scale=SCALE,
            strategy=strategy,
            block_q=8,
            block_k=8,
        )

    def ref(q, k, v):
        return f(q, jnp.repeat(k, 2, axis=2), jnp.repeat(v, 2, axis=2))

    assert _rel(f(q, k, v), ref(q, k, v)) < 1e-6
    g = jax.grad(lambda *a: jnp.sum(f(*a) ** 2), argnums=(1, 2))(q, k, v)
    gr = jax.grad(lambda *a: jnp.sum(ref(*a) ** 2), argnums=(1, 2))(q, k, v)
    for a, b in zip(g, gr, strict=True):
        assert _rel(a, b) < 1e-5


def test_bfloat16_close_to_float32():
    q, k, v = _qkv(seq=64)
    ref = xla_attention(q, k, v, mask=CausalMask(), sm_scale=SCALE)
    for strategy in ("dense", "tiled"):
        out = xla_attention(
            *(x.astype(jnp.bfloat16) for x in (q, k, v)),
            mask=CausalMask(),
            sm_scale=SCALE,
            strategy=strategy,
            block_q=16,
            block_k=16,
        )
        assert out.dtype == jnp.bfloat16
        assert _rel(out, ref) < 2e-2


def test_flex_attention_cpu_unaligned_matches_reference():
    q, k, v = _qkv(seq=50, dim=24)
    bias = jax.random.normal(jax.random.key(3), (1, 1, 50, 50))
    out = flex_attention(q, k, v, mask=CausalMask(), bias=DenseBias(bias))
    ref = jax.nn.dot_product_attention(
        q, k, v, bias=bias, is_causal=True, scale=1 / np.sqrt(24)
    )
    assert _rel(out, ref) < 1e-5
    g = jax.grad(lambda q: jnp.sum(flex_attention(q, k, v, mask=CausalMask()) ** 2))(q)
    gr = jax.grad(
        lambda q: jnp.sum(
            jax.nn.dot_product_attention(q, k, v, is_causal=True, scale=1 / np.sqrt(24))
            ** 2
        )
    )(q)
    assert _rel(g, gr) < 1e-5


def test_tiled_backward_memory_is_bounded():
    """The tiled backward saves no [Q, K] residuals: its temp memory grows
    roughly linearly in T, while the dense strategy grows quadratically."""

    def temp_mb(strategy, seq):
        x = jax.ShapeDtypeStruct((1, seq, 2, 32), jnp.float32)
        loss = _loss(strategy, 128, 128, mask=CausalMask())
        fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        return fn.lower(x, x, x).compile().memory_analysis().temp_size_in_bytes

    t1, t2 = temp_mb("tiled", 1024), temp_mb("tiled", 4096)
    d2 = temp_mb("dense", 4096)
    assert t2 < 6 * t1  # ~4x for 4x the length (quadratic would be 16x)
    assert t2 * 8 < d2


@pytest.mark.parametrize("absorbed", [False, True])
@pytest.mark.parametrize("q_len", [32, 4])
def test_mla_cpu_path_matches_materialized(absorbed, q_len):
    ks = jax.random.split(jax.random.key(0), 5)
    rank, heads, dk, dv = 24, 2, 16, 8
    cache = jax.random.normal(ks[0], (B, T, rank))
    if absorbed:
        wk = wv = None
        q = jax.random.normal(ks[1], (B, q_len, heads, rank))
    else:
        q = jax.random.normal(ks[1], (B, q_len, heads, dk))
        wk = jax.random.normal(ks[2], (heads, dk, rank)) / np.sqrt(rank)
        wv = jax.random.normal(ks[3], (heads, dv, rank)) / np.sqrt(rank)
    mask = KVLenMask(jnp.array([20, 32], jnp.int32))

    def f(q, cache, wk, wv):
        return mla(q, cache, wk, wv, mask=mask, bias=CausalAlibiBias())

    def ref(q, cache, wk, wv):
        key, value = _materialize_kv(cache, wk, wv, heads)
        scale = 1.0 / np.sqrt(q.shape[-1])
        return xla_attention(
            q,
            key,
            value,
            mask=mask,
            bias=CausalAlibiBias(),
            sm_scale=scale,
            strategy="dense",
        )

    jaxpr = str(jax.make_jaxpr(f)(q, cache, wk, wv))
    assert "pallas_call" not in jaxpr
    assert _rel(f(q, cache, wk, wv), ref(q, cache, wk, wv)) < 1e-5
    argnums = (0, 1) if absorbed else (0, 1, 2, 3)
    g = jax.grad(lambda *a: jnp.sum(jnp.sin(f(*a))), argnums=argnums)(q, cache, wk, wv)
    gr = jax.grad(lambda *a: jnp.sum(jnp.sin(ref(*a))), argnums=argnums)(
        q, cache, wk, wv
    )
    for a, b in zip(g, gr, strict=True):
        assert _rel(a, b) < 1e-5
