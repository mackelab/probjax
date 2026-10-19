"""Multi-head latent attention: layer and Pallas kernel against references.

The shared-latent design has one cache entry per token, but the key and value
roles stay distinct (``K_h = C@W_K[h]``, ``V_h = C@W_V[h]``) unless the
absorbed form is selected (``K = V = C``).  These tests pin both the maths and
that distinction.
"""

import math

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn.layers.attention import QASSMaxQueryScale
from probjax.nn.layers.latent_attention import MultiHeadLatentAttention
from probjax.nn.layers.lowrank import LowRankLinear
from probjax.nn.pallas_kernels import (
    BlockDiagonalMask,
    CausalAlibiBias,
    CausalFromBottomRightMask,
    CausalLocalWindowMask,
    CausalMask,
    ConstantBias,
    DenseBias,
    DistanceDecayBias,
    IdentityBias,
    KVLenMask,
    KeyPaddingMask,
    LocalWindowMask,
    MarginalizationMask,
    NoMask,
    PrefixLMMask,
    QKVLengthMask,
    SameSegmentMask,
    SeqLenMask,
    SoftCappingBias,
    SymmetricAlibiBias,
)
from probjax.nn.pallas_kernels.kernels.attention import BlockSizes, mha
from probjax.nn.pallas_kernels.kernels.latent_attention import (
    _materialize_kv,
    mla,
)


def _block_sizes(seq: int, mul: int = 16) -> BlockSizes:
    size = min(seq, mul)
    return BlockSizes(
        block_q=size,
        block_k=size,
        block_q_dkv=size,
        block_kv_dkv=size,
        block_q_dq=size,
        block_kv_dq=size,
    )


def _reference(q, cache, wk, wv, num_heads):
    """Dense softmax attention over the materialized per-head K/V."""
    key, value = _materialize_kv(cache, wk, wv, num_heads)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, key) / math.sqrt(q.shape[-1])
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, -1), value)


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["materialized", "absorbed"])
def test_layer_matches_dense_reference(mode):
    heads, features, rank, head_dim = 4, 32, 16, 8
    # Absorbed fixes the query head dim to the latent width (no KV projection).
    query_head_dim = rank if mode == "absorbed" else head_dim
    layer = MultiHeadLatentAttention(
        num_heads=heads,
        in_features=features,
        kv_lora_rank=rank,
        mode=mode,
        qk_head_dim=query_head_dim,
        v_head_dim=head_dim,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (2, 5, features))
    out = layer(x)
    assert out.shape == x.shape
    assert jnp.isfinite(out).all()

    # The layer's own pieces reproduce a plain attention over its latent.
    latent = layer.encode_latent(x)
    q = layer._project_queries(x)
    if mode == "materialized":
        key, value = layer._kv_from_latent(latent)
    else:
        key = jnp.broadcast_to(latent[..., None, :], latent.shape[:-1] + (heads, rank))
        value = key
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, key) / math.sqrt(q.shape[-1])
    attn = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, -1), value)
    expected = layer._out_from_attention(attn)
    assert jnp.allclose(out, expected, atol=1e-5)


def test_layer_defaults_to_absorbed_without_kv_projection():
    """The default must not learn KV projections; K = V = C."""
    layer = MultiHeadLatentAttention(
        num_heads=4, in_features=32, kv_lora_rank=16, rngs=nnx.Rngs(0)
    )
    assert layer.mode == "absorbed"
    assert layer.wk is None
    assert layer.qk_head_dim == 16  # fixed to kv_lora_rank
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    assert layer(x).shape == (2, 5, 32)


def test_layer_absorbed_rejects_mismatched_qk_head_dim():
    """A query head dim != kv_lora_rank in absorbed mode is a config error."""
    with pytest.raises(ValueError, match="qk_head_dim == kv_lora_rank"):
        MultiHeadLatentAttention(
            num_heads=4,
            in_features=32,
            kv_lora_rank=16,
            qk_head_dim=8,
            rngs=nnx.Rngs(0),
        )


def test_layer_absorbed_accepts_matching_qk_head_dim():
    layer = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        qk_head_dim=16,
        rngs=nnx.Rngs(0),
    )
    assert layer.qk_head_dim == 16


def test_layer_cache_is_one_latent_per_token():
    """The whole point: the cache width is the latent, not heads * head_dim."""
    heads, features, rank, head_dim = 4, 32, 16, 8
    layer = MultiHeadLatentAttention(
        num_heads=heads,
        in_features=features,
        kv_lora_rank=rank,
        mode="materialized",
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (1, 6, features))
    latent = layer.encode_latent(x)
    assert latent.shape == (1, 6, rank)
    assert latent.shape[-1] < heads * head_dim


def test_layer_cross_attention_with_distinct_kv_width():
    layer = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        in_kv_features=48,
        kv_lora_rank=16,
        mode="materialized",
        qk_head_dim=8,
        v_head_dim=8,
        rngs=nnx.Rngs(0),
    )
    q = jax.random.normal(jax.random.key(1), (2, 5, 32))
    kv = jax.random.normal(jax.random.key(2), (2, 9, 48))
    assert layer(q, kv).shape == (2, 5, 32)


@pytest.mark.parametrize("mode", ["materialized", "absorbed"])
def test_layer_use_kernel_matches_dense(mode):
    """The fused path must reproduce the dense path exactly, both directions."""
    # Absorbed queries live in the latent width; materialized ones in qk_head_dim.
    qk_head_dim = 16 if mode == "absorbed" else 8
    dense = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        mode=mode,
        qk_head_dim=qk_head_dim,
        v_head_dim=8,
        rngs=nnx.Rngs(0),
    )
    fused = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        mode=mode,
        qk_head_dim=qk_head_dim,
        v_head_dim=8,
        rngs=nnx.Rngs(1),
        use_kernel=True,
        block_sizes=_block_sizes(32),
    )
    nnx.update(fused, nnx.state(dense))

    x = jax.random.normal(jax.random.key(1), (2, 32, 32))
    assert jnp.allclose(dense(x), fused(x), atol=1e-4)

    g_dense = nnx.grad(lambda m, z: m(z).sum(), argnums=1)(dense, x)
    g_fused = nnx.grad(lambda m, z: m(z).sum(), argnums=1)(fused, x)
    assert jnp.allclose(g_dense, g_fused, atol=1e-4)


def test_layer_gradients_are_finite():
    layer = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        mode="materialized",
        qk_head_dim=8,
        v_head_dim=8,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    grad = nnx.grad(lambda m, z: m(z).sum(), argnums=1)(layer, x)
    assert grad.shape == x.shape
    assert jnp.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def test_kernel_materialized_matches_reference():
    b, s, heads, head_dim, rank, v_dim = 2, 32, 4, 16, 8, 12
    q = jax.random.normal(jax.random.key(0), (b, s, heads, head_dim))
    cache = jax.random.normal(jax.random.key(1), (b, s, rank))
    wk = jax.random.normal(jax.random.key(2), (heads, head_dim, rank))
    wv = jax.random.normal(jax.random.key(3), (heads, v_dim, rank))

    out = mla(q, cache, wk, wv, block_sizes=_block_sizes(s), interpret=True)
    expected = _reference(q, cache, wk, wv, heads)
    assert jnp.allclose(out, expected, atol=1e-5)


def test_kernel_absorbed_matches_reference():
    b, s, heads, rank = 2, 32, 4, 16
    q = jax.random.normal(jax.random.key(0), (b, s, heads, rank))
    cache = jax.random.normal(jax.random.key(1), (b, s, rank))

    out = mla(
        q,
        cache,
        None,
        None,
        block_sizes=_block_sizes(s),
        interpret=True,
    )
    expected = _reference(q, cache, None, None, heads)
    assert jnp.allclose(out, expected, atol=1e-5)


def test_kernel_rejects_mixed_projections():
    """Mixed wk/wv is not a supported mode: both or neither."""
    b, s, heads, rank, v_dim = 2, 32, 4, 16, 12
    q = jax.random.normal(jax.random.key(0), (b, s, heads, rank))
    cache = jax.random.normal(jax.random.key(1), (b, s, rank))
    wv = jax.random.normal(jax.random.key(3), (heads, v_dim, rank))
    with pytest.raises(ValueError, match="both wk and wv"):
        mla(
            q,
            cache,
            None,
            wv,
            block_sizes=_block_sizes(s),
            interpret=True,
        )


def test_kernel_gradients_match_reference():
    b, s, heads, head_dim, rank, v_dim = 2, 32, 4, 16, 8, 12
    q = jax.random.normal(jax.random.key(0), (b, s, heads, head_dim))
    cache = jax.random.normal(jax.random.key(1), (b, s, rank))
    wk = jax.random.normal(jax.random.key(2), (heads, head_dim, rank))
    wv = jax.random.normal(jax.random.key(3), (heads, v_dim, rank))
    bs = _block_sizes(s)

    def run(q_, c_, wk_, wv_):
        return mla(q_, c_, wk_, wv_, block_sizes=bs, interpret=True).sum()

    def ref(q_, c_, wk_, wv_):
        return _reference(q_, c_, wk_, wv_, heads).sum()

    grads = jax.grad(run, argnums=(0, 1, 2, 3))(q, cache, wk, wv)
    expected = jax.grad(ref, argnums=(0, 1, 2, 3))(q, cache, wk, wv)
    for got, want in zip(grads, expected, strict=True):
        assert jnp.allclose(got, want, atol=1e-4)


def test_kernel_respects_causal_mask():
    from probjax.nn.pallas_kernels import CausalMask

    b, s, heads, head_dim, rank = 2, 32, 4, 16, 8
    q = jax.random.normal(jax.random.key(0), (b, s, heads, head_dim))
    cache = jax.random.normal(jax.random.key(1), (b, s, rank))
    wk = jax.random.normal(jax.random.key(2), (heads, head_dim, rank))
    wv = jax.random.normal(jax.random.key(3), (heads, head_dim, rank))

    out = mla(
        q,
        cache,
        wk,
        wv,
        mask=CausalMask(),
        block_sizes=_block_sizes(s),
        interpret=True,
    )
    key = jnp.einsum("bsr,hdr->bshd", cache, wk)
    value = jnp.einsum("bsr,hdr->bshd", cache, wv)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, key) / math.sqrt(head_dim)
    causal = jnp.tril(jnp.ones((s, s), dtype=bool))
    expected = jnp.einsum(
        "bhqk,bkhd->bqhd",
        jax.nn.softmax(jnp.where(causal, logits, -1e15), -1),
        value,
    )
    assert jnp.allclose(out, expected, atol=1e-5)


def test_kernel_rejects_absorbed_with_mismatched_head_dim():
    q = jnp.zeros((1, 16, 4, 8))
    cache = jnp.zeros((1, 16, 16))  # rank 16 != q head dim 8
    with pytest.raises(ValueError, match="absorbed mode"):
        mla(q, cache, None, None, interpret=True)


# ---------------------------------------------------------------------------
# Kernel parity with ``mha`` across the mask / bias surface
# ---------------------------------------------------------------------------
# ``mla`` shares ``mha``'s mask/bias machinery, so the contract is exact
# equality with ``mha`` over the materialized K/V -- not a hand-written softmax.
# Array-carrying masks/biases are what expose argument-order and leaf-splitting
# mistakes, so every supported one is exercised, singly and combined.

_PARITY_B, _PARITY_S, _PARITY_H, _PARITY_D, _PARITY_R = 2, 32, 4, 16, 8


def _parity_inputs():
    keys = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(keys[0], (_PARITY_B, _PARITY_S, _PARITY_H, _PARITY_D))
    cache = jax.random.normal(keys[1], (_PARITY_B, _PARITY_S, _PARITY_R))
    wk = jax.random.normal(keys[2], (_PARITY_H, _PARITY_D, _PARITY_R))
    wv = jax.random.normal(keys[3], (_PARITY_H, _PARITY_D, _PARITY_R))
    return q, cache, wk, wv


def _assert_matches_mha(mask=None, bias=None):
    q, cache, wk, wv = _parity_inputs()
    key = jnp.einsum("bsr,hdr->bshd", cache, wk)
    value = jnp.einsum("bsr,hdr->bshd", cache, wv)
    block_sizes = _block_sizes(_PARITY_S)
    sm_scale = 1.0 / math.sqrt(_PARITY_D)
    got = mla(
        q,
        cache,
        wk,
        wv,
        mask=mask,
        bias=bias,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        interpret=True,
    )
    want = mha(
        q,
        key,
        value,
        mask=mask,
        bias=bias,
        sm_scale=sm_scale,
        block_sizes=block_sizes,
        interpret=True,
    )
    assert jnp.allclose(got, want, atol=1e-5), float(jnp.abs(got - want).max())


# Every mask ``mha`` supports.  Constructed at module scope so a signature
# change in one surfaces immediately as a collection error.
_PARITY_MASKS = {
    "no_mask": NoMask(),
    "causal": CausalMask(),
    "local_window": LocalWindowMask(8, 8),
    "causal_local_window": CausalLocalWindowMask(8),
    "causal_from_bottom_right": CausalFromBottomRightMask(_PARITY_S, _PARITY_S),
    "prefix_lm": PrefixLMMask(jnp.array([8, 8])),
    "block_diagonal": BlockDiagonalMask(jnp.zeros((_PARITY_B, _PARITY_S), jnp.int32)),
    "qkv_length": QKVLengthMask(_PARITY_S, _PARITY_S),
    "key_padding": KeyPaddingMask(jnp.array([32, 20])),
    "seq_len": SeqLenMask(jnp.array([32, 20])),
    "kv_len": KVLenMask(jnp.array([32, 20])),
    "same_segment": SameSegmentMask(
        jnp.zeros((_PARITY_B, _PARITY_S), jnp.int32),
        jnp.zeros((_PARITY_B, _PARITY_S), jnp.int32),
    ),
    "marginalization": MarginalizationMask(
        jnp.ones((_PARITY_B, _PARITY_S), dtype=bool)
    ),
}

_PARITY_BIASES = {
    "identity": IdentityBias(),
    "constant": ConstantBias(0.3),
    "dense": DenseBias(
        jax.random.normal(jax.random.key(1), (1, 1, _PARITY_S, _PARITY_S))
    ),
    "causal_alibi": CausalAlibiBias(),
    "symmetric_alibi": SymmetricAlibiBias(),
    "distance_decay": DistanceDecayBias(0.7),
    "soft_capping": SoftCappingBias(10.0),
}


@pytest.mark.parametrize("mask", _PARITY_MASKS.values(), ids=list(_PARITY_MASKS))
def test_kernel_masks_match_mha(mask):
    _assert_matches_mha(mask=mask)


@pytest.mark.parametrize("bias", _PARITY_BIASES.values(), ids=list(_PARITY_BIASES))
def test_kernel_biases_match_mha(bias):
    _assert_matches_mha(bias=bias)


@pytest.mark.parametrize(
    "mask_name,bias_name",
    [
        ("causal", "dense"),
        ("kv_len", "causal_alibi"),
        ("local_window", "identity"),
        ("same_segment", "constant"),
    ],
)
def test_kernel_mask_and_bias_together_match_mha(mask_name, bias_name):
    """Mask and bias are two pytrees; the leaf split must not confuse them."""
    _assert_matches_mha(mask=_PARITY_MASKS[mask_name], bias=_PARITY_BIASES[bias_name])


def test_kernel_absorbed_masks_match_mha():
    """Absorbed (K=V=C) must honour masks identically to mha."""
    q, cache, _wk, _wv = _parity_inputs()
    q = jax.random.normal(
        jax.random.key(9), (_PARITY_B, _PARITY_S, _PARITY_H, _PARITY_R)
    )
    cache3 = cache  # rank == q head dim for absorbed
    key = jnp.broadcast_to(
        cache3[..., None, :], cache3.shape[:-1] + (_PARITY_H, _PARITY_R)
    )
    block_sizes = _block_sizes(_PARITY_S)
    sm_scale = 1.0 / math.sqrt(_PARITY_R)
    for mask in _PARITY_MASKS.values():
        got = mla(
            q,
            cache3,
            None,
            None,
            mask=mask,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            interpret=True,
        )
        want = mha(
            q,
            key,
            key,
            mask=mask,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            interpret=True,
        )
        assert jnp.allclose(got, want, atol=1e-5)


def test_kernel_dropout_forward_runs():
    q, cache, wk, wv = _parity_inputs()
    out = mla(
        q,
        cache,
        wk,
        wv,
        rng=jax.random.key(7),
        dropout_rate=0.3,
        block_sizes=_block_sizes(_PARITY_S),
        interpret=True,
    )
    assert out.shape == (_PARITY_B, _PARITY_S, _PARITY_H, _PARITY_D)
    assert jnp.isfinite(out).all()


@pytest.mark.parametrize("dropout_impl", ["counter", "materialize"])
def test_kernel_dropout_gradients_match_mha(dropout_impl):
    """Gradients through dropout must run and match ``mha``.

    The counter-based backward had an undefined ``span_k`` when no mask/bias was
    present, so a forward-only dropout test passed while training crashed.
    """
    q, cache, wk, wv = _parity_inputs()
    key = jnp.einsum("bsr,hdr->bshd", cache, wk)
    value = jnp.einsum("bsr,hdr->bshd", cache, wv)
    block_sizes = _block_sizes(_PARITY_S)
    sm_scale = 1.0 / math.sqrt(_PARITY_D)
    rng = jax.random.key(7)

    def via_mla(q_):
        return mla(
            q_,
            cache,
            wk,
            wv,
            rng=rng,
            dropout_rate=0.3,
            dropout_impl=dropout_impl,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            interpret=True,
        ).sum()

    def via_mha(q_):
        return mha(
            q_,
            key,
            value,
            rng=rng,
            dropout_rate=0.3,
            dropout_impl=dropout_impl,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            interpret=True,
        ).sum()

    got = jax.grad(via_mla)(q)
    want = jax.grad(via_mha)(q)
    assert jnp.isfinite(got).all()
    assert jnp.allclose(got, want, atol=1e-5)


# ---------------------------------------------------------------------------
# Layer fused path: dropout resolution and batch-rank handling
# ---------------------------------------------------------------------------


def _layer(use_kernel=False, **kwargs):
    # materialized: exercises the per-head K/V path the fused kernel fuses.
    return MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        mode="materialized",
        qk_head_dim=8,
        v_head_dim=8,
        rngs=nnx.Rngs(0),
        use_kernel=use_kernel,
        **kwargs,
    )


def test_layer_kernel_deterministic_eval_needs_no_rng():
    """dropout_rate>0 but deterministic=True must not require a key."""
    layer = _layer(use_kernel=True, block_sizes=_block_sizes(32), dropout_rate=0.3)
    x = jax.random.normal(jax.random.key(1), (2, 32, 32))
    assert jnp.isfinite(layer(x, deterministic=True)).all()


def test_layer_kernel_training_uses_rng():
    layer = _layer(use_kernel=True, block_sizes=_block_sizes(32), dropout_rate=0.3)
    x = jax.random.normal(jax.random.key(1), (2, 32, 32))
    assert jnp.isfinite(layer(x, deterministic=False, rng=jax.random.key(2))).all()


def test_layer_kernel_handles_extra_batch_dims():
    """The kernel takes rank-4 q / rank-3 cache; leading dims must fold in."""
    layer = _layer(use_kernel=True, block_sizes=_block_sizes(32))
    x = jax.random.normal(jax.random.key(1), (2, 3, 32, 32))
    assert layer(x).shape == (2, 3, 32, 32)


def test_layer_use_kernel_with_custom_attention_fn_raises():
    from probjax.nn.layers.attention import dot_product_attention

    with pytest.raises(ValueError, match="use_kernel"):
        MultiHeadLatentAttention(
            num_heads=4,
            in_features=32,
            kv_lora_rank=16,
            qk_head_dim=8,
            v_head_dim=8,
            rngs=nnx.Rngs(0),
            use_kernel=True,
            attention_fn=dot_product_attention,
        )


def test_layer_does_not_sow_attention_weights():
    """MLA must not pass ``module=self`` to the attention function.

    Flax attention would then sow the attention weights into the module, which
    fails under ``jax.checkpoint`` / ``lax.scan`` (the module is a trace-level
    mismatch).  Weight sowing is unsupported here, so the layer must not opt in.
    """
    from probjax.nn.layers.attention import dot_product_attention

    layer = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        attention_fn=dot_product_attention,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (3, 2, 8, 32))

    def step(carry, z):
        return carry, layer(z)

    _, ys = jax.lax.scan(step, None, x)
    assert ys.shape == (3, 2, 8, 32)

    # Materialized mode takes the same call path and must also survive scan.
    materialized = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        mode="materialized",
        qk_head_dim=8,
        v_head_dim=8,
        attention_fn=dot_product_attention,
        rngs=nnx.Rngs(0),
    )

    def step_m(carry, z):
        return carry, materialized(z)

    _, ys_m = jax.lax.scan(step_m, None, x)
    assert ys_m.shape == (3, 2, 8, 32)


def test_layer_kernel_accepts_bias_like_flex():
    """The kernel handles additive bias, so the layer must forward it."""
    dense = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        mode="materialized",
        qk_head_dim=8,
        v_head_dim=8,
        rngs=nnx.Rngs(0),
    )
    fused = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        mode="materialized",
        qk_head_dim=8,
        v_head_dim=8,
        rngs=nnx.Rngs(1),
        use_kernel=True,
        block_sizes=_block_sizes(32),
    )
    nnx.update(fused, nnx.state(dense))
    x = jax.random.normal(jax.random.key(1), (2, 32, 32))
    bias = DenseBias(jax.random.normal(jax.random.key(2), (2, 4, 32, 32)))
    assert jnp.allclose(dense(x, bias=bias), fused(x, bias=bias), atol=1e-4)


def test_layer_kv_len_is_accepted_without_query_scale():
    """kv_len is now consumed by the optional query-scale module.

    With no query-scale module it is a no-op (it must not raise), so layers can
    be called uniformly whether or not they scale queries.
    """
    layer = _layer(use_kernel=True, block_sizes=_block_sizes(32))
    x = jax.random.normal(jax.random.key(1), (2, 32, 32))
    assert jnp.allclose(layer(x, kv_len=32), layer(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Bank sharing: one encoder, several consuming layers
# ---------------------------------------------------------------------------


def _absorbed_layer(kv_encoder=True, seed=0, **kwargs):
    return MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        kv_encoder=kv_encoder,
        rngs=nnx.Rngs(seed),
        **kwargs,
    )


def _copy_shared_params(dst, src):
    """Copy only the parameters the two layers share.

    A ``kv_encoder=False`` layer has no ``wkv_a``/``latent_norm``, so a plain
    ``nnx.update`` would try to set those (None) subtrees and fail.
    """
    src_state = nnx.state(src)
    dst_state = nnx.state(dst)
    shared = {k: v for k, v in src_state.items() if k in dst_state}
    nnx.update(dst, shared)


def _absorbed_reference(layer, x, latent=None, num_heads=4):
    """Plain softmax attention over the shared latent (K = V = C)."""
    lat = layer.encode_latent(x) if latent is None else latent
    q = layer._project_queries(x)
    rank = lat.shape[-1]
    kv = jnp.broadcast_to(lat[..., None, :], lat.shape[:-1] + (num_heads, rank))
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, kv) / math.sqrt(q.shape[-1])
    attn = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, -1), kv)
    return layer._out_from_attention(attn)


def test_layer_consumer_has_no_encoder_params():
    layer = _absorbed_layer(kv_encoder=False)
    assert layer.wkv_a is None
    assert layer.latent_norm is None
    with pytest.raises(RuntimeError, match="kv_encoder=False"):
        layer.encode_latent(jax.random.normal(jax.random.key(0), (2, 5, 32)))


def test_layer_shared_bank_matches_encoding_layer():
    """A consumer fed a precomputed bank equals one that encodes it itself."""
    encoder = _absorbed_layer()
    consumer = _absorbed_layer(kv_encoder=False)
    _copy_shared_params(consumer, encoder)  # same q/out weights

    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    latent = encoder.encode_latent(x)
    out = consumer(x, latent)
    assert jnp.allclose(out, _absorbed_reference(encoder, x), atol=1e-5)
    assert jnp.allclose(out, encoder(x), atol=1e-5)


def test_layer_consumer_rejects_wrong_bank_width():
    layer = _absorbed_layer(kv_encoder=False)
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    with pytest.raises(ValueError, match="latent dimension"):
        layer(x, jax.random.normal(jax.random.key(2), (2, 5, 8)))


def test_two_layers_share_one_bank():
    """The point of sharing: encode once, attend from two distinct layers."""
    encoder = _absorbed_layer()
    layer_a = _absorbed_layer(kv_encoder=False, seed=1)
    layer_b = _absorbed_layer(kv_encoder=False, seed=2)
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    latent = encoder.encode_latent(x)

    out_a = layer_a(x, latent)
    out_b = layer_b(x, latent)
    assert out_a.shape == x.shape and out_b.shape == x.shape
    assert jnp.isfinite(out_a).all() and jnp.isfinite(out_b).all()
    # Distinct layers must produce distinct outputs from the same bank.
    assert not jnp.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# Head folding: folded and unfolded absorbed forms are identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_kernel", [False, True])
def test_layer_folded_heads_match_unfolded(use_kernel):
    """Folding heads into the query axis must not change the result."""
    kwargs = dict(
        block_sizes=_block_sizes(32),
        use_kernel=use_kernel,
    )
    folded = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        fold_heads=True,
        rngs=nnx.Rngs(0),
        **kwargs,
    )
    unfolded = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        fold_heads=False,
        rngs=nnx.Rngs(1),
        **kwargs,
    )
    nnx.update(unfolded, nnx.state(folded))
    x = jax.random.normal(jax.random.key(1), (2, 32, 32))
    assert jnp.allclose(folded(x), unfolded(x), atol=1e-5)

    g_fold = nnx.grad(lambda m, z: m(z).sum(), argnums=1)(folded, x)
    g_unfold = nnx.grad(lambda m, z: m(z).sum(), argnums=1)(unfolded, x)
    assert jnp.allclose(g_fold, g_unfold, atol=1e-5)


def test_layer_folded_heads_pack_roundtrip():
    """_pack_heads / _unpack_heads are exact inverses in the documented order."""
    layer = _absorbed_layer()
    q = jax.random.normal(jax.random.key(0), (2, 5, 4, 16))
    packed = layer._pack_heads(q)
    assert packed.shape == (2, 20, 16)
    # head-major within a token: packed[i*H + h] == q[i, h]
    assert jnp.allclose(packed[0, 2 * 4 + 3], q[0, 2, 3])
    restored = layer._unpack_heads(packed, 5)
    assert jnp.allclose(restored, q)


def test_layer_folded_heads_falls_back_with_mask():
    """Masks live in token space, so they force the unfolded form (same answer)."""
    folded = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        fold_heads=True,
        rngs=nnx.Rngs(0),
    )
    unfolded = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        fold_heads=False,
        rngs=nnx.Rngs(1),
    )
    nnx.update(unfolded, nnx.state(folded))
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    mask = CausalMask()
    assert jnp.allclose(folded(x, mask=mask), unfolded(x, mask=mask), atol=1e-5)


@pytest.mark.parametrize("seq", [48, 32])
def test_layer_folded_kernel_handles_non_tiling_query_length(seq):
    """Folding multiplies Q by H; the kernel needs that to tile block_q.

    With Q=48 and H=4 the folded length is 192, which does not tile the default
    block_q=128, so the layer must fall back to the unfolded kernel instead of
    raising. The fallback must still match the unfolded layer exactly.
    """
    folded = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        fold_heads=True,
        use_kernel=True,
        rngs=nnx.Rngs(0),
    )
    unfolded = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        fold_heads=False,
        use_kernel=True,
        rngs=nnx.Rngs(1),
    )
    nnx.update(unfolded, nnx.state(folded))
    x = jax.random.normal(jax.random.key(1), (2, seq, 32))
    out = folded(x)  # must not raise
    assert out.shape == x.shape
    assert jnp.allclose(out, unfolded(x), atol=1e-4)


# ---------------------------------------------------------------------------
# No value projection (absorbed): attended latent maps straight to out
# ---------------------------------------------------------------------------


def test_layer_without_value_projection_matches_reference():
    layer = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        value_projection=False,
        rngs=nnx.Rngs(0),
    )
    assert layer.wv is None
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    out = layer(x)
    assert out.shape == x.shape and jnp.isfinite(out).all()
    # Reference: attended latent (no wv) -> out.
    latent = layer.encode_latent(x)
    q = layer._project_queries(x)
    kv = jnp.broadcast_to(latent[..., None, :], latent.shape[:-1] + (4, 16))
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, kv) / math.sqrt(16)
    attn = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, -1), kv)
    expected = layer.out(attn.reshape(2, 5, 4 * 16))
    assert jnp.allclose(out, expected, atol=1e-5)


def test_layer_without_value_projection_has_fewer_params():
    def nparams(m):
        return sum(p.size for p in jax.tree.leaves(nnx.state(m, nnx.Param)))

    with_wv = MultiHeadLatentAttention(
        num_heads=4, in_features=32, kv_lora_rank=16, rngs=nnx.Rngs(0)
    )
    without = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        value_projection=False,
        rngs=nnx.Rngs(0),
    )
    assert nparams(without) < nparams(with_wv)


def test_layer_value_projection_requires_absorbed_mode():
    with pytest.raises(ValueError, match="value_projection=False"):
        MultiHeadLatentAttention(
            num_heads=4,
            in_features=32,
            kv_lora_rank=16,
            mode="materialized",
            qk_head_dim=8,
            v_head_dim=8,
            value_projection=False,
            rngs=nnx.Rngs(0),
        )


def test_layer_without_value_projection_rejects_mismatched_v_head_dim():
    """Without wv the output stays kv_lora_rank-wide; a different v_head_dim
    cannot be honoured, so it must raise at construction (not reshape at run)."""
    with pytest.raises(ValueError, match="v_head_dim must equal"):
        MultiHeadLatentAttention(
            num_heads=4,
            in_features=32,
            kv_lora_rank=16,
            v_head_dim=8,
            value_projection=False,
            rngs=nnx.Rngs(0),
        )


# ---------------------------------------------------------------------------
# Query scaling (QASSMax)
# ---------------------------------------------------------------------------


def test_layer_qassmax_is_identity_at_init():
    plain = _absorbed_layer()
    scaled = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        query_scale_cls=QASSMaxQueryScale,
        rngs=nnx.Rngs(0),
    )
    nnx.update(scaled, nnx.state(plain))
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    # Zero-init base/gate: scaling must be exactly identity at init.
    assert jnp.allclose(plain(x), scaled(x, kv_len=5), atol=1e-6)


def test_layer_qassmax_changes_output_when_scales_move():
    scaled = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        query_scale_cls=QASSMaxQueryScale,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    before = scaled(x)
    # Nudge the scale parameters away from identity.
    scaled._query_scale.base_mlp.proj.kernel[...] += 0.5
    after = scaled(x, kv_len=5)
    assert not jnp.allclose(before, after)


def test_layer_qassmax_uses_kv_len_from_bank():
    """With a shared bank, kv_len defaults to the bank length."""
    scaled = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        kv_encoder=False,
        query_scale_cls=QASSMaxQueryScale,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    bank = jax.random.normal(jax.random.key(2), (2, 9, 16))
    # Should not need an explicit kv_len and must run with the bank's length.
    assert scaled(x, bank).shape == x.shape


def test_layer_qassmax_scale_receives_supplied_kv_len():
    """An explicit kv_len must reach the scaling module."""
    seen = {}

    class SpyScale(QASSMaxQueryScale):
        def __call__(self, query, *, kv_len):
            seen["kv_len"] = kv_len
            return super().__call__(query, kv_len=kv_len)

    layer = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        query_scale_cls=SpyScale,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    layer(x, kv_len=11)
    assert int(seen["kv_len"]) == 11


def test_layer_kernel_dropout_gradients_run():
    """Training through the fused path with dropout must not crash."""
    layer = _layer(use_kernel=True, block_sizes=_block_sizes(32), dropout_rate=0.3)
    x = jax.random.normal(jax.random.key(1), (2, 32, 32))

    def loss(layer_, x_):
        out = layer_(x_, deterministic=False, rng=jax.random.key(2))
        return jnp.sum(out**2)

    grad = nnx.grad(loss, argnums=1)(layer, x)
    assert jnp.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Low-rank projection (LoRA factorization as a usable projection)
# ---------------------------------------------------------------------------


class TestLowRankLinear:
    def test_is_a_bottleneck_not_a_full_matrix(self):
        layer = LowRankLinear(32, 64, rank=8, rngs=nnx.Rngs(0))
        params = sum(p.size for p in jax.tree.leaves(nnx.state(layer, nnx.Param)))
        assert params == 32 * 8 + 8 * 64
        assert params < 32 * 64  # never stores the full matrix

    def test_effective_rank_equals_rank(self):
        import numpy as np

        layer = LowRankLinear(32, 64, rank=8, rngs=nnx.Rngs(0))
        eff = np.asarray(layer.down.kernel[...]) @ np.asarray(layer.up.kernel[...])
        assert np.linalg.matrix_rank(eff) <= 8

    def test_is_not_zero_at_init(self):
        """Unlike nnx.LoRA (zero lora_b), a projection must carry signal."""
        layer = LowRankLinear(32, 64, rank=8, rngs=nnx.Rngs(0))
        x = jnp.ones((1, 32))
        assert not jnp.allclose(layer(x), 0.0)

    def test_gradients_are_finite(self):
        layer = LowRankLinear(32, 64, rank=8, norm_cls=nnx.RMSNorm, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(0), (2, 5, 32))
        grad = nnx.grad(lambda m, z: (m(z) ** 2).sum(), argnums=1)(layer, x)
        assert grad.shape == x.shape
        assert jnp.isfinite(grad).all()

    def test_rank_larger_than_min_dim_raises(self):
        with pytest.raises(ValueError, match="not a bottleneck"):
            LowRankLinear(8, 4, rank=16, rngs=nnx.Rngs(0))


def test_layer_factored_query_reduces_params():
    """q_lora_rank routes the query through LowRankLinear (factors only)."""
    plain = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        mode="materialized",
        qk_head_dim=8,
        v_head_dim=8,
        rngs=nnx.Rngs(0),
    )
    factored = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        mode="materialized",
        qk_head_dim=8,
        v_head_dim=8,
        q_lora_rank=8,
        rngs=nnx.Rngs(0),
    )
    assert isinstance(factored.q_projection, LowRankLinear)

    def nparams(m):
        return sum(p.size for p in jax.tree.leaves(nnx.state(m, nnx.Param)))

    assert nparams(factored) < nparams(plain)
    x = jax.random.normal(jax.random.key(1), (2, 5, 32))
    assert factored(x).shape == plain(x).shape


def test_layer_q_projection_cls_requires_q_lora_rank():
    with pytest.raises(ValueError, match="q_projection_cls requires q_lora_rank"):
        MultiHeadLatentAttention(
            num_heads=4,
            in_features=32,
            kv_lora_rank=16,
            q_projection_cls=LowRankLinear,
            rngs=nnx.Rngs(0),
        )


def test_layer_q_projection_cls_override_is_used():
    """A custom projection class must actually be instantiated."""

    class SpyLowRank(LowRankLinear):
        pass

    layer = MultiHeadLatentAttention(
        num_heads=4,
        in_features=32,
        kv_lora_rank=16,
        mode="materialized",
        qk_head_dim=8,
        v_head_dim=8,
        q_lora_rank=8,
        q_projection_cls=SpyLowRank,
        rngs=nnx.Rngs(0),
    )
    assert isinstance(layer.q_projection, SpyLowRank)
