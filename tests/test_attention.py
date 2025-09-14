import jax
import jax.numpy as jnp
import pytest

from probjax.nn.layers.attention import (
    dot_product_attention_jax,
    dot_product_attention,
    flex_attention,
    memory_efficient_dot_product_attention,
)
from probjax.nn.pallas_kernels.attention_mask_bias import (
    CausalMask,
    LocalWindowMask,
    KeyPaddingMask,
    SameSegmentMask,
    NoMask,
    ConstantBias,
    DenseBias,
    FromMaskBias,
    materialize_mask,
    materialize_bias,
)
from probjax.nn.pallas_kernels.attention import mha, BlockSizes


# @pytest.mark.gpu
@pytest.mark.parametrize(
    "attention_fn",
    [
        dot_product_attention,
        flex_attention,
        memory_efficient_dot_product_attention,
        dot_product_attention_jax,
    ],
)
@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 32, 8, 32),
        (1, 16, 2, 8),
        (3, 50, 8, 32),
    ],
)
def test_attention_functions(attention_fn, batch_size, seq_len, num_heads, qkv_dim):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    out = attention_fn(q, k, v)
    assert out.shape == (batch_size, seq_len, num_heads, qkv_dim)


# @pytest.mark.gpu
def test_attention_function_outputs_are_same():
    q = k = v = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4, 16))
    outputs = []
    attention_fns = [
        dot_product_attention,
        flex_attention,
        memory_efficient_dot_product_attention,
        dot_product_attention_jax,
    ]
    for attention_fn in attention_fns:
        outputs.append(attention_fn(q, k, v))

    for i in range(1, len(outputs)):
        assert jnp.allclose(outputs[0], outputs[i], atol=1e-2), (
            f"Outputs are not same for {attention_fns[i]}"
        )


def test_attention_function_gradients_are_same():
    q = k = v = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4, 16))
    attention_fns = [
        dot_product_attention,
        flex_attention,
        memory_efficient_dot_product_attention,
        dot_product_attention_jax,
    ]
    grads = []
    for attention_fn in attention_fns:
        grad_fn = jax.grad(
            lambda q, k, v: jnp.sum(attention_fn(q, k, v)), argnums=(0, 1, 2)
        )
        grads.append(grad_fn(q, k, v))

    for i in range(1, len(grads)):
        for g1, g2 in zip(grads[0], grads[i]):
            assert jnp.allclose(g1, g2, atol=1e-2), (
                f"Gradients are not same for {attention_fns[i]}"
                f" error is {jnp.mean(jnp.abs(g1 - g2))}, std {jnp.std(g1 - g2)}"
            )


# Deprecated: callable mask_fn tests removed. Use class-based masks below.


@pytest.mark.parametrize(
    "mask_builder",
    [
        lambda B, Q, K, H: NoMask(),
        lambda B, Q, K, H: CausalMask(),
        lambda B, Q, K, H: LocalWindowMask(left_window=4, right_window=2),
        lambda B, Q, K, H: KeyPaddingMask(key_lengths=jnp.array([K - 2 for _ in range(B)], dtype=jnp.int32)),
        lambda B, Q, K, H: SameSegmentMask(
            query_segment_ids=jnp.tile(jnp.arange(Q) % 3, (B, 1)),
            key_segment_ids=jnp.tile(jnp.arange(K) % 3, (B, 1)),
        ),
    ],
)
def test_flex_attention_supported_masks(mask_builder):
    B, Q, K, H, D = 2, 32, 32, 4, 16
    key = jax.random.normal(jax.random.PRNGKey(1), (B, K, H, D))
    val = jax.random.normal(jax.random.PRNGKey(2), (B, K, H, D))
    qry = jax.random.normal(jax.random.PRNGKey(3), (B, Q, H, D))
    mask_obj = mask_builder(B, Q, K, H)

    # flex_attention with mask object
    out_flex = flex_attention(qry, key, val, mask=mask_obj)

    # materialize dense boolean mask [B, H, Q, K] for baseline
    dense_mask = materialize_mask(mask_obj, batch_size=B, num_heads=H, q_len=Q, kv_len=K, segment_ids=None)
    out_ref = dot_product_attention(qry, key, val, mask=dense_mask)

    assert out_flex.shape == out_ref.shape
    assert jnp.allclose(out_flex, out_ref, atol=1e-2)

    # Gradients w.r.t. query
    g_flex = jax.grad(lambda x: jnp.sum(flex_attention(x, key, val, mask=mask_obj)))(qry)
    g_ref = jax.grad(lambda x: jnp.sum(dot_product_attention(x, key, val, mask=dense_mask)))(qry)
    assert jnp.allclose(g_flex, g_ref, atol=1e-2), "Gradients w.r.t. query do not match"


@pytest.mark.parametrize(
    "bias_builder",
    [
        lambda B, Q, K, H: ConstantBias(0.2),
        lambda B, Q, K, H: DenseBias(jax.random.normal(jax.random.PRNGKey(10), (1, 1, Q, K))),
        lambda B, Q, K, H: FromMaskBias(CausalMask()),
    ],
)
def test_flex_attention_supported_biases(bias_builder):
    B, Q, K, H, D = 2, 32, 32, 4, 16
    key = jax.random.normal(jax.random.PRNGKey(11), (B, K, H, D))
    val = jax.random.normal(jax.random.PRNGKey(12), (B, K, H, D))
    qry = jax.random.normal(jax.random.PRNGKey(13), (B, Q, H, D))

    bias_mod = bias_builder(B, Q, K, H)

    # flex_attention with bias modifier (callable or AttentionBiasBase)
    out_flex = flex_attention(qry, key, val, bias=bias_mod)

    # materialize bias [B, H, Q, K] for baseline
    if isinstance(bias_mod, DenseBias):
        dense_bias = bias_mod.bias  # underlying tensor
    elif isinstance(bias_mod, ConstantBias) or isinstance(bias_mod, FromMaskBias):
        dense_bias = materialize_bias(lambda s, b, h, qi, ki: bias_mod(s, b, h, qi, ki), B, H, Q, K)
    else:  # stateless callable
        dense_bias = materialize_bias(bias_mod, B, H, Q, K)

    out_ref = dot_product_attention(qry, key, val, bias=dense_bias)

    assert out_flex.shape == out_ref.shape
    # Bias functions can be large; allow slightly higher tolerance
    assert jnp.allclose(out_flex, out_ref, atol=2e-2)

    # Gradients w.r.t. query
    g_flex = jax.grad(lambda x: jnp.sum(flex_attention(x, key, val, bias=bias_mod)))(qry)
    g_ref = jax.grad(lambda x: jnp.sum(dot_product_attention(x, key, val, bias=dense_bias)))(qry)
    assert jnp.allclose(g_flex, g_ref, atol=2e-2)


def test_flex_attention_qkv_len_mismatch():
    # Different Q and KV sequence lengths
    B, Q, K, H, D = 2, 24, 32, 4, 16
    key = jax.random.normal(jax.random.PRNGKey(21), (B, K, H, D))
    val = jax.random.normal(jax.random.PRNGKey(22), (B, K, H, D))
    qry = jax.random.normal(jax.random.PRNGKey(23), (B, Q, H, D))

    # Use a simple mask to exercise block-sparse path
    mask_obj = CausalMask()
    out_flex = flex_attention(qry, key, val, mask=mask_obj)

    dense_mask = materialize_mask(mask_obj, batch_size=B, num_heads=H, q_len=Q, kv_len=K, segment_ids=None)
    out_ref = dot_product_attention(qry, key, val, mask=dense_mask)

    assert out_flex.shape == (B, Q, H, D)
    assert jnp.allclose(out_flex, out_ref, atol=2e-2)


def test_mha_with_densebias_and_blocksparse_toggle():
    B, Q, K, H, D = 2, 16, 16, 2, 8
    q = jax.random.normal(jax.random.PRNGKey(0), (B, Q, H, D))
    k = jax.random.normal(jax.random.PRNGKey(1), (B, K, H, D))
    v = jax.random.normal(jax.random.PRNGKey(2), (B, K, H, D))
    # Dense bias with broadcasting over batch or head
    dense_bias = jax.random.normal(jax.random.PRNGKey(3), (1, 1, Q, K))
    bias = DenseBias(dense_bias)
    mask = CausalMask()
    # blocks
    bs = BlockSizes.get_default()
    out_sparse = mha(q, k, v, mask=mask, bias_mod=bias, block_sizes=bs, block_sparse=True, interpret=True)
    out_dense = mha(q, k, v, mask=mask, bias_mod=bias, block_sizes=bs, block_sparse=False, interpret=True)
    assert out_sparse.shape == out_dense.shape == (B, Q, H, D)
    assert jnp.allclose(out_sparse, out_dense, atol=2e-2)


def test_mha_constant_bias_matches_reference():
    B, Q, K, H, D = 1, 8, 8, 2, 8
    q = jax.random.normal(jax.random.PRNGKey(10), (B, Q, H, D))
    k = jax.random.normal(jax.random.PRNGKey(11), (B, K, H, D))
    v = jax.random.normal(jax.random.PRNGKey(12), (B, K, H, D))
    bias = ConstantBias(0.1)
    import math
    sm_scale = 1.0 / math.sqrt(D)
    out_kernel = mha(q, k, v, mask=None, bias_mod=bias, sm_scale=sm_scale, interpret=True)
    dense_bias = materialize_bias(lambda s, b, h, qi, ki: bias(s, b, h, qi, ki), B, H, Q, K)
    out_ref = dot_product_attention(q, k, v, bias=dense_bias)
    assert jnp.allclose(out_kernel, out_ref, atol=2e-2)
