from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.layers.attention import (
    dot_product_attention,
    dot_product_attention_jax,
    flex_attention,
    memory_efficient_dot_product_attention,
)
from probjax.nn.pallas_kernels.attention_mask_bias import (
    CausalMask,
    KeyPaddingMask,
    LocalWindowMask,
    MarginalizationMask,
    NoMask,
    QKVLengthMask,
    SameSegmentMask,
)
from probjax.nn.pallas_kernels.utils import materialize_mask


@pytest.fixture(
    params=[
        CausalMask,
        LocalWindowMask,
        NoMask,
        KeyPaddingMask,
        QKVLengthMask,
        # SameSegmentMask,
        MarginalizationMask,
    ]
)
def mask_fn(request):
    def mask_builder(q_len, k_len):
        if request.param in {CausalMask, NoMask}:
            return request.param()
        elif request.param == LocalWindowMask:
            left_window_size = int(np.random.randint(1, q_len))
            right_window_size = int(np.random.randint(1, k_len))
            return request.param(left_window_size, right_window_size)
        elif request.param == QKVLengthMask:
            q_lengths = int(np.random.randint(1, q_len + 1))
            k_lengths = int(np.random.randint(1, k_len + 1))
            return request.param(q_lengths, k_lengths)
        elif request.param == KeyPaddingMask:
            key_lengths = np.random.randint(1, k_len + 1, size=(k_len,))
            return request.param(jnp.asarray(key_lengths))
        elif request.param == SameSegmentMask:
            segment_ids = jnp.asarray(np.random.randint(0, 5, size=(q_len,)))
            segment_ids_k = jnp.asarray(np.random.randint(0, 5, size=(k_len,)))
            return request.param(segment_ids, segment_ids_k)
        elif request.param == MarginalizationMask:
            marginalize = np.random.choice([True, False], size=(q_len,))
            return request.param(jnp.asarray(marginalize))

    return mask_builder


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
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
    ],
)
def test_attention_functions(attention_fn, batch_size, seq_len, num_heads, qkv_dim):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    out = attention_fn(q, k, v)
    assert out.shape == (batch_size, seq_len, num_heads, qkv_dim)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_function_outputs_are_same(batch_size, seq_len, num_heads, qkv_dim):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
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
        assert jnp.allclose(outputs[0], outputs[i], atol=1e-5), (
            f"Outputs are not same for {attention_fns[i]}"
        )


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_function_gradients_are_same(batch_size, seq_len, num_heads, qkv_dim):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
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
        for g1, g2 in zip(grads[0], grads[i], strict=False):
            assert jnp.allclose(g1, g2, atol=1e-3), (
                f"Gradients are not same for {attention_fns[i]}"
                f" error is {jnp.mean(jnp.abs(g1 - g2))}, std {jnp.std(g1 - g2)}"
            )


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_with_dropout(batch_size, seq_len, num_heads, qkv_dim):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    out1 = flex_attention(
        q,
        k,
        v,
        dropout_rate=0.1,
        deterministic=False,
        dropout_rng=jax.random.PRNGKey(0),
    )
    assert out1.shape == (batch_size, seq_len, num_heads, qkv_dim)
    out2 = flex_attention(
        q,
        k,
        v,
        dropout_rate=0.1,
        deterministic=False,
        dropout_rng=jax.random.PRNGKey(1),
    )
    assert out2.shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert not jnp.allclose(out1, out2, atol=1e-5), "Dropout did not change the output"


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_with_masks(batch_size, seq_len, num_heads, qkv_dim, mask_fn):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    mask = mask_fn(seq_len, seq_len)
    mask_dense = materialize_mask(mask, seq_len, seq_len)
    out = dot_product_attention(q, k, v, mask=mask_dense)
    assert out.shape == (batch_size, seq_len, num_heads, qkv_dim)
    out2 = flex_attention(q, k, v, mask)
    assert out2.shape == (batch_size, seq_len, num_heads, qkv_dim)
    if isinstance(mask, QKVLengthMask):
        # If stuff is completly gone including the diagonal they behave a bit differently.
        assert jnp.allclose(
            out[:, : mask.q_length], out2[:, : mask.q_length], atol=1e-4
        )
    else:
        assert jnp.allclose(out, out2, atol=1e-5), (
            f"Outputs are not same with mask, with error {jnp.max(jnp.abs(out - out2))}"
        )


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_gradient_with_masks(
    batch_size, seq_len, num_heads, qkv_dim, mask_fn
):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    mask = mask_fn(seq_len, seq_len)
    mask_dense = materialize_mask(mask, seq_len, seq_len)

    def loss_fn1(params):
        q, k, v = params
        out = dot_product_attention(q, k, v, mask=mask_dense)
        return jnp.sum(out**2)

    def loss_fn2(params):
        q, k, v = params
        out = flex_attention(q, k, v, mask)
        return jnp.sum(out**2)

    out = jax.grad(loss_fn1)((q, k, v))
    out2 = jax.grad(loss_fn2)((q, k, v))
    assert out[0].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out[1].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out[2].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out2[0].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out2[1].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out2[2].shape == (batch_size, seq_len, num_heads, qkv_dim)

    # Different whole rows are masked out
    if isinstance(mask, QKVLengthMask):
        return

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(partial(jnp.allclose, atol=1e-2), out, out2)
    )
