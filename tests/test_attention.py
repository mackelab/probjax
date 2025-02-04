import numpy as np
import jax
import jax.numpy as jnp
import pytest
from probjax.nn.attention import (
    dot_product_attention,
    flex_attention_fn,
    memory_efficient_dot_product_attention,
    attention_fn_jax,
)


@pytest.mark.parametrize(
    "attention_fn",
    [
        dot_product_attention,
        flex_attention_fn,
        memory_efficient_dot_product_attention,
        attention_fn_jax,
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


def test_attention_function_outputs_are_same():
    q = k = v = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4, 16))
    outputs = []
    attention_fns = [
        dot_product_attention,
        flex_attention_fn,
        memory_efficient_dot_product_attention,
        attention_fn_jax,
    ]
    for attention_fn in attention_fns:
        outputs.append(attention_fn(q, k, v))

    for i in range(1, len(outputs)):
        assert jnp.allclose(
            outputs[0], outputs[i], atol=1e-2
        ), f"Outputs are not same for {attention_fns[i]}"


def test_attention_function_gradients_are_same():
    q = k = v = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4, 16))
    attention_fns = [
        dot_product_attention,
        flex_attention_fn,
        memory_efficient_dot_product_attention,
        attention_fn_jax,
    ]
    grads = []
    for attention_fn in attention_fns:
        grad_fn = jax.grad(
            lambda q, k, v: jnp.sum(attention_fn(q, k, v)), argnums=(0, 1, 2)
        )
        grads.append(grad_fn(q, k, v))

    for i in range(1, len(grads)):
        for g1, g2 in zip(grads[0], grads[i]):
            assert jnp.allclose(
                g1, g2, atol=1e-2
            ), f"Gradients are not same for {attention_fns[i]}"


@pytest.mark.parametrize(
    "mask_fn",
    [
        lambda b, h, q_idx, k_idx: q_idx[None, :] >= k_idx[:, None],
        lambda b, h, q_idx, k_idx: q_idx[None, :] <= k_idx[:, None],
        lambda b, h, q_idx, k_idx: (q_idx[None, :] + k_idx[:, None]) % 2 == 0,
    ],
)
def test_flex_attention_masking(mask_fn):
    q = k = v = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4, 16))
    mask_initiated = mask_fn(2, 4, jnp.arange(16), jnp.arange(16))

    out = flex_attention_fn(q, k, v, mask=mask_fn)
    out2 = dot_product_attention(q, k, v, mask=mask_initiated)

    assert jnp.allclose(out, out2, atol=1e-2)
