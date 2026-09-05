from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probjax.nn.layers.attention import dot_product_attention
from probjax.nn.layers.rattention import GroupRMSNorm, RAttention
from probjax.nn.nets.transformer import Transformer


def _make_attention(**kwargs):
    return RAttention(
        num_heads=2,
        in_features=8,
        qkv_features=8,
        out_features=8,
        sliding_window_size=2,
        chunk_size=4,
        attention_fn=dot_product_attention,
        rngs=nnx.Rngs(0),
        **kwargs,
    )


def test_group_rms_norm_normalizes_each_head():
    norm = GroupRMSNorm(2, 4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (2, 5, 2, 4))
    y = norm(x)
    np.testing.assert_allclose(jnp.mean(y**2, axis=-1), 1.0, atol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"residual_la": False},
        {"sliding_window_size": -1},
        {"use_learned_init": True},
        {"use_qk_scale": True},
    ],
)
def test_rattention_variants_forward_and_grad(kwargs):
    if "sliding_window_size" in kwargs:
        attention = RAttention(
            num_heads=2,
            in_features=8,
            qkv_features=8,
            out_features=8,
            chunk_size=4,
            attention_fn=dot_product_attention,
            rngs=nnx.Rngs(0),
            **kwargs,
        )
    else:
        attention = _make_attention(**kwargs)
    x = jax.random.normal(jax.random.key(1), (2, 8, 8))
    y = attention(x)
    grad = jax.grad(lambda inputs: jnp.sum(attention(inputs)))(x)
    assert y.shape == x.shape
    assert grad.shape == x.shape
    assert jnp.all(jnp.isfinite(y))
    assert jnp.all(jnp.isfinite(grad))


def test_rattention_uses_pallas_interpreter():
    attention = _make_attention(linear_attention_implementation="pallas")
    x = jax.random.normal(jax.random.key(1), (1, 8, 8))
    y = attention(x)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_rattention_supports_grouped_query_attention():
    attention = RAttention(
        num_heads=4,
        num_kv_heads=2,
        in_features=8,
        qkv_features=8,
        out_features=8,
        sliding_window_size=2,
        chunk_size=4,
        linear_attention_implementation="pallas",
        attention_fn=dot_product_attention,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (1, 8, 8))
    assert attention(x).shape == x.shape


def test_transformer_accepts_rattention_as_mha_class():
    transformer = Transformer(
        model_dim=8,
        num_heads=2,
        num_layers=1,
        attn_size=4,
        mha_cls=partial(RAttention, sliding_window_size=2, chunk_size=4),
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (2, 8, 8))
    y = transformer(x)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))
