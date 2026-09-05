import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from probjax.nn.layers.attention import dot_product_attention, flex_attention
from probjax.nn.pallas_kernels import (
    CausalAlibiBias,
    CausalMask,
    LearnedAlibiBias,
    LowRankBias,
    T5RelativePositionBias,
)


def _attention_inputs():
    keys = jax.random.split(jax.random.key(0), 5)
    q = jax.random.normal(keys[0], (1, 16, 2, 8))
    k = jax.random.normal(keys[1], (1, 16, 2, 8))
    v = jax.random.normal(keys[2], (1, 16, 2, 8))
    query_factors = jax.random.normal(keys[3], (2, 16, 3))
    key_factors = jax.random.normal(keys[4], (2, 16, 3))
    return q, k, v, query_factors, key_factors


class _MLPPositionBias(nnx.Module):
    def __init__(self, num_heads: int, rank: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.rank = rank
        self.query_mlp = nnx.Linear(4, num_heads * rank, rngs=rngs)
        self.key_mlp = nnx.Linear(4, num_heads * rank, rngs=rngs)

    def __call__(self, positions: jax.Array) -> LowRankBias:
        query_factors = self.query_mlp(positions)
        key_factors = self.key_mlp(positions)
        query_factors = query_factors.reshape(
            positions.shape[0], self.num_heads, self.rank
        )
        key_factors = key_factors.reshape(positions.shape[0], self.num_heads, self.rank)
        return LowRankBias(
            jnp.swapaxes(query_factors, 0, 1),
            jnp.swapaxes(key_factors, 0, 1),
            scale=1.0 / self.rank,
        )


def test_low_rank_bias_matches_explicit_factor_product():
    _, _, _, query_factors, key_factors = _attention_inputs()
    bias = LowRankBias(query_factors, key_factors, scale=0.25)
    actual = bias.dense(16, 16, num_heads=2)
    expected = 0.25 * jnp.einsum("hqr,hkr->hqk", query_factors, key_factors)
    np.testing.assert_allclose(actual[0], expected, atol=1e-6, rtol=1e-6)


def test_flex_low_rank_bias_matches_dense_output_and_gradients():
    q, k, v, query_factors, key_factors = _attention_inputs()

    def loss(attention_fn, query_factor, key_factor):
        bias = LowRankBias(query_factor, key_factor, scale=0.5)
        output = attention_fn(q, k, v, bias=bias)
        return jnp.mean(output**2)

    dense_output = dot_product_attention(
        q, k, v, bias=LowRankBias(query_factors, key_factors, scale=0.5)
    )
    flex_output = flex_attention(
        q, k, v, bias=LowRankBias(query_factors, key_factors, scale=0.5)
    )
    np.testing.assert_allclose(flex_output, dense_output, atol=2e-5, rtol=2e-5)

    dense_grads = jax.grad(loss, argnums=(1, 2))(
        dot_product_attention, query_factors, key_factors
    )
    flex_grads = jax.grad(loss, argnums=(1, 2))(
        flex_attention, query_factors, key_factors
    )
    for flex_grad, dense_grad in zip(flex_grads, dense_grads, strict=True):
        np.testing.assert_allclose(flex_grad, dense_grad, atol=3e-5, rtol=3e-5)


def test_flex_extracts_low_rank_terms_from_sum_bias():
    q, k, v, query_factors, key_factors = _attention_inputs()
    bias = LowRankBias(query_factors, key_factors, scale=0.25) + CausalAlibiBias()
    expected = dot_product_attention(q, k, v, bias=bias)
    actual = flex_attention(q, k, v, bias=bias)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_low_rank_factors_can_be_generated_by_an_nnx_mlp():
    q, k, v, _, _ = _attention_inputs()
    module = _MLPPositionBias(num_heads=2, rank=3, rngs=nnx.Rngs(0))
    positions = jnp.stack(
        (
            jnp.linspace(-1, 1, 16),
            jnp.sin(jnp.arange(16)),
            jnp.cos(jnp.arange(16)),
            jnp.ones(16),
        ),
        axis=-1,
    )

    gradient = nnx.grad(
        lambda bias_module: jnp.mean(
            flex_attention(q, k, v, bias=bias_module(positions)) ** 2
        )
    )(module)
    assert jnp.any(gradient.query_mlp.kernel[...] != 0)
    assert jnp.any(gradient.key_mlp.kernel[...] != 0)


def test_learned_alibi_matches_causal_linear_distance():
    slopes = jnp.array([0.5, 0.25], dtype=jnp.float32)
    dense = LearnedAlibiBias(slopes).dense(8, 8, num_heads=2)[0]
    positions = jnp.arange(8)
    distance = positions[:, None] - positions[None, :]
    expected = -slopes[:, None, None] * distance[None]
    np.testing.assert_allclose(dense, expected, atol=1e-6, rtol=1e-6)


def test_learned_alibi_receives_gradients_through_flex_attention():
    q, k, v, _, _ = _attention_inputs()

    def loss(log_slopes):
        return jnp.mean(
            flex_attention(
                q,
                k,
                v,
                mask=CausalMask(),
                bias=LearnedAlibiBias(jnp.exp(log_slopes)),
            )
            ** 2
        )

    gradient = jax.grad(loss)(jnp.log(jnp.array([0.5, 0.25])))
    assert gradient.shape == (2,)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0)


def test_t5_table_receives_gradients_through_dense_attention():
    q, k, v, _, _ = _attention_inputs()
    table = jnp.zeros((2, 16), dtype=jnp.float32)

    def loss(bias_table):
        bias = T5RelativePositionBias(
            bias_table,
            num_buckets=16,
            max_distance=8,
            bidirectional=True,
        )
        return jnp.mean(dot_product_attention(q, k, v, bias=bias) ** 2)

    gradient = jax.grad(loss)(table)
    assert gradient.shape == table.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0)
