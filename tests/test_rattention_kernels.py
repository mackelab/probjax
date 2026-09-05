import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.pallas_kernels import (
    FeatureMap,
    residual_linear_attention_pallas,
    residual_linear_attention_scan,
    right_shift_and_zero_pad,
)


def _inputs(*, num_heads=2, num_kv_heads=2):
    keys = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(keys[0], (1, num_heads, 8, 4))
    k = jax.random.normal(keys[1], (1, num_kv_heads, 8, 4))
    v = jax.random.normal(keys[2], (1, num_kv_heads, 8, 4))
    h0 = 0.1 * jax.random.normal(keys[3], (1, num_heads, 8, 4))
    return q, k, v, h0


def test_right_shift_and_zero_pad():
    x = jnp.arange(5)
    np.testing.assert_array_equal(
        right_shift_and_zero_pad(x, 2, axis=0), jnp.array([0, 0, 0, 1, 2])
    )
    np.testing.assert_array_equal(right_shift_and_zero_pad(x, 0, axis=0), x)


@pytest.mark.parametrize("feature_map", list(FeatureMap))
@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_pallas_interpreter_matches_scan(feature_map, num_kv_heads):
    inputs = _inputs(num_kv_heads=num_kv_heads)
    expected = residual_linear_attention_scan(
        *inputs, window_size=2, feature_map=feature_map
    )
    actual = residual_linear_attention_pallas(
        *inputs, window_size=2, feature_map=feature_map, chunk_size=4
    )
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual[1], expected[1], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("feature_map", list(FeatureMap))
def test_pallas_interpreter_gradient_matches_scan(feature_map):
    inputs = _inputs(num_kv_heads=1)

    def loss(implementation, q, k, v, h0):
        output, final_state = implementation(
            q,
            k,
            v,
            h0,
            window_size=2,
            feature_map=feature_map,
            **(
                {"chunk_size": 4}
                if implementation is residual_linear_attention_pallas
                else {}
            ),
        )
        return jnp.mean(output**2) + 0.01 * jnp.mean(final_state**2)

    expected = jax.grad(
        lambda *x: loss(residual_linear_attention_scan, *x), argnums=(0, 1, 2, 3)
    )(*inputs)
    actual = jax.grad(
        lambda *x: loss(residual_linear_attention_pallas, *x), argnums=(0, 1, 2, 3)
    )(*inputs)
    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=3e-5, rtol=3e-5)
