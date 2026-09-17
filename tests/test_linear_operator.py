"""Dense-oracle regressions for operator composition and compiled filtering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.utils.linear_operator import LinearOperator
from probjax.inference.filtering.kalman_filter import kalman_filter


@pytest.mark.parametrize("compiled", [False, True])
def test_rectangular_products(compiled):
    a = jnp.array([[1.0, 2.0, 3.0], [-1.0, 4.0, 2.0]])
    right = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4)
    left = jnp.array([[2.0, -1.0], [1.0, 3.0], [0.0, 4.0]])
    x, y = jnp.array([1.0, -2.0, 4.0]), jnp.array([2.0, 3.0])

    def products(a, right, left, x, y):
        op = LinearOperator.from_array(a)
        assert op.shape == a.shape
        assert op.T.shape == a.T.shape
        return (
            op @ x,
            y @ op,
            (op @ right).as_array(),
            (left @ op).as_array(),
            (op @ LinearOperator.from_array(right)).as_array(),
        )

    actual = (jax.jit(products) if compiled else products)(a, right, left, x, y)
    expected = (a @ x, y @ a, a @ right, left @ a, a @ right)
    for result, oracle in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, oracle, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("observed", [False, True])
def test_compiled_operator_filter_matches_dense(observed):
    a = jnp.array([[0.9, 0.3], [-0.1, 0.8]])
    q = jnp.array([[0.1, 0.02], [0.02, 0.2]])
    c = jnp.array([[1.0, 2.0]])
    r = jnp.array([[0.3]])
    mean = jnp.array([1.0, -0.5])
    cov = jnp.array([[0.7, 0.2], [0.2, 1.3]])
    y = jnp.array([0.8]) if observed else None

    def run(a, operator):
        convert = LinearOperator.from_array if operator else lambda x: x
        kernel = kalman_filter(
            lambda t0, t: (convert(a), convert(q)),
            lambda t: (convert(c), r),
        )
        return kernel.step(kernel.init(mean, cov, t=0.0), t=1.0, observed=y)

    dense = jax.jit(lambda a: run(a, False))(a)
    actual = jax.jit(lambda a: run(a, True))(a)
    for result, oracle in zip(
        jax.tree.leaves(actual), jax.tree.leaves(dense), strict=True
    ):
        np.testing.assert_allclose(result, oracle, rtol=2e-5, atol=2e-6)
    loss = lambda a, operator: jnp.sum(run(a, operator)[0].cov)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda a: loss(a, True)))(a),
        jax.jit(jax.grad(lambda a: loss(a, False)))(a),
        rtol=2e-5,
        atol=2e-6,
    )
