import jax
import jax.numpy as jnp

from probjax.core import inverse, inverse_and_logabsdet
from probjax.core.registry import Context, REGISTRY


def test_inverse_dot_general_lhs():
    matrix = jnp.array(
        [[2.0, 0.2, -0.1], [0.0, 1.5, 0.3], [0.0, 0.0, 1.1]]
    )

    def f(x):
        return x @ matrix

    x = jnp.array([0.3, -1.2, 2.1])
    assert jnp.allclose(inverse(f)(f(x)), x, atol=1e-6, rtol=1e-6)


def test_inverse_dot_general_rhs():
    matrix = jnp.array(
        [[1.2, 0.1, -0.2], [0.0, 1.1, 0.3], [-0.4, 0.2, 0.9]]
    )

    def f(weights):
        return matrix @ weights

    weights = jnp.array([[0.4, -1.0], [1.2, 0.5], [-0.3, 2.0]])
    assert jnp.allclose(inverse(f)(f(weights)), weights, atol=1e-6, rtol=1e-6)


def test_inverse_dot_general_non_square_returns_nan():
    matrix = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def f(x):
        return x @ matrix

    x = jnp.array([0.2, -0.3, 1.4])
    eqn = jax.make_jaxpr(f)(x).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.dot_general_p, Context.INVERSE)
    recovered = rule(eqn, [None, matrix], [f(x)]).resolved_vals[0]
    assert recovered.shape == x.shape
    assert jnp.all(jnp.isnan(recovered))


def test_inverse_and_logabsdet_dot_general_lhs():
    matrix = jnp.array(
        [[1.8, 0.2, -0.1], [0.0, 1.3, 0.4], [0.0, 0.0, 0.9]]
    )

    def f(x):
        return x @ matrix

    x = jnp.array([0.2, -0.7, 1.3])
    recovered, logdet = inverse_and_logabsdet(f)(f(x))
    _, matrix_logdet = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(recovered, x, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, -matrix_logdet, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_dot_general_rhs():
    matrix = jnp.array(
        [[1.6, 0.1, -0.2], [0.0, 1.2, 0.3], [0.0, 0.0, 1.1]]
    )

    def f(weights):
        return matrix @ weights

    weights = jnp.array([[0.4, -1.0], [1.2, 0.5], [-0.3, 2.0]])
    recovered, logdet = inverse_and_logabsdet(f)(f(weights))
    _, matrix_logdet = jnp.linalg.slogdet(matrix)
    expected = -weights.shape[-1] * matrix_logdet
    assert jnp.allclose(recovered, weights, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(logdet, expected, atol=1e-6, rtol=1e-6)


def test_inverse_and_logabsdet_dot_general_non_square_returns_nan():
    matrix = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def f(x):
        return x @ matrix

    x = jnp.array([0.2, -0.3, 1.4])
    eqn = jax.make_jaxpr(f)(x).jaxpr.eqns[0]
    rule = REGISTRY.get(jax.lax.dot_general_p, Context.INVERSE_LOGDET)
    result = rule(eqn, [None, matrix], [f(x)], context=None)
    assert jnp.all(jnp.isnan(result.resolved_vals[0]))
    assert jnp.isnan(result.state[eqn.invars[0]])


def test_logabsdet_dot_general_then_exp_does_not_double_count():
    matrix = jnp.array(
        [[1.8, 0.2, -0.1], [0.0, 1.3, 0.4], [0.0, 0.0, 0.9]]
    )

    def f(x):
        return jnp.exp(x @ matrix)

    x = jnp.array([0.2, -0.7, 1.3])
    recovered, logdet = inverse_and_logabsdet(f)(f(x))
    _, matrix_logdet = jnp.linalg.slogdet(matrix)
    expected = -(matrix_logdet + jnp.sum(x @ matrix))
    jacobian = jax.jacobian(f)(x)
    expected_from_jacobian = -jnp.log(jnp.abs(jnp.linalg.det(jacobian)))

    assert jnp.allclose(recovered, x, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(logdet, expected, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(logdet, expected_from_jacobian, atol=1e-5, rtol=1e-5)
