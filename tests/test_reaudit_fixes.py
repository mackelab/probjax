"""Regression coverage for the nine re-audit findings."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as sp
from probjax.utils.linalg import lanczos_logdet, batched_pcg_solve
from probjax.stats import genpareto, truncnorm
from probjax.nn.utils import call_with_optional_rng, filter_supported_kwargs


def test_logdet_gradient_scaled_identity():
    with jax.enable_x64():
        actual = jax.grad(lambda s: lanczos_logdet(s * jnp.eye(2), num_steps=2))(2.0)
        np.testing.assert_allclose(actual, 1.0, atol=1e-12)


def test_genpareto_explicit_shape_parameter_sampling():
    sample = genpareto.rvs(jax.random.key(0), c=0.5, shape=(5,))
    assert sample.shape == (5,)
    assert jnp.all(jnp.isfinite(sample))


def test_truncnorm_small_positive_quantile():
    with jax.enable_x64():
        actual = truncnorm.ppf(1e-100, a=-1.0, b=1.0)
        assert jnp.isfinite(actual)
        np.testing.assert_allclose(
            actual, sp.truncnorm.ppf(1e-100, -1.0, 1.0), atol=1e-14
        )


def test_truncnorm_quantile_second_derivative():
    with jax.enable_x64():
        q = 0.7
        actual = jax.grad(jax.grad(lambda p: truncnorm.ppf(p)))(q)
        reference = sp.norm.ppf(q) / sp.norm.pdf(sp.norm.ppf(q)) ** 2
        np.testing.assert_allclose(actual, reference, rtol=1e-8)


def test_truncnorm_one_sided_tail_variance():
    actual = truncnorm.var(a=jnp.float32(20.0), b=jnp.inf)
    np.testing.assert_allclose(actual, sp.truncnorm.var(20.0, np.inf), rtol=1e-3)


def test_pcg_small_representable_rhs():
    b = jnp.full((2, 1), 1e-20, dtype=jnp.float32)
    actual, info = batched_pcg_solve(lambda x: x, b, block_size=1)
    # Check in float64 on the host so the assertion itself cannot underflow.
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        rtol=1e-5,
        atol=0,
    )
    assert jnp.all(info.converged)


def test_optional_rng_plain_function():
    assert call_with_optional_rng(lambda x: x + 1, 1.0, rng=jax.random.key(0)) == 2.0


def test_forwarding_constructor_kwargs():
    class ForwardingLayer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    kwargs = {'kernel_sharding': ('data', None)}
    assert filter_supported_kwargs(ForwardingLayer, **kwargs) == kwargs


def test_genpareto_negative_shape_mode():
    # At c=-2, the density increases to a singularity at its upper endpoint.
    np.testing.assert_allclose(genpareto.mode(c=-2.0), 0.5)


import pytest
from functools import partial


@pytest.mark.parametrize('dimension', [1, 3, 20])
@pytest.mark.parametrize('operator', [False, True])
def test_logdet_repeated_spectrum_gradients(dimension, operator):
    from probjax.utils.linear_operator import LinearOperator

    with jax.enable_x64():

        def objective(scale):
            a = scale * jnp.eye(dimension)
            s = LinearOperator(lambda x: a @ x, dimension, dimension) if operator else a
            return lanczos_logdet(s, num_steps=dimension)

        np.testing.assert_allclose(
            jax.jit(jax.grad(objective))(2.0), dimension / 2, rtol=1e-10
        )


def test_logdet_dense_gradient_and_truncated_directional_derivative():
    with jax.enable_x64():
        factor = jax.random.normal(jax.random.key(10), (5, 5))
        a = factor @ factor.T + jnp.eye(5)
        np.testing.assert_allclose(
            jax.grad(lambda a: lanczos_logdet(a, num_steps=5))(a),
            jnp.linalg.inv(a),
            rtol=1e-9,
            atol=1e-9,
        )
        direction = jnp.diag(jnp.arange(1.0, 6.0))
        objective = lambda t: lanczos_logdet(
            a + t * direction, num_steps=2, num_probes=2, key=jax.random.key(2)
        )
        difference = (objective(1e-5) - objective(-1e-5)) / 2e-5
        np.testing.assert_allclose(jax.grad(objective)(0.0), difference, rtol=1e-7)


@pytest.mark.parametrize('q', [0.1, 0.7])
@pytest.mark.parametrize('bounds', [(-np.inf, np.inf), (-1.0, 2.0), (8.0, 9.0)])
def test_truncnorm_three_quantile_derivatives(q, bounds):
    with jax.enable_x64():
        a, b = bounds
        fn = lambda q: truncnorm.ppf(q, a=a, b=b)
        x = sp.truncnorm.ppf(q, a, b)
        dx = 1 / sp.truncnorm.pdf(x, a, b)
        references = [dx, x * dx**2, (1 + 2 * x * x) * dx**3]
        for reference in references:
            fn = jax.grad(fn)
            np.testing.assert_allclose(fn(q), reference, rtol=2e-8, atol=1e-9)


@pytest.mark.parametrize('method', ['ppf', 'isf'])
@pytest.mark.parametrize('bounds', [(-1.0, 1.0), (8.0, 9.0)])
def test_truncnorm_tiny_probabilities_are_in_support(method, bounds):
    with jax.enable_x64():
        a, b = bounds
        q = jnp.array([1e-100, 1e-30, 1e-15, 0.5])
        actual = jax.jit(lambda q: getattr(truncnorm, method)(q, a=a, b=b))(q)
        assert jnp.all((actual >= a) & (actual <= b))
        np.testing.assert_allclose(
            actual, getattr(sp.truncnorm, method)(np.asarray(q), a, b), atol=2e-14
        )


@pytest.mark.parametrize('bound', [5.0, 8.0, 20.0, 50.0])
@pytest.mark.parametrize('sign', [-1, 1])
def test_tail_variance_both_sides(bound, sign):
    a, b = (bound, np.inf) if sign > 0 else (-np.inf, -bound)
    np.testing.assert_allclose(
        truncnorm.var(a=a, b=b), sp.truncnorm.var(a, b), rtol=2e-4
    )
    assert np.isfinite(truncnorm.entropy(a=a, b=b))


def test_pcg_mixed_rhs_scales_and_initial_guess():
    a = jnp.array([[2.0, 0.2], [0.2, 1.0]])
    scales = jnp.array([1e-20, 1.0, 1e20, 0.0])
    b = jnp.array([[1.0], [2.0]]) * scales
    initial = 0.1 * b
    x, info = jax.jit(
        lambda b, x0: batched_pcg_solve(
            lambda x: a @ x, b, x0=x0, block_size=2, tol=1e-5
        )
    )(b, initial)
    expected = np.linalg.solve(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    np.testing.assert_allclose(
        np.asarray(x, dtype=np.float64), expected, rtol=2e-6, atol=0
    )
    assert jnp.all(info.converged)


def test_genpareto_sampling_variants():
    from probjax.core import joint_sample

    key = jax.random.key(12)
    c = jnp.array([0.0, 0.2, -2.0])
    distribution = genpareto(c=c, loc=jnp.array([1.0, 2.0, 3.0]), scale=2.0)
    samples = jax.jit(lambda k: distribution.rvs(k, shape=(4, 5)))(key)
    assert samples.shape == (4, 5, 3)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(samples[..., 2] <= 4.0)

    def model(k):
        return genpareto.rvs(k, c=0.5, loc=1.0, scale=2.0, shape=(5,), name='x')

    named = joint_sample(jax.jit(model))(key)['x']
    np.testing.assert_array_equal(named, model(key))


def test_optional_rng_distinct_functions_and_partials():
    key = jax.random.key(0)

    def plain(x):
        return x + 1

    def stochastic(x, *, rng):
        return x + jax.random.normal(rng)

    for _ in range(2):
        assert call_with_optional_rng(plain, 1.0, rng=key) == 2.0
        assert call_with_optional_rng(partial(plain), 1.0, rng=key) == 2.0
        np.testing.assert_array_equal(
            call_with_optional_rng(stochastic, 1.0, rng=key), stochastic(1.0, rng=key)
        )

    def positional_only(rng, /, x):
        return x

    from probjax.nn.utils import module_accepts_rng

    assert not module_accepts_rng(positional_only)


def test_custom_mlp_receives_forwarded_metadata(monkeypatch):
    from flax import nnx
    from probjax.nn.nets import simple

    seen = []

    class ForwardingLinear(nnx.Linear):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.pop('kernel_metadata', None))
            kwargs.pop('bias_metadata', None)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(simple, 'param_metadata', lambda *axes: {'axes': axes})
    model = simple.MLP([2, 4, 1], linear_cls=ForwardingLinear, rngs=nnx.Rngs(0))
    assert all(metadata is not None for metadata in seen)
    assert model(jnp.ones((3, 2))).shape == (3, 1)


@pytest.mark.parametrize('guess,converged', [(5e-5, True), (1e20, False)])
def test_pcg_zero_rhs_absolute_diagnostics(guess, converged):
    b = jnp.zeros((1, 1))
    _, info = batched_pcg_solve(
        lambda x: x, b, x0=jnp.full((1, 1), guess), block_size=1, maxiter=0, tol=1e-4
    )
    assert bool(info.converged[0]) == converged
    np.testing.assert_allclose(info.rel_residual[0], guess, rtol=1e-6)


def test_filter_matrix_free_solve_and_logdet_gradient():
    from probjax.inference.filtering.kalman_filter import default_logdet, default_solve
    from probjax.utils.linear_operator import LinearOperator

    a = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    operator = LinearOperator(lambda x: a @ x, 2, 2)
    rhs = jnp.array([[1e-20, 2e-20], [1e20, 2e20]])
    actual = jax.jit(lambda rhs: default_solve(operator, rhs, dense_mem_limit=0))(rhs)
    expected = np.linalg.solve(
        np.asarray(a, dtype=np.float64), np.asarray(rhs, dtype=np.float64).T
    ).T
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64), expected, rtol=1e-5, atol=0
    )
    derivative = jax.grad(
        lambda scale: default_logdet(
            LinearOperator(lambda x: scale * x, 2, 2), dense_mem_limit=0
        )
    )(2.0)
    np.testing.assert_allclose(derivative, 1.0, rtol=1e-6)
