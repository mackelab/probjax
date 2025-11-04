import jax
import jax.numpy as jnp
import pytest

from probjax.core.custom_primitives.random_variable import rv_p
from probjax.stats import beta, binomial, gamma, norm, poisson

# Each entry is (distribution, *args)
dist_params = [
    (norm, 0.0, 1.0),
    (gamma, 1.0, 1.0),
    (beta, 1.0, 1.0),
    (poisson, 1.0),
    (binomial, 10, 0.5),
]


@pytest.mark.parametrize("test_case", dist_params)
def test_bind(test_case):
    dist, *args = test_case
    key = jax.random.PRNGKey(0)
    value = rv_p.bind(key, *args, dist=dist)
    assert isinstance(value, jax.Array)
    assert value.shape == ()


@pytest.mark.parametrize("test_case", dist_params)
def test_jaxpr(test_case):
    dist, *args = test_case

    def f(key):
        x = rv_p.bind(key, *args, dist=dist)
        return x

    key = jax.random.PRNGKey(0)
    jaxpr = jax.make_jaxpr(f)(key)
    assert isinstance(jaxpr, jax.extend.core.ClosedJaxpr)


@pytest.mark.parametrize("test_case", dist_params)
def test_jit(test_case):
    dist, *args = test_case

    def f(key):
        x = rv_p.bind(key, *args, dist=dist)
        return x

    key = jax.random.PRNGKey(0)
    jitted_f = jax.jit(f)
    value = jitted_f(key)
    assert isinstance(value, jax.Array)
    assert value.shape == ()


@pytest.mark.parametrize("test_case", dist_params)
def test_vmap(test_case):
    dist, *args = test_case

    def f(key):
        x = rv_p.bind(key, *args, dist=dist)
        return x

    keys = jax.random.split(jax.random.PRNGKey(0), 10)
    vmapped_f = jax.vmap(f)
    values = vmapped_f(keys)
    assert isinstance(values, jax.Array)
    assert values.shape == (10,)


@pytest.mark.parametrize("test_case", dist_params)
def test_jit_vmap(test_case):
    dist, *args = test_case

    def f(key):
        x = rv_p.bind(key, *args, dist=dist)
        return x

    keys = jax.random.split(jax.random.PRNGKey(0), 10)
    jitted_vmapped_f = jax.jit(jax.vmap(f))
    values = jitted_vmapped_f(keys)
    assert isinstance(values, jax.Array)
    assert values.shape == (10,)


# Only test grad for continuous distributions
grad_dist_params = [
    (norm, 0.0, 1.0),
    (gamma, 1.0, 1.0),
    (beta, 1.0, 1.0),
]


@pytest.mark.parametrize("test_case", grad_dist_params)
def test_grad(test_case):
    dist, *args = test_case

    def g(loc):
        # Assuming the first argument is the one we can differentiate with respect to
        new_args = (loc,) + tuple(args[1:])
        x = rv_p.bind(jax.random.PRNGKey(0), *new_args, dist=dist)
        return x

    grad_g = jax.grad(g)
    value = grad_g(args[0])
    assert isinstance(value, jax.Array)
    assert value.shape == ()
    assert jnp.issubdtype(value.dtype, jnp.floating)


@pytest.mark.parametrize("test_case", dist_params)
def test_scan(test_case):
    dist, *args = test_case

    def g(loc):
        x = rv_p.bind(jax.random.PRNGKey(0), loc, *args[1:], dist=dist)
        return x

    def body(x, y):
        # We need to handle cases where the distribution takes more than one parameter
        # For scan, we are varying the first parameter.
        if len(args) > 1:
            test_args = (y,) + tuple(args[1:])
        else:
            test_args = (y,)

        return ((), rv_p.bind(jax.random.PRNGKey(0), *test_args, dist=dist))

    # Create appropriate inputs for scan
    if dist in [poisson, binomial]:
        scan_inputs = jnp.arange(1, 11)
    else:
        scan_inputs = jnp.linspace(0.1, 1.0, 10)

    _, outputs = jax.lax.scan(body, (), scan_inputs)
    assert isinstance(outputs, jax.Array)
    assert outputs.shape == (10,)
