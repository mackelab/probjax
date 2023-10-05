import pytest

import jax
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)

# Invertile function testcase fixtures


def for_loop_sum(x):
    x0 = x
    for i in range(10):
        x0 += 2
    return x0


def for_loop_mul(x):
    x0 = x
    for i in range(10):
        x0 *= 2
    return x0


# jnp.where does not yet work!
INVERTIBLE_FUNCTIONS = [
    jnp.log,
    jnp.log2,
    jnp.log10,
    jnp.log1p,
    # lambda x: jnp.logaddexp(x,1.), # TODO jnp.where does not yet work!
    # lambda x: jnp.logaddexp2(x,1.), # TODO jnp.where does not yet work!
    jnp.exp,
    jnp.exp2,
    # jnp.flip, # TODO ERROR
    for_loop_sum,
    for_loop_mul,
    lambda x: x + 1,
    lambda x: x - 1,
    lambda x: x**3,
    lambda x: x * 2,
    lambda x: x / 2,
]


@pytest.fixture(params=INVERTIBLE_FUNCTIONS)
def invertible_function(request):
    return request.param


# SDE problems fixtures
from probjax.utils.odeint import get_methods

METHODS = get_methods()

@pytest.fixture(params=METHODS, ids=METHODS)
def sde_method(request):
    return request.param









# ODE problems fixtures
A1 = jnp.array([[0.0, 1.0], [-1.0, 0.0]])  # Peridoic
A2 = jnp.array([[0.0, 1.0], [-1.0, -1.0]])  # Stable
A3 = jnp.array([[0.0, 1.0], [-1.0, 1.0]])  # Unstable
A4 = jax.random.normal(key, (5, 5)) * 0.1  # Random


@pytest.fixture(params=METHODS, ids=METHODS)
def ode_method(request):
    return request.param


@pytest.fixture
def ode_methods():
    return METHODS


@pytest.fixture(
    params=[A1, A2, A3, A4],
    ids=["linear_periodic_2d", "linear_stable_2d", "linear_unstable_2d", "random_5d"],
)
def linear_ode_problem(request):
    A = request.param
    x0 = jnp.ones((A.shape[0],))

    def f(t, x):
        return jnp.dot(A, x)

    def true_f(t, x0):
        E = A.reshape((1,) + A.shape)
        t = t.reshape(t.shape + (1,) * len(A.shape))
        Phi = jax.scipy.linalg.expm(E * t)
        return jnp.dot(Phi, x0)

    return x0, f, true_f
