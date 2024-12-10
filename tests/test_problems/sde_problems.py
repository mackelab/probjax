import jax.numpy as jnp
import pytest


@pytest.fixture
def scalar_sde_problem():
    x0 = jnp.array([0.5])

    def f(t, x):
        return -((1 / 10) ** 2) * jnp.sin(x) * jnp.cos(x) ** 3

    def g(t, x):
        return 1 / 10 * jnp.cos(x) ** 2

    def f_true(Wt, t, x0):
        return jnp.arctan(1 / 10 * Wt + jnp.tan(x0))

    return x0, f, g, f_true


@pytest.fixture
def two_dimensional_sde_problem():
    x0 = jnp.array([0.5, 0.5])

    def f2(t, x):
        return 0.5 * x * (1 - x) * (1 - 2 * x)

    def g2(t, x):
        return x * (1 - x)

    def f2_true(Wt, t, x0):
        return 1 / (1 + jnp.exp(-Wt + jnp.log(x0 / (1 - x0)).reshape(-1, 2)))

    return x0, f2, g2, f2_true
