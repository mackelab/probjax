import jax
import jax.numpy as jnp
import numpy as np
import pytest

A1 = jnp.array([[0.0, 1.0], [-1.0, 0.0]])  # Peridoic
A2 = jnp.array([[0.0, 1.0], [-1.0, -1.0]])  # Stable
A3 = jnp.array([[0.0, 1.0], [-1.0, 1.0]])  # Unstable
A4 = jnp.array(np.random.normal(0, 1, (5, 5)) * 0.1)  # Random


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


def logistic_problem():
    x0 = jnp.array([0.5])

    def f(t, x):
        return x * (1 - x)

    def true_f(t, x0):
        return 1.0 / (1.0 + jnp.exp(-t + jnp.log(x0 / (1.0 - x0))))

    return x0, f, true_f


@pytest.fixture(
    params=[logistic_problem],
    ids=["logistic_problem"],
)
def nonlinear_ode_problem(request):
    true_f = request.param
    x0, f, true_f = true_f()
    return x0, f, true_f
