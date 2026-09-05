import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.utils.functions import linear_drift, split_drift

A1 = jnp.array([[0.0, 1.0], [-1.0, 0.0]])  # Peridoic
A2 = jnp.array([[0.0, 1.0], [-1.0, -1.0]])  # Stable
A3 = jnp.array([[0.0, 1.0], [-1.0, 1.0]])  # Unstable
A4 = jnp.array(np.random.normal(0, 1, (5, 5)) * 0.1)  # Random


@pytest.fixture(
    params=[A1, A2, A3, A4],
    ids=["linear_periodic_2d", "linear_stable_2d", "linear_unstable_2d", "random_5d"],
)
def linear_ode_problem(request):
    """Linear ODE problem with linear_drift wrapper for specialized solvers."""
    A = request.param
    x0 = jnp.ones((A.shape[0],))

    # Use linear_drift wrapper for linear_exact and exponential methods
    drift = linear_drift(A=A)

    def true_f(t, x0):
        E = A.reshape((1,) + A.shape)
        t = t.reshape(t.shape + (1,) * len(A.shape))
        Phi = jax.scipy.linalg.expm(E * t)
        return jnp.dot(Phi, x0)

    return x0, drift, true_f


@pytest.fixture(
    params=[A1, A2, A3, A4],
    ids=["linear_periodic_2d", "linear_stable_2d", "linear_unstable_2d", "random_5d"],
)
def split_drift_ode_problem(request):
    """Linear ODE problem with split_drift wrapper for exponential integrators.

    Formulates linear drift dy/dt = A*y as split_drift with:
    - lin_coeff(t) returning eigenvalue (approximated as scalar)
    - nonlin(t, y) handling the full dynamics
    """
    A = request.param
    x0 = jnp.ones((A.shape[0],))

    # For exponential methods, formulate as split drift
    # Use zero linear coefficient and put all dynamics in nonlinear part
    def lin_coeff(t):
        return jnp.array(0.0)

    def nonlin(t, y):
        return A @ y

    drift = split_drift(lin_coeff=lin_coeff, nonlin=nonlin)

    def true_f(t, x0):
        E = A.reshape((1,) + A.shape)
        t = t.reshape(t.shape + (1,) * len(A.shape))
        Phi = jax.scipy.linalg.expm(E * t)
        return jnp.dot(Phi, x0)

    return x0, drift, true_f


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
