from probjax.utils.odeint import get_method_info, get_methods
from probjax.utils.odeint import _odeint as odeint

import pytest

import jax
import jax.numpy as jnp

METHODS = get_methods()

A = jnp.array([[0.0, 1.0], [-1.0, 0.0]])


# Define the ODE
def f(t, x):
    return jnp.dot(A, x)


def true_f(t):
    Phi = jax.scipy.linalg.expm(A * t)
    return Phi @ x0


# Define the initial condition
x0 = jnp.ones((2,))
ts = jnp.linspace(0.0, 10.0, 100)
f_ts = jax.vmap(true_f)(ts)

ts_dense = jnp.linspace(0.0, 1.0, 1000)
f_ts_dense = jax.vmap(true_f)(ts_dense)

@pytest.mark.parametrize("method", METHODS)
def test_odeint(method):
    f_approx = odeint(f, x0, ts_dense, method=method)
    error = jnp.mean((f_approx - f_ts_dense) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"
