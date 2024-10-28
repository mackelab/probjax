import jax.numpy as jnp
import pytest
from probjax.utils.odeint import _odeint

pytest_plugins = ["test_problems.ode_problems"]

KNOWN_ERROR = ["bogacki_shampine"]


ts_dense = jnp.linspace(0, 1, 100)


def test_odeint_basic_linear_ode(linear_ode_problem, ode_method):
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")
    x0, drift, f_true = linear_ode_problem
    f_approx = _odeint(drift, x0, ts_dense, method=ode_method, atol=1e-2, rtol=1e-2)
    f_true = f_true(ts_dense, x0)
    error = jnp.mean((f_approx - f_true) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_odeint_nonlienar_ode(nonlinear_ode_problem, ode_method):
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")
    x0, drift, f_true = nonlinear_ode_problem
    f_approx = _odeint(drift, x0, ts_dense, method=ode_method, atol=1e-2, rtol=1e-2)
    f_true = f_true(ts_dense, x0)
    error = jnp.mean((f_approx - f_true) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"
