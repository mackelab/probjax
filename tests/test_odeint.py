import jax.numpy as jnp
import pytest

from probjax.utils.odeint import AdaptiveParams, _odeint

pytest_plugins = ["test_problems.ode_problems"]

KNOWN_ERROR = ["bogacki_shampine"]


ts_dense = jnp.linspace(0, 1, 100)


def test_odeint_basic_linear_ode(linear_ode_problem, ode_method):
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")
    x0, drift, f_true = linear_ode_problem
    adaptive_params = AdaptiveParams(atol=1e-2, rtol=1e-2)
    f_approx = _odeint(
        drift, x0, ts_dense, method=ode_method, adaptive_params=adaptive_params
    )
    f_true = f_true(ts_dense, x0)
    error = jnp.mean((f_approx - f_true) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_odeint_nonlienar_ode(nonlinear_ode_problem, ode_method):
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")
    x0, drift, f_true = nonlinear_ode_problem
    adaptive_params = AdaptiveParams(atol=1e-2, rtol=1e-2)
    f_approx = _odeint(
        drift, x0, ts_dense, method=ode_method, adaptive_params=adaptive_params
    )
    f_true = f_true(ts_dense, x0)
    error = jnp.mean((f_approx - f_true) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_odeint_with_pytree(ode_method):
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    x0 = {"x": jnp.ones(1) * 10.0, "y": jnp.ones(1) * 5.0}
    ts = jnp.linspace(0, 1, 100)

    def drift(t, x):
        return {"x": x["x"] * x["y"], "y": x["y"] * x["x"]}

    f_approx = _odeint(drift, x0, ts, method=ode_method)

    # Test that pytree is preserved
    assert isinstance(f_approx, dict)
    assert f_approx["x"].shape == (100, 1)
    assert f_approx["y"].shape == (100, 1)


def test_odeint_with_pytree_filter_state(ode_method):
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    x0 = {"x": jnp.ones(1) * 10.0, "y": jnp.ones(1) * 5.0}
    ts = jnp.linspace(0, 1, 100)

    def filter_state(state):
        return state["x"]

    def drift(t, x):
        return {"x": x["x"] * x["y"], "y": x["y"] * x["x"]}

    f_approx = _odeint(drift, x0, ts, method=ode_method, filter_state=filter_state)

    # Test that pytree is preserved
    assert f_approx.shape == (100, 1)
