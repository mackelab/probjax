import jax.numpy as jnp
import pytest

from probjax.utils.odeint import AdaptiveParams, _odeint
from probjax.utils.odeutil import TraceNothing

pytest_plugins = ["test_problems.ode_problems"]

KNOWN_ERROR = ["bogacki_shampine"]


ts_dense = jnp.linspace(0, 1, 100)


def test_odeint_basic_linear_ode(linear_ode_problem, ode_method):
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")
    x0, drift, f_true = linear_ode_problem
    adaptive_params = AdaptiveParams(atol=1e-2, rtol=1e-2)
    f_approx = _odeint(
        drift,
        x0,
        ts_dense,
        method=ode_method,
        adaptive_params=adaptive_params,
        collect_trace=True,
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
        drift,
        x0,
        ts_dense,
        method=ode_method,
        adaptive_params=adaptive_params,
        collect_trace=True,
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

    trace = _odeint(drift, x0, ts, method=ode_method, collect_trace=True)
    final_state = _odeint(drift, x0, ts, method=ode_method, collect_trace=False)

    # Test that pytree is preserved
    assert isinstance(trace, dict)
    assert trace["x"].shape == (100, 1)
    assert trace["y"].shape == (100, 1)
    assert isinstance(final_state, dict)
    assert final_state["x"].shape == (1,)
    assert final_state["y"].shape == (1,)


def test_odeint_with_pytree_filter_state(ode_method):
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    x0 = {"x": jnp.ones(1) * 10.0, "y": jnp.ones(1) * 5.0}
    ts = jnp.linspace(0, 1, 100)

    def filter_state(state):
        return state["x"]

    def drift(t, x):
        return {"x": x["x"] * x["y"], "y": x["y"] * x["x"]}

    trace = _odeint(
        drift,
        x0,
        ts,
        method=ode_method,
        filter_state=filter_state,
        collect_trace=True,
    )
    final_filtered = _odeint(
        drift,
        x0,
        ts,
        method=ode_method,
        filter_state=filter_state,
        collect_trace=False,
    )

    # Test that pytree is preserved
    assert trace.shape == (100, 1)
    assert final_filtered.shape == (1,)


def test_odeint_trace_nothing(ode_method):
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    x0 = jnp.ones(2)
    ts = jnp.linspace(0, 0.5, 10)

    def drift(t, x):
        del t
        return -x

    traced = _odeint(
        drift,
        x0,
        ts,
        method=ode_method,
        filter_state=TraceNothing(),
        collect_trace=True,
    )

    assert traced is None
    final_none = _odeint(
        drift,
        x0,
        ts,
        method=ode_method,
        filter_state=TraceNothing(),
        collect_trace=False,
    )
    assert final_none is None
