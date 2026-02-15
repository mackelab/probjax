import jax
import jax.numpy as jnp
import pytest

from probjax.utils.functions import linear_drift, split_drift
from probjax.utils.odeint import AdaptiveParams, _odeint, odeint
from probjax.utils.odeutil import TraceNothing

pytest_plugins = ["test_problems.ode_problems"]

KNOWN_ERROR = ["bogacki_shampine"]
# Methods that require split_drift
SPLIT_DRIFT_METHODS = ["exp_ab2_scalarL", "exp_ab3_scalarL"]


ts_dense = jnp.linspace(0, 1, 100)


def test_odeint_basic_linear_ode(linear_ode_problem, ode_method):
    """Test linear ODE solvers with linear_drift wrapper."""
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    # Exponential split methods need split_drift - tested separately
    if ode_method in SPLIT_DRIFT_METHODS:
        return

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


def test_odeint_split_drift_ode(split_drift_ode_problem, ode_method):
    """Test exponential methods that require split_drift."""
    if ode_method not in SPLIT_DRIFT_METHODS:
        return

    x0, drift, f_true = split_drift_ode_problem
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
    """Test nonlinear ODE solvers with plain functions."""
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    # Specialized methods require specific drift types - tested separately
    if ode_method in SPLIT_DRIFT_METHODS + ["linear_exact"]:
        return

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
    """Test ODE solvers with PyTree states."""
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    # Specialized methods require specific drift types
    if ode_method in SPLIT_DRIFT_METHODS + ["linear_exact"]:
        return

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
    """Test ODE solvers with PyTree states and filtering."""
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    # Specialized methods require specific drift types
    if ode_method in SPLIT_DRIFT_METHODS + ["linear_exact"]:
        return

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
    """Test ODE solvers with TraceNothing filter."""
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    # Specialized methods require specific drift types
    if ode_method in SPLIT_DRIFT_METHODS + ["linear_exact"]:
        return

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


def test_odeint_supports_drift_kwargs(ode_method):
    """Test ODE solvers support drift function kwargs."""
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    # Specialized methods require specific drift types
    if ode_method in SPLIT_DRIFT_METHODS + ["linear_exact"]:
        return

    x0 = jnp.array([1.0, -2.0])
    ts = jnp.linspace(0.0, 1.0, 50)
    rate = jnp.array(-0.3)
    bias = jnp.array(0.15)

    def drift(t, x, rate, bias=0.0):
        del t
        return rate * x + bias

    trace_positional = odeint(
        drift,
        x0,
        ts,
        rate,
        bias,
        method=ode_method,
        collect_trace=True,
    )
    trace_keyword = odeint(
        drift,
        x0,
        ts,
        rate,
        bias=bias,
        method=ode_method,
        collect_trace=True,
    )
    trace_keyword_only = odeint(
        drift,
        x0,
        ts,
        rate=rate,
        bias=bias,
        method=ode_method,
        collect_trace=True,
    )

    assert trace_positional is not None
    assert trace_keyword is not None
    assert trace_keyword_only is not None

    assert jnp.allclose(trace_keyword, trace_positional, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(trace_keyword_only, trace_positional, atol=1e-6, rtol=1e-6)

    jitted_terminal = jax.jit(
        lambda y, r, b: odeint(
            drift,
            y,
            ts,
            rate=r,
            bias=b,
            method=ode_method,
            collect_trace=False,
        )
    )
    terminal_ref = odeint(
        drift,
        x0,
        ts,
        rate=rate,
        bias=bias,
        method=ode_method,
        collect_trace=False,
    )
    terminal_jit = jitted_terminal(x0, rate, bias)
    assert terminal_ref is not None
    assert terminal_jit is not None
    assert jnp.allclose(terminal_jit, terminal_ref, atol=1e-6, rtol=1e-6)


def test_linear_exact_scalar():
    A = jnp.array(-0.5)
    x0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    drift = linear_drift(A=A)
    result = _odeint(drift, x0, ts, method="linear_exact")
    expected = (x0 * jnp.exp(A * ts)).reshape(-1, 1)  # Shape (20, 1) to match result
    assert jnp.allclose(result, expected, atol=1e-6)


def test_linear_exact_dense():
    A = jnp.array([[0.0, 1.0], [-2.0, -1.0]])
    x0 = jnp.array([1.0, 0.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    drift = linear_drift(A=A)
    result = _odeint(drift, x0, ts, method="linear_exact")

    def true_solution(t, x0):
        return jax.scipy.linalg.expm(A * t) @ x0

    expected = jnp.array([true_solution(t, x0) for t in ts])
    assert jnp.allclose(result, expected, atol=1e-5)


def test_linear_exact_with_bias():
    A = jnp.array(-1.0)

    def b(t):
        return jnp.array(0.5)

    x0 = jnp.array([2.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    drift = linear_drift(A=A, b=b)
    result = _odeint(drift, x0, ts, method="linear_exact")

    dt = ts[1] - ts[0]
    z = A * dt
    phi1 = (jnp.expm1(z) / z).reshape(())
    expected = (x0 * jnp.exp(A * ts) + (0.5 / A) * (jnp.exp(A * ts) - 1.0)).reshape(
        -1, 1
    )
    assert jnp.allclose(result, expected, atol=1e-5)


def test_linear_exact_matches_rk4():
    A = jnp.array([[0.0, 1.0], [-5.0, -2.0]])
    x0 = jnp.array([1.0, 0.5])
    ts = jnp.linspace(0.0, 0.5, 10)

    drift = linear_drift(A=A)
    result_exact = _odeint(drift, x0, ts, method="linear_exact")
    result_rk4 = _odeint(drift, x0, ts, method="rk4")

    # RK4 has discretization error, so use a looser tolerance
    assert jnp.allclose(result_exact, result_rk4, atol=0.05)


def test_linear_exact_requires_linear_drift():
    A = jnp.array(-0.5)
    x0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 1.0, 10)

    def bad_drift(t, x):
        return A * x

    with pytest.raises(TypeError, match="linear_exact requires"):
        odeint(bad_drift, x0, ts, method="linear_exact")
