import jax
import jax.numpy as jnp
import pytest
from typing import Any, cast

from probjax.utils.functions import const_diffusion, linear_drift, split_drift
from probjax.utils.odeint import odeint
from probjax.utils.odeutil import AdaptiveParams
from probjax.utils.odeutil import TraceNothing
from probjax.utils.sdeint import sdeint

pytest_plugins = ["test_problems.ode_problems", "test_problems.sde_problems"]

KNOWN_ERROR = []  # All methods should work now
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
    f_approx = odeint(
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
    f_approx = odeint(
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
    f_approx = odeint(
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

    trace = odeint(drift, x0, ts, method=ode_method, collect_trace=True)
    final_state = odeint(drift, x0, ts, method=ode_method, collect_trace=False)

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

    trace = odeint(
        drift,
        x0,
        ts,
        method=ode_method,
        filter_state=filter_state,
        collect_trace=True,
    )
    final_filtered = odeint(
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

    traced = odeint(
        drift,
        x0,
        ts,
        method=ode_method,
        filter_state=TraceNothing(),
        collect_trace=True,
    )

    assert traced is None
    final_none = odeint(
        drift,
        x0,
        ts,
        method=ode_method,
        filter_state=TraceNothing(),
        collect_trace=False,
    )
    assert final_none is None


def test_odeint_supports_drift_args(ode_method):
    """Test ODE solvers forward positional ``*args`` to plain-callable drifts.

    The public API no longer accepts drift keyword arguments — users pass
    parameters positionally via ``*args`` or bind them with
    ``functools.partial`` / ``drift.bind_args(...)``.
    """
    if ode_method in KNOWN_ERROR:
        pytest.xfail(f"{ode_method} method has known error")

    # Specialized methods require specific drift types
    if ode_method in SPLIT_DRIFT_METHODS + ["linear_exact"]:
        return

    x0 = jnp.array([1.0, -2.0])
    ts = jnp.linspace(0.0, 1.0, 50)
    rate = jnp.array(-0.3)
    bias = jnp.array(0.15)

    def drift(t, x, rate, bias):
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

    # Equivalent via functools.partial (no kwargs on odeint itself).
    from functools import partial

    drift_partial = partial(drift, rate=rate, bias=bias)

    trace_partial = odeint(
        drift_partial,
        x0,
        ts,
        method=ode_method,
        collect_trace=True,
    )

    assert trace_positional is not None
    assert trace_partial is not None
    assert jnp.allclose(trace_partial, trace_positional, atol=1e-6, rtol=1e-6)

    jitted_terminal = jax.jit(
        lambda y, r, b: odeint(
            drift,
            y,
            ts,
            r,
            b,
            method=ode_method,
            collect_trace=False,
        )
    )
    terminal_ref = odeint(
        drift,
        x0,
        ts,
        rate,
        bias,
        method=ode_method,
        collect_trace=False,
    )
    terminal_jit = jitted_terminal(x0, rate, bias)
    assert terminal_ref is not None
    assert terminal_jit is not None
    assert jnp.allclose(terminal_jit, terminal_ref, atol=1e-6, rtol=1e-6)


def test_odeint_split_drift_supports_args(ode_method):
    if ode_method not in SPLIT_DRIFT_METHODS:
        return

    x0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 1.0, 40)
    scale = jnp.array(-0.1)
    bias = jnp.array(0.05)

    def lin_coeff(t):
        del t
        return jnp.array(-0.2)

    def nonlin(t, x, scale, bias):
        del t
        return scale * x + bias

    drift = split_drift(lin_coeff=lin_coeff, nonlin=nonlin)

    trace_positional = odeint(
        drift,
        x0,
        ts,
        scale,
        bias,
        method=ode_method,
        collect_trace=True,
    )

    # ``bind_args`` threads positional args into the nonlinear part,
    # preserving the ``split_drift`` marker type.
    drift_bound = drift.bind_args(scale, bias)
    trace_bound = odeint(
        drift_bound,
        x0,
        ts,
        method=ode_method,
        collect_trace=True,
    )

    assert trace_positional is not None
    assert trace_bound is not None
    assert jnp.allclose(trace_bound, trace_positional, atol=1e-6, rtol=1e-6)


def test_linear_exact_supports_linear_drift_bind_args():
    """``linear_drift.b`` extra positional args flow through ``bind_args``."""
    A = jnp.array(-0.7)
    x0 = jnp.array([1.25])
    ts = jnp.linspace(0.0, 1.0, 30)
    bias = 0.15

    def b(t, bias):
        del t
        return jnp.asarray([bias])

    drift = linear_drift(A=A, b=b).bind_args(bias)

    result = odeint(drift, x0, ts, method="linear_exact")

    def b_bound(t):
        return b(t, bias)

    expected = odeint(linear_drift(A=A, b=b_bound), x0, ts, method="linear_exact")
    assert result is not None
    assert expected is not None
    assert jnp.allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_linear_exact_scalar():
    A = jnp.array(-0.5)
    x0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    drift = linear_drift(A=A)
    result = odeint(drift, x0, ts, method="linear_exact")
    expected = (x0 * jnp.exp(A * ts)).reshape(-1, 1)  # Shape (20, 1) to match result
    assert jnp.allclose(result, expected, atol=1e-6)


def test_linear_exact_dense():
    A = jnp.array([[0.0, 1.0], [-2.0, -1.0]])
    x0 = jnp.array([1.0, 0.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    drift = linear_drift(A=A)
    result = odeint(drift, x0, ts, method="linear_exact")

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
    result = odeint(drift, x0, ts, method="linear_exact")

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
    result_exact = odeint(drift, x0, ts, method="linear_exact")
    result_rk4 = odeint(drift, x0, ts, method="rk4")

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


# Scalar sde with ground truth solution

t = jnp.linspace(0.0, 10.0, 200)

# Methods that require split_drift
SPLIT_DRIFT_SDE_METHODS = ["exp_euler_maruyama"]


def _sde_trace(*args: Any, **kwargs: Any) -> jax.Array:
    result = sdeint(*args, **kwargs)
    if isinstance(result, tuple):
        trace = result[0]
        assert trace is not None
        return cast(jax.Array, trace)
    assert result is not None
    return cast(jax.Array, result)


def _sde_trace_and_brownian(*args: Any, **kwargs: Any) -> tuple[jax.Array, jax.Array]:
    result = sdeint(*args, **kwargs)
    assert isinstance(result, tuple)
    trace, brownian = result
    assert trace is not None
    assert brownian is not None
    return cast(jax.Array, trace), cast(jax.Array, brownian)


def test_sdeint_scalar(sde_method, scalar_sde_problem):
    """Test SDE solvers with scalar problem."""
    # Exponential methods need split_drift (see test_sdeint_split_drift)
    if sde_method in SPLIT_DRIFT_SDE_METHODS:
        return
    # linear_exact_sde needs linear_drift + const_diffusion (tested separately)
    if sde_method == "linear_exact_sde":
        return

    x0, f, g, f_true = scalar_sde_problem
    key = jax.random.PRNGKey(0)
    f_approx, dWt = _sde_trace_and_brownian(
        key, f, g, x0, t, method=sde_method, return_brownian=True
    )
    Wt = jnp.cumsum(dWt, axis=0)
    f_sol = f_true(Wt, t, x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_sdeint_split_drift(sde_method, split_drift_sde_problem):
    """Test exponential SDE methods that require split_drift."""
    if sde_method not in SPLIT_DRIFT_SDE_METHODS:
        return

    x0, f, g, f_true = split_drift_sde_problem
    key = jax.random.PRNGKey(0)
    f_approx, dWt = _sde_trace_and_brownian(
        key, f, g, x0, t, method=sde_method, return_brownian=True
    )
    Wt = jnp.cumsum(dWt, axis=0)
    f_sol = f_true(Wt, t, x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_sdeint_collect_trace_false(sde_method, scalar_sde_problem):
    """Test SDE solvers with collect_trace=False."""
    # Specialized methods require specific drift type wrappers
    if sde_method in SPLIT_DRIFT_SDE_METHODS + ["linear_exact_sde"]:
        return

    x0, f, g, _ = scalar_sde_problem
    key = jax.random.PRNGKey(1)
    final = _sde_trace(
        key,
        f,
        g,
        x0,
        t,
        method=sde_method,
        collect_trace=False,
        return_brownian=False,
    )
    assert final.shape == x0.shape


def test_sdeint_return_brownian_requires_trace(scalar_sde_problem):
    x0, f, g, _ = scalar_sde_problem
    key = jax.random.PRNGKey(2)
    with pytest.raises(ValueError):
        sdeint(
            key,
            f,
            g,
            x0,
            t,
            return_brownian=True,
            collect_trace=False,
        )


def test_sdeint_2d(sde_method, two_dimensional_sde_problem):
    """Test SDE solvers with 2D problem."""
    # Specialized methods require specific drift type wrappers
    if sde_method in SPLIT_DRIFT_SDE_METHODS + ["linear_exact_sde"]:
        return

    x0, f, g, f_true = two_dimensional_sde_problem
    key = jax.random.PRNGKey(0)
    f_approx, dWt = _sde_trace_and_brownian(
        key, f, g, x0, t, method=sde_method, return_brownian=True
    )
    Wt = jnp.cumsum(dWt, axis=0)
    f_sol = f_true(Wt, t, x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_sdeint_supports_args(sde_method, scalar_sde_problem):
    """SDE solvers forward ``*args`` to drift and diffusion.

    The public API no longer accepts per-function kwargs; parameters are
    passed positionally or bound via ``functools.partial``.
    """
    # Specialized methods require specific drift type wrappers
    if sde_method in SPLIT_DRIFT_SDE_METHODS + ["linear_exact_sde"]:
        return

    x0, base_drift, base_diffusion, _ = scalar_sde_problem
    key = jax.random.PRNGKey(3)

    scale = jnp.array(0.7)
    bias = jnp.array(0.02)

    def drift(t, x, scale, bias):
        return scale * base_drift(t, x) + bias

    def diffusion(t, x, scale, bias):
        del bias
        return scale * base_diffusion(t, x)

    positional = _sde_trace(
        key, drift, diffusion, x0, t, scale, bias, method=sde_method
    )

    from functools import partial

    drift_bound = partial(drift, scale=scale, bias=bias)
    diffusion_bound = partial(diffusion, scale=scale, bias=bias)
    bound = _sde_trace(
        key,
        drift_bound,
        diffusion_bound,
        x0,
        t,
        method=sde_method,
    )

    assert jnp.allclose(positional, bound, atol=1e-6, rtol=1e-6)


def test_sdeint_split_drift_supports_args(sde_method):
    if sde_method not in SPLIT_DRIFT_SDE_METHODS:
        return

    key = jax.random.PRNGKey(7)
    x0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 0.5, 64)
    scale = jnp.array(0.6)
    bias = jnp.array(0.03)

    def lin_coeff(t):
        del t
        return jnp.array(-0.4)

    def nonlin(t, y, scale, bias):
        del t
        return scale * y + bias

    def diffusion(t, y, scale, bias):
        del t, bias
        return jnp.abs(scale) * jnp.ones_like(y)

    drift = split_drift(lin_coeff=lin_coeff, nonlin=nonlin)

    positional = _sde_trace(
        key, drift, diffusion, x0, ts, scale, bias, method=sde_method
    )

    drift_bound = drift.bind_args(scale, bias)
    from functools import partial

    diffusion_bound = partial(diffusion, scale=scale, bias=bias)
    bound = _sde_trace(
        key,
        drift_bound,
        diffusion_bound,
        x0,
        ts,
        method=sde_method,
    )

    assert jnp.allclose(positional, bound, atol=1e-6, rtol=1e-6)


def test_exp_euler_maruyama_weights_diffusion_with_linear_coeff():
    key = jax.random.PRNGKey(17)
    x0 = jnp.array([0.4])
    ts = jnp.array([0.0, 0.2])
    c = jnp.array(3.0)

    def lin_coeff(t):
        del t
        return c

    def nonlin(t, y):
        del t, y
        return jnp.zeros_like(x0)

    # Depend on state to avoid additive_diffusion marker path.
    def diffusion(t, y):
        del t
        return jnp.ones_like(y)

    drift = split_drift(lin_coeff=lin_coeff, nonlin=nonlin)
    _, brownian_trace = _sde_trace_and_brownian(
        key,
        drift,
        diffusion,
        x0,
        ts,
        method="exp_euler_maruyama",
        return_brownian=True,
    )

    dt = ts[1] - ts[0]
    weighted_var = jnp.abs(dt) * (
        (jnp.exp(2.0 * c * jnp.abs(dt)) - 1.0) / (2.0 * c * jnp.abs(dt))
    )
    expected_std = jnp.sqrt(weighted_var)
    step_key = jax.random.split(key, ts.shape[0] - 1)[0]
    expected_increment = jax.random.normal(step_key, (x0.shape[0],)) * expected_std
    brownian_step = jnp.asarray(brownian_trace)[1]

    assert jnp.allclose(brownian_step, expected_increment, atol=1e-6, rtol=1e-6)


def test_sdeint_rectangular_diffusion_is_supported(sde_method):
    """Test SDE solvers support rectangular diffusion matrices."""
    # Specialized methods require specific drift type wrappers
    if sde_method in SPLIT_DRIFT_SDE_METHODS + ["linear_exact_sde"]:
        return

    key = jax.random.PRNGKey(4)
    ts = jnp.linspace(0.0, 1.0, 64)
    x0 = jnp.array([0.2, -0.4])
    diffusion_matrix = jnp.array([
        [0.3, -0.2, 0.1],
        [0.1, 0.4, 0.25],
    ])

    def drift(t, x, scale):
        del t
        return -0.15 * scale * x

    def diffusion(t, x, scale):
        del t, x
        return scale * diffusion_matrix

    state_trace, brownian_trace = _sde_trace_and_brownian(
        key,
        drift,
        diffusion,
        x0,
        ts,
        0.8,
        method=sde_method,
        return_brownian=True,
    )

    assert state_trace.shape == (ts.shape[0], x0.shape[0])
    assert brownian_trace.shape == (ts.shape[0], diffusion_matrix.shape[1])


def test_linear_exact_sde_scalar():
    key = jax.random.PRNGKey(42)
    A = jnp.array(-2.0)
    G = jnp.array(0.5)
    x0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    drift = linear_drift(A=A)
    diffusion = const_diffusion(G=G)

    result = _sde_trace(key, drift, diffusion, x0, ts, method="linear_exact_sde")
    assert result.shape == (ts.shape[0], x0.shape[0])


def test_linear_exact_sde_dense():
    key = jax.random.PRNGKey(123)
    A = jnp.array([[-1.0, 0.5], [0.3, -0.8]])
    G = jnp.array([[0.2, 0.0], [0.0, 0.15]])
    x0 = jnp.array([1.0, 0.5])
    ts = jnp.linspace(0.0, 0.5, 10)

    drift = linear_drift(A=A)
    diffusion = const_diffusion(G=G)

    result = _sde_trace(key, drift, diffusion, x0, ts, method="linear_exact_sde")
    assert result.shape == (ts.shape[0], x0.shape[0])


def test_linear_exact_sde_mean_matches_analytical():
    key = jax.random.PRNGKey(0)
    A = jnp.array(-1.0)
    G = jnp.array(0.0)
    x0 = jnp.array([2.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    drift = linear_drift(A=A)
    diffusion = const_diffusion(G=G)

    result = _sde_trace(key, drift, diffusion, x0, ts, method="linear_exact_sde")
    expected_mean = x0 * jnp.exp(A * ts)
    empirical_mean = result[:, 0]
    assert jnp.allclose(empirical_mean, expected_mean, atol=0.1)


def test_linear_exact_sde_matches_euler_maruyama():
    key = jax.random.PRNGKey(99)
    A = jnp.array(-0.5)
    G = jnp.array(0.1)
    x0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 0.2, 5)

    drift = linear_drift(A=A)
    diffusion = const_diffusion(G=G)

    result_exact = _sde_trace(key, drift, diffusion, x0, ts, method="linear_exact_sde")

    key = jax.random.PRNGKey(99)
    result_em = _sde_trace(key, drift, diffusion, x0, ts, method="euler_maruyama")

    assert jnp.allclose(result_exact, result_em, atol=0.15)


def test_linear_exact_sde_requires_markers():
    key = jax.random.PRNGKey(0)
    A = jnp.array(-1.0)
    G = jnp.array(0.5)
    x0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 0.1, 5)

    def plain_drift(t, x):
        return A * x

    def plain_diffusion(t, x):
        return G

    with pytest.raises(TypeError, match="linear_exact_sde requires.*linear_drift"):
        sdeint(key, plain_drift, plain_diffusion, x0, ts, method="linear_exact_sde")


def test_linear_exact_sde_supports_linear_drift_bind_args():
    """``linear_drift.b``'s extra positional args flow through ``bind_args``."""
    key = jax.random.PRNGKey(11)
    A = jnp.array(-0.8)
    x0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 0.2, 8)
    offset = 0.12

    def b(t, offset):
        del t
        return jnp.asarray([offset])

    drift = linear_drift(A=A, b=b).bind_args(offset)
    diffusion = const_diffusion(G=jnp.array(0.2))

    result = _sde_trace(
        key,
        drift,
        diffusion,
        x0,
        ts,
        method="linear_exact_sde",
    )

    def b_bound(t):
        return b(t, offset)

    expected = _sde_trace(
        key,
        linear_drift(A=A, b=b_bound),
        diffusion,
        x0,
        ts,
        method="linear_exact_sde",
    )

    assert jnp.allclose(result, expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Pytree-native drift tests (post closure-lifting-removal refactor)
# ---------------------------------------------------------------------------


def test_odeint_with_equinox_neural_drift():
    """An ``eqx.Module`` with array fields should flow through as a pytree."""
    eqx = pytest.importorskip("equinox")

    class LinearDrift(eqx.Module):
        A: jax.Array
        b: jax.Array

        def __call__(self, t, y):
            del t
            return self.A @ y + self.b

    A = jnp.array([[-0.5, 0.1], [0.0, -0.3]])
    b = jnp.array([0.05, -0.05])
    drift = LinearDrift(A=A, b=b)

    x0 = jnp.array([1.0, 2.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    trace = odeint(drift, x0, ts, method="rk4", collect_trace=True)
    assert trace is not None
    assert trace.shape == (ts.shape[0], x0.shape[0])

    # Grad through an array parameter of the drift (the whole point of
    # pytree-native drifts: parameters participate in transformations).
    def loss(A_, b_, x0_):
        d = LinearDrift(A=A_, b=b_)
        out = odeint(d, x0_, ts, method="rk4", collect_trace=False)
        return jnp.sum(out ** 2)

    gA = jax.grad(loss, argnums=0)(A, b, x0)
    assert gA.shape == A.shape
    assert jnp.all(jnp.isfinite(gA))


def test_odeint_plain_closure_captures_are_constant_only():
    """Plain drifts may close over *constant* (non-traced) values.

    Traced values must be passed explicitly through ``*args``/``**kwargs`` —
    the closure-lifting hack has been removed.
    """
    rate = jnp.array(-0.4)  # concrete constant, not traced
    bias = jnp.array(0.05)

    def drift(t, y):
        del t
        return rate * y + bias

    x0 = jnp.array([1.0, -2.0])
    ts = jnp.linspace(0.0, 1.0, 25)

    trace = odeint(drift, x0, ts, method="rk4", collect_trace=True)
    assert trace is not None
    assert trace.shape == (ts.shape[0], x0.shape[0])


def test_odeint_with_partial_drift():
    """``functools.partial`` over constants should work as a drift."""
    from functools import partial as fpartial

    def drift(t, y, rate, bias):
        del t
        return rate * y + bias

    bound = fpartial(drift, rate=jnp.array(-0.2), bias=jnp.array(0.03))

    x0 = jnp.array([1.5, -0.25])
    ts = jnp.linspace(0.0, 1.0, 20)

    trace = odeint(bound, x0, ts, method="rk4", collect_trace=True)
    assert trace is not None
    assert trace.shape == (ts.shape[0], x0.shape[0])


def test_odeint_with_pytree_dataclass_drift():
    """User-registered pytree dataclass with array leaves flows through."""

    class LinDriftPyTree:
        def __init__(self, A, b):
            self.A = A
            self.b = b

        def __call__(self, t, y):
            del t
            return self.A @ y + self.b

    def _flatten(obj):
        return (obj.A, obj.b), None

    def _unflatten(_, children):
        A, b = children
        return LinDriftPyTree(A, b)

    jax.tree_util.register_pytree_node(LinDriftPyTree, _flatten, _unflatten)

    A = jnp.array([[-0.2, 0.0], [0.1, -0.4]])
    b = jnp.array([0.02, -0.02])
    drift = LinDriftPyTree(A, b)

    x0 = jnp.array([1.0, 1.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    trace = odeint(drift, x0, ts, method="rk4", collect_trace=True)
    assert trace is not None
    assert trace.shape == (ts.shape[0], x0.shape[0])


def test_odeint_traced_args_under_outer_jit():
    """Traced positional args must survive an outer ``jax.jit``."""

    def drift(t, y, rate, bias):
        del t
        return rate * y + bias

    x0 = jnp.array([1.0, -0.5])
    ts = jnp.linspace(0.0, 1.0, 30)

    @jax.jit
    def run(y0, rate, bias):
        return odeint(
            drift,
            y0,
            ts,
            rate,
            bias,
            method="rk4",
            collect_trace=False,
        )

    out_ref = odeint(
        drift,
        x0,
        ts,
        jnp.array(-0.3),
        jnp.array(0.1),
        method="rk4",
        collect_trace=False,
    )
    out_jit = run(x0, jnp.array(-0.3), jnp.array(0.1))

    assert out_ref is not None
    assert out_jit is not None
    assert jnp.allclose(out_jit, out_ref, atol=1e-6, rtol=1e-6)


def test_odeint_inverse_still_works_with_neural_drift():
    """Inverse through an ``eqx.Module`` drift should still reconstruct y0."""
    eqx = pytest.importorskip("equinox")
    from probjax.core.transformation import inverse

    class LinearDrift(eqx.Module):
        A: jax.Array

        def __call__(self, t, y):
            del t
            return self.A @ y

    A = jnp.array([[-0.4, 0.0], [0.0, -0.2]])
    drift = LinearDrift(A=A)

    x0 = jnp.array([1.0, 2.0])
    ts = jnp.linspace(0.0, 1.0, 200)

    def forward(y0):
        return odeint(drift, y0, ts, method="rk4", collect_trace=False)

    y = forward(x0)
    inv_forward = inverse(forward, invertible_arg=0)
    x_inv = inv_forward(y)

    assert jnp.allclose(x0, x_inv, atol=1e-3, rtol=1e-3)
