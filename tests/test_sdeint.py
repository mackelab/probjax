import jax
import jax.numpy as jnp
import pytest

from probjax.utils.functions import const_diffusion, linear_drift, split_drift
from probjax.utils.sdeint import sdeint

pytest_plugins = ["test_problems.sde_problems"]

# Scalar sde with ground truth solution

t = jnp.linspace(0.0, 10.0, 200)

# Methods that require split_drift
SPLIT_DRIFT_SDE_METHODS = ["exp_euler_maruyama"]


def test_sdeint_scalar(sde_method, scalar_sde_problem):
    """Test SDE solvers with scalar problem."""
    # Skip exponential methods - they need split_drift
    if sde_method in SPLIT_DRIFT_SDE_METHODS:
        pytest.skip(f"{sde_method} requires split_drift (see test_sdeint_split_drift)")
    # Skip linear_exact_sde - it needs linear_drift + const_diffusion
    if sde_method == "linear_exact_sde":
        pytest.skip(f"{sde_method} requires linear_drift + const_diffusion")

    x0, f, g, f_true = scalar_sde_problem
    key = jax.random.PRNGKey(0)
    f_approx, dWt = sdeint(key, f, g, x0, t, method=sde_method, return_brownian=True)
    Wt = jnp.cumsum(dWt, axis=0)
    f_sol = f_true(Wt, t, x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_sdeint_split_drift(sde_method, split_drift_sde_problem):
    """Test exponential SDE methods that require split_drift."""
    if sde_method not in SPLIT_DRIFT_SDE_METHODS:
        pytest.skip(f"{sde_method} doesn't require split_drift")

    x0, f, g, f_true = split_drift_sde_problem
    key = jax.random.PRNGKey(0)
    f_approx, dWt = sdeint(key, f, g, x0, t, method=sde_method, return_brownian=True)
    Wt = jnp.cumsum(dWt, axis=0)
    f_sol = f_true(Wt, t, x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_sdeint_collect_trace_false(sde_method, scalar_sde_problem):
    """Test SDE solvers with collect_trace=False."""
    # Skip specialized methods
    if sde_method in SPLIT_DRIFT_SDE_METHODS + ["linear_exact_sde"]:
        pytest.skip(f"{sde_method} requires specific drift type wrappers")

    x0, f, g, _ = scalar_sde_problem
    key = jax.random.PRNGKey(1)
    final = sdeint(
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
    # Skip specialized methods
    if sde_method in SPLIT_DRIFT_SDE_METHODS + ["linear_exact_sde"]:
        pytest.skip(f"{sde_method} requires specific drift type wrappers")

    x0, f, g, f_true = two_dimensional_sde_problem
    key = jax.random.PRNGKey(0)
    f_approx, dWt = sdeint(key, f, g, x0, t, method=sde_method, return_brownian=True)
    Wt = jnp.cumsum(dWt, axis=0)
    f_sol = f_true(Wt, t, x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_sdeint_supports_kwargs(sde_method, scalar_sde_problem):
    """Test SDE solvers support kwargs."""
    # Skip specialized methods
    if sde_method in SPLIT_DRIFT_SDE_METHODS + ["linear_exact_sde"]:
        pytest.skip(f"{sde_method} requires specific drift type wrappers")

    x0, base_drift, base_diffusion, _ = scalar_sde_problem
    key = jax.random.PRNGKey(3)

    scale = jnp.array(0.7)
    bias = jnp.array(0.02)

    def drift(t, x, scale, bias=0.0):
        return scale * base_drift(t, x) + bias

    def diffusion(t, x, scale, bias=0.0):
        del bias
        return scale * base_diffusion(t, x)

    positional = sdeint(key, drift, diffusion, x0, t, scale, bias, method=sde_method)
    keyword = sdeint(
        key,
        drift,
        diffusion,
        x0,
        t,
        scale=scale,
        bias=bias,
        method=sde_method,
    )

    assert jnp.allclose(positional, keyword, atol=1e-6, rtol=1e-6)


def test_sdeint_rectangular_diffusion_is_supported(sde_method):
    """Test SDE solvers support rectangular diffusion matrices."""
    # Skip specialized methods
    if sde_method in SPLIT_DRIFT_SDE_METHODS + ["linear_exact_sde"]:
        pytest.skip(f"{sde_method} requires specific drift type wrappers")

    key = jax.random.PRNGKey(4)
    ts = jnp.linspace(0.0, 1.0, 64)
    x0 = jnp.array([0.2, -0.4])
    diffusion_matrix = jnp.array([
        [0.3, -0.2, 0.1],
        [0.1, 0.4, 0.25],
    ])

    def drift(t, x, scale=1.0):
        del t
        return -0.15 * scale * x

    def diffusion(t, x, scale=1.0):
        del t, x
        return scale * diffusion_matrix

    state_trace, brownian_trace = sdeint(
        key,
        drift,
        diffusion,
        x0,
        ts,
        scale=0.8,
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

    result = sdeint(key, drift, diffusion, x0, ts, method="linear_exact_sde")
    assert result.shape == (ts.shape[0], x0.shape[0])


def test_linear_exact_sde_dense():
    key = jax.random.PRNGKey(123)
    A = jnp.array([[-1.0, 0.5], [0.3, -0.8]])
    G = jnp.array([[0.2, 0.0], [0.0, 0.15]])
    x0 = jnp.array([1.0, 0.5])
    ts = jnp.linspace(0.0, 0.5, 10)

    drift = linear_drift(A=A)
    diffusion = const_diffusion(G=G)

    result = sdeint(key, drift, diffusion, x0, ts, method="linear_exact_sde")
    assert result.shape == (ts.shape[0], x0.shape[0])


def test_linear_exact_sde_mean_matches_analytical():
    key = jax.random.PRNGKey(0)
    A = jnp.array(-1.0)
    G = jnp.array(0.0)
    x0 = jnp.array([2.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    drift = linear_drift(A=A)
    diffusion = const_diffusion(G=G)

    result = sdeint(key, drift, diffusion, x0, ts, method="linear_exact_sde")
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

    result_exact = sdeint(key, drift, diffusion, x0, ts, method="linear_exact_sde")

    key = jax.random.PRNGKey(99)
    result_em = sdeint(key, drift, diffusion, x0, ts, method="euler_maruyama")

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
