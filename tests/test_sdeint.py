import jax
import jax.numpy as jnp
import pytest

from probjax.utils.sdeint import sdeint

pytest_plugins = ["test_problems.sde_problems"]

# Scalar sde with ground truth solution

t = jnp.linspace(0.0, 10.0, 200)


def test_sdeint_scalar(sde_method, scalar_sde_problem):
    x0, f, g, f_true = scalar_sde_problem
    key = jax.random.PRNGKey(0)
    f_approx, dWt = sdeint(key, f, g, x0, t, method=sde_method, return_brownian=True)
    Wt = jnp.cumsum(dWt, axis=0)
    f_sol = f_true(Wt, t, x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_sdeint_collect_trace_false(sde_method, scalar_sde_problem):
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
    x0, f, g, f_true = two_dimensional_sde_problem
    key = jax.random.PRNGKey(0)
    f_approx, dWt = sdeint(key, f, g, x0, t, method=sde_method, return_brownian=True)
    Wt = jnp.cumsum(dWt, axis=0)
    f_sol = f_true(Wt, t, x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


def test_sdeint_supports_kwargs(sde_method, scalar_sde_problem):
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
