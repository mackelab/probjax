import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from probjax.utils.functions import const_diffusion, linear_drift
from probjax.utils.linear_operator import phi_matrices
from probjax.utils.odeint import odeint
from probjax.utils.odeutil.solvers.exponential import (
    ExpODEState,
    build_exp_euler_step,
)
from probjax.utils.odeutil.solvers.multistep import build_am, build_bdf, init_lms
from probjax.utils.sdeutil.solver.linear_exact import (
    LinearExactSDEState,
    build_linear_exact_sde_step,
)
from probjax.utils.solver import newton_raphson, root


# Analytic root of x^2 - 2 is sqrt(2); started from 1 with a small iteration budget.
def test_root_iterates_to_machine_precision_with_small_budget():
    x, _ = root(lambda x: x * x - 2.0, jnp.array([1.0]), max_iter=4)
    assert jnp.allclose(x, np.sqrt(2.0), atol=1e-6)


def test_newton_raphson_returns_bool_convergence_flag():
    _, converged = newton_raphson(lambda x: x - 1.0, jnp.array([0.0]), max_iter=3)
    assert converged.dtype == bool
    assert bool(converged)


def test_newton_raphson_stays_converged_after_reaching_tol():
    # Linear equation: converged after step 2, step 3 must not flip the flag back.
    _, converged = newton_raphson(lambda x: x - 1.0, jnp.array([0.0]), max_iter=3)
    assert converged.item() is True


@pytest.mark.parametrize("builder", [build_bdf, build_am], ids=["bdf1", "am1"])
def test_multistep_step_is_jit_compatible_with_tolerance(builder):
    drift = lambda t, y: -y
    step = builder(drift, order=1, newton_iters=3, tol=1e-6)
    state = init_lms(0.0, jnp.array([1.0]), drift=drift)
    new_state, info = jax.jit(step)(state, 0.1)
    assert jnp.all(jnp.isfinite(new_state.y0))
    assert bool(info.converged) in (True, False)


@pytest.mark.parametrize("builder", [build_bdf, build_am], ids=["bdf1", "am1"])
def test_multistep_newton_is_stable_on_stiff_problem(builder):
    # y' = -1000 y with dt = 0.1: a fixed-point (Picard) step multiplies y by -99.
    drift = lambda t, y: -1000.0 * y
    step = builder(drift, order=1)
    state = init_lms(0.0, jnp.array([1.0]), drift=drift)
    for _ in range(5):
        state, _ = step(state, 0.1)
    assert jnp.all(jnp.abs(state.y0) <= 1.0)


def _affine_reference(A, b, x0, ts):
    """Exact solution of y' = A y + b via the augmented matrix [[A, b], [0, 0]]."""
    n = A.shape[0]
    aug = np.zeros((n + 1, n + 1))
    aug[:n, :n] = A
    aug[:n, n] = b
    x_aug = np.concatenate([x0, [1.0]])
    return np.array([(expm(aug * t) @ x_aug)[:n] for t in ts])


def test_linear_exact_dense_affine_constant_a():
    A = np.array([[-1.0, 2.0], [0.0, -3.0]])
    b = np.array([1.0, 0.5])
    x0 = np.array([1.0, -1.0])
    ts = np.linspace(0.0, 1.0, 5)
    with jax.enable_x64():
        drift = linear_drift(A=jnp.asarray(A), b=lambda t: jnp.asarray(b))
        result = odeint(
            drift,
            jnp.asarray(x0),
            jnp.asarray(ts),
            method="linear_exact",
            dtype=jnp.float64,
        )
    assert np.allclose(result, _affine_reference(A, b, x0, ts), atol=1e-10)


def test_linear_exact_dense_affine_time_varying_a():
    A = np.array([[-1.0, 2.0], [0.0, -3.0]])
    b = np.array([1.0, 0.5])
    x0 = np.array([1.0, -1.0])
    ts = np.linspace(0.0, 1.0, 5)
    with jax.enable_x64():
        drift = linear_drift(
            A=lambda t: jnp.asarray(A) * (1.0 + 0.0 * t), b=lambda t: jnp.asarray(b)
        )
        result = odeint(
            drift,
            jnp.asarray(x0),
            jnp.asarray(ts),
            method="linear_exact",
            dtype=jnp.float64,
        )
    assert np.allclose(result, _affine_reference(A, b, x0, ts), atol=1e-10)


def _sample_covariance(drift, diffusion, dt, n=20000):
    """Covariance of one exact SDE step from y0 = 0 over many independent keys."""
    step = build_linear_exact_sde_step(drift, diffusion)
    state = LinearExactSDEState(t0=jnp.asarray(0.0), y0=jnp.zeros(2))
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    samples = jax.vmap(lambda k: step(k, state, dt)[0].y0)(keys)
    return np.cov(np.asarray(samples).T)


def _integrated_covariance(A, GG, dt, num=4001):
    """Q = integral_0^dt exp(A s) GG exp(A^T s) ds by the trapezoid rule."""
    s = np.linspace(0.0, dt, num)
    vals = np.array([expm(A * si) @ GG @ expm(A.T * si) for si in s])
    return np.trapezoid(vals, s, axis=0)


def test_linear_exact_sde_diagonal_noise_covariance_includes_drift():
    # Scalar A with diagonal G has a closed form, but the old code ignored A.
    dt = 0.5
    G = jnp.array([1.0, 2.0])
    drift = linear_drift(A=jnp.array(-2.0))
    cov = _sample_covariance(drift, const_diffusion(G=G), dt)
    expected = _integrated_covariance(-2.0 * np.eye(2), np.diag(np.asarray(G) ** 2), dt)
    assert np.allclose(np.diag(cov), np.diag(expected), rtol=0.05)


def test_linear_exact_sde_dense_covariance_has_no_extra_dt():
    dt = 0.5
    A = np.array([[-1.0, 2.0], [0.0, -3.0]])
    G = np.array([[1.0, 0.5], [0.0, 1.0]])
    drift = linear_drift(A=jnp.asarray(A))
    cov = _sample_covariance(drift, const_diffusion(G=jnp.asarray(G)), dt)
    expected = _integrated_covariance(A, G @ G.T, dt)
    assert np.allclose(cov, expected, rtol=0.05, atol=1e-3)


def test_linear_exact_sde_dense_mean_with_bias():
    # Zero noise makes the step deterministic, so the mean is checked exactly.
    dt = 0.5
    A = np.array([[-1.0, 2.0], [0.0, -3.0]])
    b = np.array([1.0, 0.5])
    x0 = np.array([1.0, -1.0])
    drift = linear_drift(A=jnp.asarray(A), b=lambda t: jnp.asarray(b))
    step = build_linear_exact_sde_step(drift, const_diffusion(G=jnp.zeros((2, 2))))
    state = LinearExactSDEState(t0=jnp.asarray(0.0), y0=jnp.asarray(x0))
    y1 = step(jax.random.PRNGKey(0), state, dt)[0].y0
    assert np.allclose(y1, _affine_reference(A, b, x0, [dt])[0], atol=1e-3)


def test_phi_matrices_match_scalar_formulas():
    z = -0.5
    dt = 0.5
    A = jnp.array([[z / dt]])  # A * dt = z
    phi0, phi1, phi2 = phi_matrices(A * dt, k=2)
    assert jnp.allclose(phi0, np.exp(z), atol=1e-6)
    assert jnp.allclose(phi1, (np.exp(z) - 1) / z, atol=1e-6)
    assert jnp.allclose(phi2, (np.exp(z) - 1 - z) / z**2, atol=1e-6)


def test_exp_euler_is_exact_for_affine_linear_ode():
    # exp_euler uses phi_1, so it is exact for y' = A y + c with constant c.
    A = np.array([[-1.0, 2.0], [0.0, -3.0]])
    c = np.array([1.0, 0.5])
    x0 = np.array([1.0, -1.0])
    dt = 0.3
    drift = lambda t, y: jnp.asarray(A) @ y + jnp.asarray(c)
    with jax.enable_x64():
        step = build_exp_euler_step(drift)
        state = ExpODEState(
            t0=jnp.asarray(0.0), y0=jnp.asarray(x0), f0=drift(0.0, jnp.asarray(x0))
        )
        y1 = step(state, dt)[0].y0
    assert np.allclose(y1, _affine_reference(A, c, x0, [dt])[0], atol=1e-10)
