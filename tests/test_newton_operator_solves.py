"""Newton root finding with operator linear solves and implicit gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.utils.odeutil.solvers.multistep import build_am, build_bdf, init_lms
from probjax.utils.odeutil.solvers.rk_implicit import (
    ImpEulerState,
    build_implicit_euler_step,
    imp_trapez,
)
from probjax.utils.solver import newton_raphson, root

NEWTON_OPTIONS = [
    dict(jacobian="dense", linear_solver="auto"),
    dict(jacobian="dense", linear_solver="gmres"),
    dict(jacobian="matrix_free", linear_solver="auto"),
    dict(jacobian="matrix_free", linear_solver="bicgstab"),
]


def _cube_root(a, **options):
    return newton_raphson(lambda x: x**3 - a, jnp.ones_like(a), **options)


@pytest.mark.parametrize("options", NEWTON_OPTIONS)
def test_newton_root_and_implicit_gradient(options):
    a = jnp.array([2.0, 5.0, 0.5])
    x, converged = _cube_root(a, **options)
    assert bool(converged)
    np.testing.assert_allclose(x, a ** (1 / 3), rtol=1e-5)
    # Implicit function theorem: d(a^(1/3))/da = a^(-2/3) / 3.
    grad = jax.jit(jax.grad(lambda a: _cube_root(a, **options)[0].sum()))(a)
    np.testing.assert_allclose(grad, a ** (-2 / 3) / 3, rtol=1e-4)


def test_newton_on_shaped_state_and_under_vmap():
    a = jnp.arange(1.0, 7.0).reshape(2, 3)
    x, _ = newton_raphson(lambda x: x**3 - a, jnp.ones((2, 3)))
    np.testing.assert_allclose(x, a ** (1 / 3), rtol=1e-5)
    batched = jax.vmap(lambda a: _cube_root(a)[0])(a)
    np.testing.assert_allclose(batched, a ** (1 / 3), rtol=1e-5)


def test_newton_raw_gradient_matches_finite_differences():
    # Gradient of raw newton_raphson (not just through a solver step).
    def total(a):
        x, _ = newton_raphson(lambda x: x**3 - 2.0 * x - a, jnp.ones(3))
        return x.sum()

    with jax.enable_x64():
        a = jnp.asarray([1.0, 2.0, 3.0])
        grad = jax.grad(total)(a)
        eps = 1e-6
        basis = np.eye(3)
        finite = np.array(
            [(total(a + eps * b) - total(a - eps * b)) / (2 * eps) for b in basis]
        )
        np.testing.assert_allclose(grad, finite, rtol=1e-6)


def test_newton_with_lineax_solver():
    lx = pytest.importorskip("lineax")
    x, _ = _cube_root(jnp.array([2.0, 5.0]), linear_solver=lx.LU())
    np.testing.assert_allclose(x, jnp.array([2.0, 5.0]) ** (1 / 3), rtol=1e-5)


def test_newton_with_lineax_solver_gradient():
    lx = pytest.importorskip("lineax")

    def total(a):
        x, _ = _cube_root(a, linear_solver=lx.LU())
        return x.sum()

    with jax.enable_x64():
        a = jnp.asarray([2.0, 5.0])
        grad = jax.grad(total)(a)
        np.testing.assert_allclose(grad, a ** (-2 / 3) / 3, rtol=1e-4)


def test_non_converged_root_reports_flag_without_error():
    # One iteration cannot solve x^3 - 50 from ones: the flag (not an
    # exception) reports it, and callers keep stepping. Gradients are only
    # valid when converged (see newton_raphson docstring).
    x, converged = newton_raphson(
        lambda x: x**3 - 50.0, jnp.ones(2), tol=1e-12, max_iter=1
    )
    assert not bool(converged)
    assert bool(jnp.all(jnp.isfinite(x)))


def test_newton_rejects_non_positive_max_iter():
    with pytest.raises(ValueError, match="max_iter"):
        newton_raphson(lambda x: x, jnp.ones(2), max_iter=0)


def test_newton_rejects_unknown_jacobian_mode():
    with pytest.raises(ValueError, match="jacobian"):
        newton_raphson(lambda x: x, jnp.ones(2), jacobian="sparse")


def _stiff_drift(t, y):
    # Stiff linear coupling plus a mild nonlinearity.
    a = jnp.array([[-1000.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, -50.0]])
    return a @ y + 0.1 * jnp.sin(y)


@pytest.mark.parametrize("builder", [build_bdf, build_am], ids=["bdf2", "am2"])
def test_multistep_matrix_free_newton_matches_dense(builder):
    y0 = jnp.array([1.0, -0.5, 2.0])

    def run(**options):
        bootstrap = builder(_stiff_drift, order=1, newton_iters=4, **options)
        step = builder(
            _stiff_drift, order=2, newton_iters=4, bootstrap_step=bootstrap, **options
        )
        state = init_lms(0.0, y0, drift=_stiff_drift)
        for _ in range(6):
            state, _ = step(state, 0.05)
        return state.y0

    dense = run()
    matrix_free = run(jacobian="matrix_free", linear_solver="gmres")
    np.testing.assert_allclose(matrix_free, dense, rtol=1e-4, atol=1e-6)
    assert jnp.all(jnp.abs(dense) <= jnp.abs(y0).max())


def test_implicit_rk_and_euler_accept_newton_options():
    y0 = jnp.array([1.0, -0.5, 2.0])
    dense = imp_trapez(_stiff_drift)
    matrix_free = imp_trapez(_stiff_drift, jacobian="matrix_free")
    state = dense.init(0.0, y0, drift=_stiff_drift)
    np.testing.assert_allclose(
        matrix_free.step(state, 0.05)[0].y0,
        dense.step(state, 0.05)[0].y0,
        rtol=1e-4,
        atol=1e-6,
    )
    euler = build_implicit_euler_step(_stiff_drift, jacobian="matrix_free")
    euler_dense = build_implicit_euler_step(_stiff_drift, solver=root)
    euler_state = ImpEulerState(t0=jnp.asarray(0.0), y0=y0)
    np.testing.assert_allclose(
        euler(euler_state, 0.05)[0].y0,
        euler_dense(euler_state, 0.05)[0].y0,
        rtol=1e-4,
        atol=1e-6,
    )


def test_implicit_step_gradient_matches_finite_differences():
    def final(scale):
        drift = lambda t, y: -scale * y**3
        step = build_bdf(drift, order=1, newton_iters=6)
        state = init_lms(0.0, jnp.array([1.0, 0.5]), drift=drift)
        for _ in range(3):
            state, _ = step(state, 0.1)
        return state.y0.sum()

    with jax.enable_x64():
        scale = jnp.asarray(2.0)
        grad = jax.grad(final)(scale)
        eps = 1e-6
        finite = (final(scale + eps) - final(scale - eps)) / (2 * eps)
        np.testing.assert_allclose(grad, finite, rtol=1e-6)
