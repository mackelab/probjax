"""Passing solver classes and configured solvers to odeint/sdeint."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.utils._solver_common import ConfiguredSolver
from probjax.utils.odeint import odeint
from probjax.utils.odeutil.solvers import get_method
from probjax.utils.odeutil.solvers.multistep import bdf1
from probjax.utils.odeutil.solvers.rk_implicit import imp_trapez, implicit_euler
from probjax.utils.sdeint import sdeint
from probjax.utils.sdeutil import milstein


def stiff(t, y):
    return -50.0 * y + jnp.sin(t)


Y0 = jnp.ones(3)
TS = jnp.linspace(0.0, 1.0, 21)


def test_configured_solver_is_a_hashable_value():
    configured = bdf1(newton_iters=2, jacobian="matrix_free")
    assert isinstance(configured, ConfiguredSolver)
    assert configured == bdf1(jacobian="matrix_free", newton_iters=2)
    assert hash(configured) == hash(bdf1(jacobian="matrix_free", newton_iters=2))
    assert configured.info == get_method("bdf1")[1]
    # Called with a drift, a solver class still builds the solver directly.
    assert callable(bdf1(stiff).step)


@pytest.mark.parametrize(
    "solver",
    [
        implicit_euler,
        implicit_euler(),
        implicit_euler(jacobian="matrix_free", linear_solver="gmres"),
    ],
    ids=["class", "configured", "matrix_free"],
)
def test_odeint_accepts_solver_objects(solver):
    reference = odeint(stiff, Y0, TS, solver="implicit_euler")
    np.testing.assert_allclose(
        odeint(stiff, Y0, TS, solver=solver), reference, rtol=1e-4
    )


def test_configured_options_reach_the_solver():
    # Unregistered by name until imported, but usable as an object.
    trajectory = odeint(stiff, Y0, TS, solver=bdf1(newton_iters=3))
    assert jnp.all(jnp.isfinite(trajectory))
    dense = odeint(stiff, Y0, TS, solver=imp_trapez())
    matrix_free = odeint(stiff, Y0, TS, solver=imp_trapez(jacobian="matrix_free"))
    np.testing.assert_allclose(matrix_free, dense, rtol=1e-4, atol=1e-6)
    with pytest.raises(TypeError, match="not_an_option"):
        odeint(stiff, Y0, TS, solver=implicit_euler(not_an_option=1))


def test_configured_solver_does_not_retrace_under_jit():
    traces = []

    @jax.jit
    def solve(y0):
        traces.append(None)
        return odeint(stiff, y0, TS, solver=implicit_euler(jacobian="matrix_free"))

    solve(Y0)
    solve(2 * Y0)
    assert len(traces) == 1


def test_sdeint_accepts_configured_solver():
    diffusion = lambda t, y: 0.1 * y
    key = jax.random.key(0)
    ts = jnp.linspace(0.0, 0.5, 11)
    drift = lambda t, y: -y
    for sde_type in ("ito", "stratonovich"):
        np.testing.assert_allclose(
            sdeint(key, drift, diffusion, Y0, ts, solver=milstein(), sde_type=sde_type),
            sdeint(key, drift, diffusion, Y0, ts, solver="milstein", sde_type=sde_type),
        )


def test_method_is_a_deprecated_alias():
    with pytest.warns(DeprecationWarning, match="solver="):
        legacy = odeint(stiff, Y0, TS, method="rk4")
    np.testing.assert_allclose(legacy, odeint(stiff, Y0, TS, solver="rk4"))
    with pytest.warns(DeprecationWarning, match="solver="):
        sdeint(jax.random.key(0), stiff, lambda t, y: y, Y0, TS, method="milstein")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        odeint(stiff, Y0, TS, solver="rk4")


def test_unknown_solver_name_lists_available():
    with pytest.raises(ValueError, match="dopri5"):
        odeint(stiff, Y0, TS, solver="not_a_solver")
