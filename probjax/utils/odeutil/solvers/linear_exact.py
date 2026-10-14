from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from probjax.utils.functions import linear_drift
from probjax.utils.linear_operator import LinearOperator
from probjax.utils.odeutil.solvers.base import (
    ODEInfo,
    ODESolverAPI,
    ODEState,
    register_method,
)
from probjax.utils.typing import Array, ArrayLike, Callable


class LinearExactState(ODEState):
    t0: Array
    y0: Array


class LinearExactInfo(ODEInfo):
    pass


def init_linear_exact(
    t0: ArrayLike,
    y0: ArrayLike,
    *args,
    drift: Optional[Callable] = None,
    **kwargs,
) -> LinearExactState:
    if not isinstance(drift, linear_drift):
        raise TypeError("linear_exact requires `drift` to be a `linear_drift`")

    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)

    return LinearExactState(t0=t0, y0=y0)


def affine_flow(A: LinearOperator, b: Optional[Array], y0: Array, dt: Array) -> Array:
    """Solution of ``dy/dt = A y + b`` after ``dt``, for constant ``A`` and ``b``:
    ``exp(A dt) y0 + dt * phi_1(A dt) b``."""
    if b is None:
        return A.expm(dt) @ y0
    phi0, phi1 = A.phi(dt, k=1)
    return phi0 @ y0 + dt * (phi1 @ jnp.broadcast_to(b, y0.shape))


def build_linear_exact_step(drift: linear_drift) -> Callable:
    if not isinstance(drift, linear_drift):
        raise TypeError("linear_exact requires a `linear_drift`")

    def step_fn(
        state: LinearExactState, dt: ArrayLike, *args
    ) -> tuple[LinearExactState, LinearExactInfo]:
        t0, y0 = state.t0, state.y0
        dt = jnp.asarray(dt)
        # A time-varying A is frozen at the step start (exact for constant A).
        A = drift.operator(t0, y0.shape)
        b = None if drift.b is None else drift.b(t0 + dt, *args)
        return LinearExactState(
            t0=t0 + dt, y0=affine_flow(A, b, y0, dt)
        ), LinearExactInfo()

    return step_fn


class linear_exact(ODESolverAPI):
    init = staticmethod(init_linear_exact)
    build_step = staticmethod(build_linear_exact_step)


register_method(
    "linear_exact",
    linear_exact,
    info={
        "explicit": False,
        "order": 1,
        "info": "Exact solver for linear ODEs dy/dt = A y + b with constant A; "
        "time-varying A is frozen at each step start",
        "adaptive": False,
    },
)
