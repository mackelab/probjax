from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from probjax.utils.odeutil.base import ODEInfo, ODESolverAPI, ODEState, register_method
from probjax.utils.solver import root

info = {
    "explicit": False,
    "order": 1,
    "info": "Implicit euler's method",
    "adaptive": False,
}


# For efficiency, we use a custom implementation of Euler's method
class ImpEulerInfo(ODEInfo):
    solver_info: Any


class ImpEulerState(ODEState):
    t0: Array
    y0: Array


def init_implicit_euler(t0: ArrayLike, y0: Array, *args, **kwargs) -> ODEState:
    return ImpEulerState(t0=t0, y0=y0)


def build_implicit_euler_step(
    drift: Callable, dtype: jnp.dtype = jnp.float32, solver: Callable = root
) -> Callable:
    def step_fn(state: ImpEulerState, dt: ArrayLike, *args) -> Tuple[ODEState, ODEInfo]:
        y0 = state.y0
        t0 = state.t0
        t1 = t0 + dt

        def f(y_next):
            y_pred = y0 + dt * drift(t1, y_next, *args)
            return y_next - y_pred

        y1, info = solver(f, y0)

        return ImpEulerState(t0=t1, y0=y1), ImpEulerInfo(solver_info=info)

    return step_fn


class implicit_euler(ODESolverAPI):
    init = init_implicit_euler
    build_step = build_implicit_euler_step


register_method("implicit_euler", implicit_euler, info=info)


class ImpRKState(ODEState):
    t0: Array
    y0: Array
    f0: Optional[Array]


class ImpRKInfo(ODEInfo):
    k: Array  # Intermediate steps
    y1_error: Optional[Array]  # Error estimate
    y1_mid: Optional[Array]  # Midpoint
    solver_info: Any


def init_imp_rk(
    t0: ArrayLike, y0: Array, *args, drift: Optional[Callable] = None
) -> ODEState:
    f0 = drift(t0, y0, *args) if drift is not None else None
    return ImpRKState(t0=t0, y0=y0, f0=f0)


def build_implicit_rk_step(
    drift: Callable,
    c: Array,
    A: Array,
    b_sol: Array,
    b_error: Optional[Array] = None,
    b_mid: Optional[Array] = None,
    last_equals_next: bool = False,
):
    stages = c.shape[0]

    def imp_rk_step_fn(
        state: ImpRKState, dt: ArrayLike, *args
    ) -> Tuple[ImpRKState, ImpRKInfo]:
        t0 = state.t0
        y0 = state.y0
        f0 = state.f0 if not last_equals_next else drift(t0, y0)

        ts = t0 + dt * c
        ts = ts.reshape(-1, 1)

        # Solve implicit equation
        def f(k):
            ys = y0 + dt * jnp.dot(A, k)
            return k - jax.vmap(
                drift,
                in_axes=(
                    0,
                    0,
                )
                + (None,) * len(args),
            )(ts, ys, *args)

        # Uses root finding to solve implicit equation
        k0 = jnp.ones((stages, f0.shape[0]), f0.dtype) * f0
        k, info = root(f, k0)

        # Compute solution
        y1 = y0 + dt * jnp.dot(b_sol, k)
        f1 = k[-1] if last_equals_next else drift(t0 + dt, y1)
        y1_error = None if b_error is None else dt * jnp.dot(b_error, k)
        y1_mid = None if b_mid is None else dt * jnp.dot(b_mid, k) + y0

        state = ImpRKState(t0=t0 + dt, y0=y1, f0=f1)
        info = ImpRKInfo(k=k, y1_error=y1_error, y1_mid=y1_mid, solver_info=info)
        return state, info

    return imp_rk_step_fn


def build_imp_rk_method(
    name: str,
    build_butcher_tableau: Callable,
    last_equals_next: bool = False,
):
    def init_method(t0, y0, drift):
        f0 = drift(t0, y0) if not last_equals_next else None
        return ImpRKState(t0, y0, f0)

    def build_step_method(drift: Callable, dtype: jnp.dtype = jnp.float32):
        c, A, b_sol, b_error, b_mid = build_butcher_tableau(dtype)

        return build_implicit_rk_step(
            drift=drift,
            c=c,
            A=A,
            b_sol=b_sol,
            b_error=b_error,
            b_mid=b_mid,
            last_equals_next=last_equals_next,
        )

    class ImpRKMethod(ODESolverAPI):
        init = init_method
        build_step = build_step_method

    ImpRKMethod.__name__ = name

    return ImpRKMethod


# 2nd order
# Implicit trapezoidal rule
def build_imp_trapezoidal(dtype: jnp.dtype = jnp.float32):
    c = jnp.array([0, 1], dtype=dtype)
    A = jnp.array([[0, 0], [1, 0]], dtype=dtype)
    b_sol = jnp.array([1 / 2, 1 / 2], dtype=dtype)
    b_error = None
    b_mid = None
    return c, A, b_sol, b_error, b_mid


imp_trapez = build_imp_rk_method("imp_trapez", build_imp_trapezoidal)
register_method(
    "imp_trapez", imp_trapez, info={"explicit": False, "order": 2, "adaptive": False}
)


# Implicit Crank-Nicolson method
def build_crank_nicolson(dtype: jnp.dtype = jnp.float32):
    c = jnp.array([0, 1.0], dtype=dtype)
    A = jnp.array([[0, 0], [0.5, 0.5]], dtype=dtype)
    b_sol = jnp.array([0.5, 0.5], dtype=dtype)
    b_error = None
    return c, A, b_sol, b_error, None


imp_crank_nicolson = build_imp_rk_method("imp_crank_nicolson", build_crank_nicolson)
register_method(
    "imp_crank_nicolson",
    imp_crank_nicolson,
    info={"explicit": False, "order": 2, "adaptive": False},
)
