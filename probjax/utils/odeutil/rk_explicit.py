from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from probjax.utils.odeutil.base import ODEInfo, ODESolverAPI, ODEState, register_method

# 1st order
# Euler's method

info = {
    "explicit": True,
    "order": 1,
    "info": "Euler's method",
    "adaptive": False,
}


# For efficiency, we use a custom implementation of Euler's method
class EulerInfo(ODEInfo):
    pass


class EulerState(ODEState):
    t0: Array
    y0: Array


def init_euler(t0: ArrayLike, y0: Array, *args, **kwargs) -> ODEState:
    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)
    return EulerState(t0=t0, y0=y0)


def build_euler_step(drift: Callable, dtype: jnp.dtype = jnp.float32):
    def step_fn(state: ODEState, dt: ArrayLike, *args) -> Tuple[ODEState, ODEInfo]:
        y0 = state.y0
        t0 = state.t0
        y1 = y0 + drift(t0, y0, *args) * dt
        return EulerState(t0=t0 + dt, y0=y1), EulerInfo()

    return step_fn


class euler(ODESolverAPI):
    init = init_euler
    build_step = build_euler_step


register_method("euler", euler, info)


# General explicit Runge-Kutta methods


class RKState(ODEState):
    t0: Array
    y0: Array
    f0: Optional[Array]


class RKInfo(ODEInfo):
    k: Array  # Intermediate steps
    y1_error: Optional[Array]  # Error estimate
    y1_mid: Optional[Array]  # Midpoint


def init_rk(t0: ArrayLike, y0: Array, *args, drift=None) -> ODEState:
    t0 = jnp.asarray(t0)
    y0 = jnp.asarray(y0)
    f0 = drift(t0, y0, *args) if drift is not None else None
    return RKState(t0=t0, y0=y0, f0=f0)


def build_rk_step(
    drift: Callable,
    c: Array,
    A: Array,
    b_sol: Array,
    b_error: Optional[Array] = None,
    b_mid: Optional[Array] = None,
    last_equals_next: bool = False,
):
    stages = c.shape[0]

    def rk_step_fn(state: RKState, dt: ArrayLike, *args) -> Tuple[RKState, RKInfo]:
        t0 = state.t0
        y0 = state.y0
        f0 = state.f0 if not last_equals_next else drift(t0, y0, *args)

        def body_fun(i, k):
            ti = t0 + dt * c[i]
            yi = y0 + dt * jnp.dot(A[i, :], k)
            ft = drift(ti, yi, *args)
            return k.at[i, :].set(ft)

        f0 = drift(t0, y0, *args) if f0 is None else f0
        d = f0.shape[0] if f0.ndim > 0 else 1
        k = jnp.zeros((stages, d), f0.dtype).at[0, :].set(f0)
        k = jax.lax.fori_loop(1, stages + 1, body_fun, k)

        y1 = (dt * jnp.dot(b_sol, k) + y0).reshape(y0.shape)
        f1 = k[-1].reshape(f0.shape) if last_equals_next else None

        y1_error = None if b_error is None else dt * jnp.dot(b_error, k)

        y1_mid = None if b_mid is None else dt * jnp.dot(b_mid, k) + y0

        state = RKState(t0=t0 + dt, y0=y1, f0=f1)
        info = RKInfo(k=k, y1_error=y1_error, y1_mid=y1_mid)
        return state, info

    return rk_step_fn


def build_rk_method(
    name: str,
    build_butcher_tableau: Callable,
    last_equals_next: bool = False,
):
    if last_equals_next:

        def init_method(t0, y0, *args, drift=None, **kwargs):
            return init_rk(t0, y0, *args, drift=drift)

    else:

        def init_method(t0, y0, *args, **kwargs):
            return init_rk(t0, y0, *args)

    def build_step_method(drift: Callable, dtype: jnp.dtype = jnp.float32):
        c, A, b_sol, b_error, b_mid = build_butcher_tableau(dtype)

        return build_rk_step(
            drift=drift,
            c=c,
            A=A,
            b_sol=b_sol,
            b_error=b_error,
            b_mid=b_mid,
            last_equals_next=last_equals_next,
        )

    class RKMethod(ODESolverAPI):
        init = init_method
        build_step = build_step_method

    RKMethod.__name__ = name

    return RKMethod


# 2nd order
# Heun's method
def build_heun_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 1.0], dtype=dtype)
    A = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    b_sol = jnp.array([0.5, 0.5], dtype=dtype)
    b_error = None
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# Adaptive Heun's method
def build_heun_adaptive_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 1.0], dtype=dtype)
    A = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    b_sol = jnp.array([0.5, 0.5], dtype=dtype)
    b_error = jnp.array([1.0, 0.0], dtype=dtype)
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# Midpoint method
def build_midpoint_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 0.5], dtype=dtype)
    A = jnp.array([[0.0, 0.0], [0.5, 0.0]], dtype=dtype)
    b_sol = jnp.array([0.0, 1.0], dtype=dtype)
    b_error = None
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# Ralston's method
def build_ralston_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 2.0 / 3.0], dtype=dtype)
    A = jnp.array([[0.0, 0.0], [2.0 / 3.0, 0.0]], dtype=dtype)
    b_sol = jnp.array([0.25, 0.75], dtype=dtype)
    b_error = None
    b_mid = None

    return c, A, b_sol, b_error, b_mid


heun = build_rk_method("heun", build_heun_tablau, last_equals_next=True)
heun_euler = build_rk_method(
    "heun_euler", build_heun_adaptive_tablau, last_equals_next=True
)
midpoint = build_rk_method("midpoint", build_midpoint_tablau, last_equals_next=False)
ralston = build_rk_method("ralston", build_ralston_tablau, last_equals_next=False)


# 3rd order
# Kutta's third-order method
def build_kutta_third_order_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 0.5, 1.0], dtype=dtype)
    A = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-1.0, 2.0, 0.0]], dtype=dtype)
    b_sol = jnp.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=dtype)
    b_error = None
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# Fehlberg's RK3(2) method (explicit) (adaptive)
def build_fehlberg_rk32_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 1.0 / 2.0, 1.0], dtype=dtype)
    A = jnp.array(
        [[0.0, 0.0, 0.0], [1.0 / 2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype
    )
    b_sol = jnp.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=dtype)
    b_error = jnp.array([1.0 / 6.0, 0.0, 1.0 / 3.0], dtype=dtype)
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# Bosh 3 method
def build_bosh3_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 0.5, 0.75], dtype=dtype)
    A = jnp.array([[0, 0, 0], [0.5, 0, 0], [0, 0.75, 0]], dtype=dtype)
    b_sol = jnp.array([2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0], dtype=dtype)
    b_error = None
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# Bogacki-Shampine method of 3rd order
def build_bogacki_shampine_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 0.5, 0.75], dtype=dtype)
    A = jnp.array([[0, 0, 0], [0.5, 0, 0], [0, 0.75, 0]], dtype=dtype)
    b_sol = jnp.array([2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0], dtype=dtype)
    b_error = jnp.array([7.0 / 24.0, 0.25, 1.0 / 3.0], dtype=dtype)
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# Heun's third-order method
def build_heun3_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=dtype)
    A = jnp.array(
        [[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0], [0.0, 2.0 / 3.0, 0.0]], dtype=dtype
    )
    b_sol = jnp.array([1.0 / 4.0, 0.0, 3.0 / 4.0], dtype=dtype)
    b_error = None
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# Van der Houwen's/Wray's method
def build_vanderhouwen_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 8.0 / 15.0, 2.0 / 3.0], dtype=dtype)
    A = jnp.array(
        [[0.0, 0.0, 0.0], [8.0 / 15.0, 0.0, 0.0], [1.0 / 4.0, 5.0 / 12.0, 0.0]],
        dtype=dtype,
    )
    b_sol = jnp.array([1.0 / 4.0, 0.0, 3.0 / 4.0], dtype=dtype)
    b_error = None
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# Ralston's third-order method
def build_ralston3_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=dtype)
    A = jnp.array(
        [[0.0, 0.0, 0.0], [1.0 / 2.0, 0.0, 0.0], [0.0, 3.0 / 4.0, 0.0]], dtype=dtype
    )
    b_sol = jnp.array([2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0], dtype=dtype)
    b_error = None
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# SSPRK3
def build_ssprk3_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 0.5, 1.0], dtype=dtype)
    A = jnp.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]], dtype=dtype)
    b_sol = jnp.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], dtype=dtype)
    b_error = None
    b_mid = None

    return c, A, b_sol, b_error, b_mid


rk3 = build_rk_method("rk3", build_kutta_third_order_tablau, last_equals_next=True)
rk32 = build_rk_method("rk32", build_fehlberg_rk32_tablau, last_equals_next=True)
bosh3 = build_rk_method("bosh3", build_bosh3_tablau, last_equals_next=False)
bogacki_shampine = build_rk_method(
    "bogacki_shampine",
    build_bogacki_shampine_tablau,
    last_equals_next=False,  # This might be wrong
)
heun3 = build_rk_method("heun3", build_heun3_tablau, last_equals_next=False)
vanderhouwen = build_rk_method(
    "vanderhouwen", build_vanderhouwen_tablau, last_equals_next=False
)
ralston3 = build_rk_method("ralston3", build_ralston3_tablau, last_equals_next=False)
ssprk3 = build_rk_method("ssprk3", build_ssprk3_tablau, last_equals_next=True)


# 4th order


# Classic Runge-Kutta method
def build_rk4_tablau(dtype: jnp.dtype):
    c = jnp.array([0, 0.5, 0.5, 1], dtype=dtype)
    A = jnp.array(
        [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 1, 0.5, 0], [0, 0, 1.0, 0]], dtype=dtype
    )
    b_sol = jnp.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], dtype=dtype)
    b_error = None

    return c, A, b_sol, b_error, None


# RK4(3) adaptive method
def build_rk43_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 0.5, 0.75, 1.0], dtype=dtype)
    A = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.75, 0.0, 0.0],
            [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0],
        ],
        dtype=dtype,
    )
    b_sol = jnp.array([2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0], dtype=dtype)
    b_error = jnp.array([7.0 / 24.0, 0.25, 1.0 / 3.0, 1.0 / 8.0], dtype=dtype)
    b_mid = None

    return c, A, b_sol, b_error, b_mid


def build_ssprk4_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 0.5, 0.5, 1.0], dtype=dtype)
    A = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=dtype,
    )
    b_sol = jnp.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0], dtype=dtype)
    b_error = None  # Non-adaptive
    b_mid = None

    return c, A, b_sol, b_error, b_mid


def build_ralston4_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 0.4, 0.455737, 1.0], dtype=dtype)
    A = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0],
            [0.29697761, 0.15875964, 0.0, 0.0],
            [0.2181004, -3.0509652, 3.83286476, 0.0],
        ],
        dtype=dtype,
    )
    b_sol = jnp.array([0.17476028, 0.55148066, 0.27404808, 0.0], dtype=dtype)
    b_error = None  # Non-adaptive
    b_mid = jnp.array([0.4, 0.6, 0.0, 0.0], dtype=dtype)

    return c, A, b_sol, b_error, b_mid


def build_rk38_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=dtype)
    A = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0, 0.0],
            [-1.0 / 3.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, 0.0],
        ],
        dtype=dtype,
    )
    b_sol = jnp.array([1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0], dtype=dtype)
    b_error = None  # Non-adaptive
    b_mid = None

    return c, A, b_sol, b_error, b_mid


rk4 = build_rk_method("rk4", build_rk4_tablau, last_equals_next=True)
rk43 = build_rk_method("rk43", build_rk43_tablau, last_equals_next=True)
ssprk4 = build_rk_method("ssprk4", build_ssprk4_tablau, last_equals_next=True)
ralston4 = build_rk_method("ralston4", build_ralston4_tablau, last_equals_next=True)
rk38 = build_rk_method("rk38", build_rk38_tablau, last_equals_next=True)

# Order 5


# Dormand-Prince method
def build_dopri5_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0], dtype=dtype)
    A = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
            [
                19372.0 / 6561.0,
                -25360.0 / 2187.0,
                64448.0 / 6561.0,
                -212.0 / 729.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                9017.0 / 3168.0,
                -355.0 / 33.0,
                46732.0 / 5247.0,
                49.0 / 176.0,
                -5103.0 / 18656.0,
                0.0,
                0.0,
            ],
            [
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
                0.0,
            ],
        ],
        dtype=dtype,
    )
    b_sol = jnp.array(
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ],
        dtype=dtype,
    )
    b_error = jnp.array(
        [
            71.0 / 57600.0,
            0.0,
            -71.0 / 16695.0,
            71.0 / 1920.0,
            -17253.0 / 339200.0,
            22.0 / 525.0,
            -1.0 / 40.0,
        ],
        dtype=dtype,
    )
    b_mid = jnp.array(
        [
            6025192743 / 30085553152 / 2,
            0,
            51252292925 / 65400821598 / 2,
            -2691868925 / 45128329728 / 2,
            187940372067 / 1594534317056 / 2,
            -1776094331 / 19743644256 / 2,
            11237099 / 235043384 / 2,
        ],
        dtype=dtype,
    )

    return c, A, b_sol, b_error, b_mid


dopri5 = build_rk_method("dopri5", build_dopri5_tablau, last_equals_next=True)


# Order 6
# RK5
def build_rk6_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 1.0 / 3.0, 1.0 / 3.0, 0.5, 1.0, 5.0 / 6.0], dtype=dtype)
    A = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 8.0, 0.0, 3.0 / 8.0, 0.0, 0.0, 0.0],
            [1.0 / 2.0, 0.0, -3.0 / 2.0, 3.0 / 2.0, 0.0, 0.0],
            [-3.0 / 7.0, 2.0, 12.0 / 7.0, -12.0 / 7.0, 8.0 / 7.0, 0.0],
        ],
        dtype=dtype,
    )
    b_sol = jnp.array(
        [7.0 / 90.0, 0.0, 16.0 / 45.0, 2.0 / 15.0, 16.0 / 45.0, 7.0 / 90.0], dtype=dtype
    )
    b_error = None  # Non-adaptive
    b_mid = None

    return c, A, b_sol, b_error, b_mid


# RK65
def build_butcher6_tablau(dtype: jnp.dtype):
    c = jnp.array([0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0], dtype=dtype)
    A = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
            [
                19372.0 / 6561.0,
                -25360.0 / 2187.0,
                64448.0 / 6561.0,
                -212.0 / 729.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                9017.0 / 3168.0,
                -355.0 / 33.0,
                46732.0 / 5247.0,
                49.0 / 176.0,
                -5103.0 / 18656.0,
                0.0,
                0.0,
            ],
            [
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
                0.0,
            ],
        ],
        dtype=dtype,
    )
    b_sol = jnp.array(
        [41.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0],
        dtype=dtype,
    )
    b_error = jnp.array(
        [-41.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0, 0.0],
        dtype=dtype,
    )
    b_mid = None

    return c, A, b_sol, b_error, b_mid


rk6 = build_rk_method("rk6", build_rk6_tablau, last_equals_next=True)
rk65 = build_rk_method("rk65", build_butcher6_tablau, last_equals_next=True)

register_method(
    "heun",
    heun,
    {"explicit": True, "order": 2, "info": "Heun's method", "adaptive": False},
)

register_method(
    "heun_euler",
    heun_euler,
    {
        "explicit": True,
        "order": 2,
        "info": "Heun's method with error estimate based on Euler's method",
        "adaptive": True,
    },
)

register_method(
    "midpoint",
    midpoint,
    {"explicit": True, "order": 2, "info": "Midpoint method", "adaptive": False},
)

register_method(
    "ralston",
    ralston,
    {"explicit": True, "order": 2, "info": "Ralston's method", "adaptive": False},
)

register_method(
    "rk3",
    rk3,
    {
        "explicit": True,
        "order": 3,
        "info": "Kutta's third-order method",
        "adaptive": False,
    },
)

register_method(
    "rk32",
    rk32,
    {
        "explicit": True,
        "order": 3,
        "info": "Fehlberg's RK3(2) method (adaptive)",
        "adaptive": True,
    },
)

register_method(
    "bosh3",
    bosh3,
    {"explicit": True, "order": 3, "info": "Bosh 3 method", "adaptive": False},
)

register_method(
    "bogacki_shampine",
    bogacki_shampine,
    {
        "explicit": True,
        "order": 3,
        "info": "Bogacki-Shampine method with error estimation",
        "adaptive": True,
        "interpolation_order": 4,
    },
)

register_method(
    "heun3",
    heun3,
    {
        "explicit": True,
        "order": 3,
        "info": "Heun's third-order method",
        "adaptive": False,
    },
)

register_method(
    "vanderhouwen",
    vanderhouwen,
    {
        "explicit": True,
        "order": 3,
        "info": "Van der Houwen's/Wray's method",
        "adaptive": False,
    },
)

register_method(
    "ralston3",
    ralston3,
    {
        "explicit": True,
        "order": 3,
        "info": "Ralston's third-order method",
        "adaptive": False,
    },
)

register_method(
    "ssprk3",
    ssprk3,
    {"explicit": True, "order": 3, "info": "SSPRK3 method", "adaptive": False},
)

register_method(
    "rk4",
    rk4,
    {
        "explicit": True,
        "order": 4,
        "info": "Classic Runge-Kutta method",
        "adaptive": False,
    },
)


register_method(
    "ralston4",
    ralston4,
    {
        "explicit": True,
        "order": 4,
        "info": "Ralston's fourth-order method",
        "adaptive": False,
    },
)

register_method(
    "rk38",
    rk38,
    {
        "explicit": True,
        "order": 4,
        "info": "Runge-Kutta 3/8 rule (fourth-order)",
        "adaptive": False,
    },
)

register_method(
    "dopri5",
    dopri5,
    {
        "explicit": True,
        "order": 5,
        "info": "Dormand-Prince RK4(5) method with adaptive error control",
        "adaptive": True,
        "interpolation_order": 4,
    },
)


register_method(
    "rk6",
    rk6,
    {
        "explicit": True,
        "order": 6,
        "info": "Runge-Kutta sixth-order method",
        "adaptive": False,
    },
)
