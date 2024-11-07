from typing import Callable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def initial_step_size(
    drift: Callable,
    args,
    t0: ArrayLike,
    y0: ArrayLike,
    f0: ArrayLike,
    order: int,
    rtol: float,
    atol: float,
) -> Array:
    # Algorithm from:
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    dtype = y0.dtype

    scale = atol + jnp.abs(y0) * rtol
    d0 = jnp.linalg.norm(y0 / scale.astype(dtype))
    d1 = jnp.linalg.norm(f0 / scale.astype(dtype))

    h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
    y1 = y0 + h0.astype(dtype) * f0
    f1 = drift(t0 + h0, y1, *args)
    d2 = jnp.linalg.norm((f1 - f0) / scale.astype(dtype)) / h0

    h1 = jnp.where(
        (d1 <= 1e-15) & (d2 <= 1e-15),
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 / jnp.maximum(d1, d2)) ** (1.0 / (order + 1.0)),
    )

    return jnp.minimum(100.0 * h0, h1)


def mean_error_ratio(
    error_estimate: ArrayLike,
    y0: ArrayLike,
    y1: ArrayLike,
    rtol: float,
    atol: float,
    norm: float | int = 2,
):
    err_tol = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
    err_ratio = error_estimate / err_tol.astype(error_estimate.dtype)
    return jnp.linalg.norm(err_ratio, ord=norm) / jnp.sqrt(len(err_ratio))


def optimal_step_size(
    last_step: ArrayLike,
    mean_error_ratio: ArrayLike,
    maxerror: float = 1.0,
    safety: float = 0.9,
    ifactor: float = 10.0,
    dfactor: float = 0.2,
    order: int = 5.0,
):
    """Compute optimal Runge-Kutta stepsize."""
    dfactor = jnp.where(mean_error_ratio < maxerror, 1.0, dfactor)

    factor = jnp.minimum(
        ifactor, jnp.maximum(mean_error_ratio ** (-1.0 / order) * safety, dfactor)
    )
    return jnp.where(mean_error_ratio == 0, last_step * ifactor, last_step * factor)


def fit_cubic_hermite(y0, y1, dy0, dy1, dt):
    """Be f(t) = a * t**3 + b * t**2 + c * t + d, then this function returns the
    coefficients a, b, c, d, which solve the system of equations:
        f(0) = y0
        f(1) = y1
        f'(0) = dy0
        f'(1) = dy1
    """
    h = dt
    c = dy0
    d = y0
    a = (2 * (y0 - y1) + h * (dy0 + dy1)) / h**3
    b = (3 * (y1 - y0) - h * (2 * dy0 + dy1)) / h**2
    return a, b, c, d


def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
    """Be f(t) = a * t**4 + b * t**3 + c * t**2 + d * t + e, then this function returns
    the coefficients a, b, c, d, e, which solve the system of equations:
        f(0) = y0
        f(1) = y1
        f(1/2) = y_mid
        f'(0) = dy0
        f'(1) = dy1
    """
    a = -2.0 * dt * dy0 + 2.0 * dt * dy1 - 8.0 * y0 - 8.0 * y1 + 16.0 * y_mid
    b = 5.0 * dt * dy0 - 3.0 * dt * dy1 + 18.0 * y0 + 14.0 * y1 - 32.0 * y_mid
    c = -4.0 * dt * dy0 + dt * dy1 - 11.0 * y0 - 5.0 * y1 + 16.0 * y_mid
    d = dt * dy0
    e = y0
    return a, b, c, d, e


def fit_3rd_order_polynomial(y0, y1, dy0, dy1, dt):
    """Be f(t) = a * t**3 + b * t**2 + c * t + d, then this function returns
    the coefficients a, b, c, d, which solve the system of equations:
    f(0) = y0
    f(1) = y1
    f'(0) = dy0
    f'(1) = dy1
    """
    d = y0
    c = dy0 * dt
    b = (-2 / 3 * c - d / dt + y1 / dt - dy1 / 3) * dt**2
    a = (y1 - b * dt**2 - c * dt - d) * dt**3

    return a, b, c, d


def interp_fit(y0, y1, f0, f1, dt, y_mid=None):
    if y_mid is not None:
        # We can use a 4th order polynomial (3 points and 2 gradients)
        return jnp.asarray(fit_4th_order_polynomial(y0, y1, y_mid, f0, f1, dt))
    else:
        # We can only use a 3rd order polynomial (2 points and 2 gradients)
        return jnp.asarray(fit_3rd_order_polynomial(y0, y1, f0, f1, dt))
