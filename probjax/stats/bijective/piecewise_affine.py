from functools import partial
from typing import Optional, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.core.custom_primitives.custom_inverse import custom_inverse


def _piecewise_affine_spline_fwd(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Forward pass for a monotone piecewise-linear spline with linear tails.
    Inside a bin: y = y_l + s * (x - x_l),  with s = (y_r - y_l) / (x_r - x_l).
    Returns (y, log|dy/dx|).
    """
    x = jnp.asarray(x)
    x_pos = jnp.asarray(x_pos)
    y_pos = jnp.asarray(y_pos)

    below_range = x <= x_pos[0]
    above_range = x >= x_pos[-1]

    # Bin selection mask
    correct_bin = jnp.logical_and(x >= x_pos[:-1], x < x_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate([
        jnp.array([True]),
        jnp.zeros(len(correct_bin) - 1, dtype=bool),
    ])
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)

    params = jnp.stack([x_pos, y_pos], axis=1)
    left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_l, y_l = left[0], left[1]
    x_r, y_r = right[0], right[1]

    dx = x_r - x_l
    s = (y_r - y_l) / dx

    y_mid = y_l + s * (x - x_l)
    logdet_mid = jnp.log(jnp.abs(s))

    # ----- lower tail -----
    slope_below = 1.0
    y_below = (x - x_pos[0]) * slope_below + y_pos[0]
    logdet_below = jnp.log(jnp.abs(slope_below))
    if x_min is not None and y_min is not None:
        denom = x_pos[0] - x_min
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_bl = (y_pos[0] - y_min) / denom
        y_lin = y_min + slope_bl * (x - x_min)
        y_lin = jnp.where(x <= x_min, y_min, y_lin)
        y_below = jnp.where(jnp.isnan(slope_bl), y_below, y_lin)
        logdet_below = jnp.where(
            jnp.isnan(slope_bl), logdet_below, jnp.log(jnp.abs(slope_bl))
        )

    # ----- upper tail -----
    slope_above = 1.0
    y_above = (x - x_pos[-1]) * slope_above + y_pos[-1]
    logdet_above = jnp.log(jnp.abs(slope_above))
    if x_max is not None and y_max is not None:
        denom = x_max - x_pos[-1]
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_ab = (y_max - y_pos[-1]) / denom
        y_lin = y_pos[-1] + slope_ab * (x - x_pos[-1])
        y_lin = jnp.where(x >= x_max, y_max, y_lin)
        y_above = jnp.where(jnp.isnan(slope_ab), y_above, y_lin)
        logdet_above = jnp.where(
            jnp.isnan(slope_ab), logdet_above, jnp.log(jnp.abs(slope_ab))
        )

    y = jnp.where(below_range, y_below, y_mid)
    y = jnp.where(above_range, y_above, y)

    logdet = jnp.where(below_range, logdet_below, logdet_mid)
    logdet = jnp.where(above_range, logdet_above, logdet)
    return y, logdet


def _piecewise_affine_spline_inv(
    y: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Inverse: inside a bin, x = x_l + (y - y_l) / s, with s = (y_r - y_l)/(x_r - x_l).
    Returns (x, log|dx/dy|).
    """
    y = jnp.asarray(y)
    x_pos = jnp.asarray(x_pos)
    y_pos = jnp.asarray(y_pos)

    below_range = y <= y_pos[0]
    above_range = y >= y_pos[-1]

    correct_bin = jnp.logical_and(y >= y_pos[:-1], y < y_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate([
        jnp.array([True]),
        jnp.zeros(len(correct_bin) - 1, dtype=bool),
    ])
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)

    params = jnp.stack([x_pos, y_pos], axis=1)
    left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_l, y_l = left[0], left[1]
    x_r, y_r = right[0], right[1]

    dx = x_r - x_l
    s = (y_r - y_l) / dx

    x_mid = x_l + (y - y_l) / s
    logdet_mid = -jnp.log(jnp.abs(s))

    # lower tail (unbounded / bounded)
    slope_below = 1.0  # / s[0]
    x_below = x_pos[0] + slope_below * (y - y_pos[0])
    logdet_below = jnp.log(jnp.abs(slope_below))
    if y_min is not None and x_min is not None:
        denom = y_pos[0] - y_min
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_bl = (x_pos[0] - x_min) / denom
        x_lin = x_min + slope_bl * (y - y_min)
        x_lin = jnp.where(y <= y_min, x_min, x_lin)
        x_below = jnp.where(jnp.isnan(slope_bl), x_below, x_lin)
        logdet_below = jnp.where(
            jnp.isnan(slope_bl), logdet_below, jnp.log(jnp.abs(slope_bl))
        )

    # upper tail
    slope_above = 1.0  # / s[-1]
    x_above = x_pos[-1] + slope_above * (y - y_pos[-1])
    logdet_above = jnp.log(jnp.abs(slope_above))
    if y_max is not None and x_max is not None:
        denom = y_max - y_pos[-1]
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_ab = (x_max - x_pos[-1]) / denom
        x_lin = x_pos[-1] + slope_ab * (y - y_pos[-1])
        x_lin = jnp.where(y >= y_max, x_max, x_lin)
        x_above = jnp.where(jnp.isnan(slope_ab), x_above, x_lin)
        logdet_above = jnp.where(
            jnp.isnan(slope_ab), logdet_above, jnp.log(jnp.abs(slope_ab))
        )

    x_out = jnp.where(below_range, x_below, x_mid)
    x_out = jnp.where(above_range, x_above, x_out)

    logdet = jnp.where(below_range, logdet_below, logdet_mid)
    logdet = jnp.where(above_range, logdet_above, logdet)
    return x_out, logdet


@partial(custom_inverse, inv_argnum=0)
def piecewise_affine_spline(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    y, _ = _piecewise_affine_spline_fwd(
        x,
        x_pos,
        y_pos,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    return y


def inv_piecewise_affine_spline(
    y: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    return _piecewise_affine_spline_inv(
        y,
        x_pos,
        y_pos,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


piecewise_affine_spline.definv_and_logdet(inv_piecewise_affine_spline)


__all__ = ["piecewise_affine_spline", "inv_piecewise_affine_spline"]
