from functools import partial
from typing import Optional, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.stats.bijective._spline_common import linear_tail, merge3, select_bin


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

    below_range, above_range, left, right = select_bin(x, x_pos, (x_pos, y_pos))

    x_l, y_l = left[0], left[1]
    x_r, y_r = right[0], right[1]

    dx = x_r - x_l
    s = (y_r - y_l) / dx

    y_mid = y_l + s * (x - x_l)
    logdet_mid = jnp.log(jnp.abs(s))

    # ----- lower tail (unbounded slope 1.0) -----
    y_below, logdet_below = linear_tail(
        x, x_pos[0], y_pos[0], x_min, y_min, 1.0, side="below"
    )

    # ----- upper tail (unbounded slope 1.0) -----
    y_above, logdet_above = linear_tail(
        x, x_pos[-1], y_pos[-1], x_max, y_max, 1.0, side="above"
    )

    y = merge3(below_range, above_range, y_below, y_mid, y_above)
    logdet = merge3(below_range, above_range, logdet_below, logdet_mid, logdet_above)
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

    below_range, above_range, left, right = select_bin(y, y_pos, (x_pos, y_pos))

    x_l, y_l = left[0], left[1]
    x_r, y_r = right[0], right[1]

    dx = x_r - x_l
    s = (y_r - y_l) / dx

    x_mid = x_l + (y - y_l) / s
    logdet_mid = -jnp.log(jnp.abs(s))

    # lower tail (unbounded slope 1.0)
    x_below, logdet_below = linear_tail(
        y, y_pos[0], x_pos[0], y_min, x_min, 1.0, side="below"
    )

    # upper tail (unbounded slope 1.0)
    x_above, logdet_above = linear_tail(
        y, y_pos[-1], x_pos[-1], y_max, x_max, 1.0, side="above"
    )

    x_out = merge3(below_range, above_range, x_below, x_mid, x_above)
    logdet = merge3(below_range, above_range, logdet_below, logdet_mid, logdet_above)
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


def piecewise_affine_spline_and_logdet(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    """Forward direction returning ``(y, logdet)``."""
    return _piecewise_affine_spline_fwd(
        x,
        x_pos,
        y_pos,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


piecewise_affine_spline.defvalue_and_logdet(piecewise_affine_spline_and_logdet)


__all__ = [
    "inv_piecewise_affine_spline",
    "piecewise_affine_spline",
    "piecewise_affine_spline_and_logdet",
]
