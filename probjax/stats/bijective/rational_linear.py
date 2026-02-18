from functools import partial
from typing import Optional, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.core.custom_primitives.custom_inverse import custom_inverse


def _rational_linear_spline_fwd(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    *,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Forward transform of a rational-linear spline (degree-1 numerator /
    degree-1 denominator).

    Monotonicity is guaranteed provided `knot_slopes > 0`.
    Outside the interior domain we use exactly the same linear tails / clamping
    policy as the rational-quadratic version for drop-in compatibility.
    """
    x = jnp.asarray(x)
    x_pos = jnp.asarray(x_pos)
    y_pos = jnp.asarray(y_pos)
    knot_slopes = jnp.asarray(knot_slopes)

    # ------------------   region identification  ---------------------------
    below_range = x <= x_pos[0]
    above_range = x >= x_pos[-1]

    correct_bin = jnp.logical_and(x >= x_pos[:-1], x < x_pos[1:])
    any_bin = jnp.any(correct_bin)
    first_bin_mask = jnp.concatenate([
        jnp.array([True]),
        jnp.zeros(len(correct_bin) - 1, dtype=bool),
    ])
    correct_bin = jnp.where(any_bin, correct_bin, first_bin_mask)

    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    p_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    p_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_l, x_r = p_left[0], p_right[0]
    y_l, y_r = p_left[1], p_right[1]
    m_l = p_left[2]

    dx = x_r - x_l
    dy = y_r - y_l
    bin_slope = dy / dx

    # parameter α := m_l / bin_slope  (positive)
    alpha = m_l / bin_slope

    z = (x - x_l) / dx
    z = jnp.clip(z, 0.0, 1.0)

    # rational-linear mapping   y = y_l + dy * (α z) / (1 + (α-1) z)
    denom = 1.0 + (alpha - 1.0) * z
    y_mid = y_l + dy * (alpha * z) / denom

    # log|dy/dx| = log(bin_slope) + log(alpha) − 2·log(denom)
    logdet_mid = jnp.log(jnp.abs(bin_slope)) + jnp.log(alpha) - 2.0 * jnp.log(denom)

    # ------------------   lower tail  --------------------------------------
    slope_below = knot_slopes[0]
    y_below = (x - x_pos[0]) * slope_below + y_pos[0]
    logdet_below = jnp.log(slope_below)

    if x_min is not None and y_min is not None:
        denom_bl = x_pos[0] - x_min
        denom_bl = jnp.where(denom_bl == 0.0, 1e-6, denom_bl)
        slope_bl = (y_pos[0] - y_min) / denom_bl
        y_bounded = y_min + slope_bl * (x - x_min)
        y_bounded = jnp.where(x <= x_min, y_min, y_bounded)
        logdet_bounded = jnp.log(jnp.abs(slope_bl))
        y_below = jnp.where(jnp.isnan(slope_bl), y_below, y_bounded)
        logdet_below = jnp.where(jnp.isnan(slope_bl), logdet_below, logdet_bounded)

    # ------------------   upper tail  --------------------------------------
    slope_above = knot_slopes[-1]
    y_above = (x - x_pos[-1]) * slope_above + y_pos[-1]
    logdet_above = jnp.log(slope_above)

    if x_max is not None and y_max is not None:
        denom_ab = x_max - x_pos[-1]
        denom_ab = jnp.where(denom_ab == 0.0, 1e-6, denom_ab)
        slope_ab = (y_max - y_pos[-1]) / denom_ab
        y_bounded = y_pos[-1] + slope_ab * (x - x_pos[-1])
        y_bounded = jnp.where(x >= x_max, y_max, y_bounded)
        logdet_bounded = jnp.log(jnp.abs(slope_ab))
        y_above = jnp.where(jnp.isnan(slope_ab), y_above, y_bounded)
        logdet_above = jnp.where(jnp.isnan(slope_ab), logdet_above, logdet_bounded)

    # ------------------   piecewise merge  ----------------------------------
    y = jnp.where(below_range, y_below, y_mid)
    y = jnp.where(above_range, y_above, y)

    logdet = jnp.where(below_range, logdet_below, logdet_mid)
    logdet = jnp.where(above_range, logdet_above, logdet)

    return y, logdet


# ---------------------------------------------------------------------------
#  (1/1) Rational-linear spline:  inverse  ----------------------------------
# ---------------------------------------------------------------------------


def _rational_linear_spline_inv(
    y: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    *,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Inverse transform (y → x) for the degree-1/1 rational spline above.
    The analytic inverse is cheap: one division, no square-roots.
    """
    y = jnp.asarray(y)
    x_pos = jnp.asarray(x_pos)
    y_pos = jnp.asarray(y_pos)
    knot_slopes = jnp.asarray(knot_slopes)

    below_range = y <= y_pos[0]
    above_range = y >= y_pos[-1]

    correct_bin = jnp.logical_and(y >= y_pos[:-1], y < y_pos[1:])
    any_bin = jnp.any(correct_bin)
    first_bin_mask = jnp.concatenate([
        jnp.array([True]),
        jnp.zeros(len(correct_bin) - 1, dtype=bool),
    ])
    correct_bin = jnp.where(any_bin, correct_bin, first_bin_mask)

    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    p_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    p_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_l, x_r = p_left[0], p_right[0]
    y_l, y_r = p_left[1], p_right[1]
    m_l = p_left[2]

    dx = x_r - x_l
    dy = y_r - y_l
    bin_slope = dy / dx

    alpha = m_l / bin_slope

    # normalised w in (0,1)
    w = (y - y_l) / dy
    w = jnp.clip(w, 0.0, 1.0)

    denom = alpha - w * (alpha - 1.0)
    denom = jnp.clip(denom, 1e-12)  # numerical safety
    z = w / denom
    x_mid = x_l + z * dx

    # log|dx/dy| = -log|dy/dx|
    logdet_mid = -(
        jnp.log(jnp.abs(bin_slope))
        + jnp.log(alpha)
        - 2.0 * jnp.log(1.0 + (alpha - 1.0) * z)
    )

    # ---------------- tails: same policy as forward -------------------------
    slope_below = 1.0 / knot_slopes[0]
    x_below = x_pos[0] + slope_below * (y - y_pos[0])
    logdet_below = -jnp.log(knot_slopes[0])

    if y_min is not None and x_min is not None:
        denom_bl = y_pos[0] - y_min
        denom_bl = jnp.where(denom_bl == 0.0, 1e-6, denom_bl)
        slope_bl = (x_pos[0] - x_min) / denom_bl
        x_bounded = x_min + slope_bl * (y - y_min)
        x_bounded = jnp.where(y <= y_min, x_min, x_bounded)
        logdet_bounded = jnp.log(jnp.abs(slope_bl))
        x_below = jnp.where(jnp.isnan(slope_bl), x_below, x_bounded)
        logdet_below = jnp.where(jnp.isnan(slope_bl), logdet_below, logdet_bounded)

    slope_above = 1.0 / knot_slopes[-1]
    x_above = x_pos[-1] + slope_above * (y - y_pos[-1])
    logdet_above = -jnp.log(knot_slopes[-1])

    if y_max is not None and x_max is not None:
        denom_ab = y_max - y_pos[-1]
        denom_ab = jnp.where(denom_ab == 0.0, 1e-6, denom_ab)
        slope_ab = (x_max - x_pos[-1]) / denom_ab
        x_bounded = x_pos[-1] + slope_ab * (y - y_pos[-1])
        x_bounded = jnp.where(y >= y_max, x_max, x_bounded)
        logdet_bounded = jnp.log(jnp.abs(slope_ab))
        x_above = jnp.where(jnp.isnan(slope_ab), x_above, x_bounded)
        logdet_above = jnp.where(jnp.isnan(slope_ab), logdet_above, logdet_bounded)

    x_out = jnp.where(below_range, x_below, x_mid)
    x_out = jnp.where(above_range, x_above, x_out)

    logdet = jnp.where(below_range, logdet_below, logdet_mid)
    logdet = jnp.where(above_range, logdet_above, logdet)

    return x_out, logdet


@partial(custom_inverse, inv_argnum=0)
def rational_linear_spline(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    y, _ = _rational_linear_spline_fwd(
        x,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    return y


def inv_rational_linear_spline(
    y: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    return _rational_linear_spline_inv(
        y,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


rational_linear_spline.definv_and_logdet(inv_rational_linear_spline)


__all__ = ["rational_linear_spline", "inv_rational_linear_spline"]
