from functools import partial
from typing import Optional, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.stats.bijective._spline_common import linear_tail, merge3, select_bin


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
    below_range, above_range, p_left, p_right = select_bin(
        x, x_pos, (x_pos, y_pos, knot_slopes)
    )

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
    y_below, logdet_below = linear_tail(
        x, x_pos[0], y_pos[0], x_min, y_min, knot_slopes[0], side="below"
    )

    # ------------------   upper tail  --------------------------------------
    y_above, logdet_above = linear_tail(
        x, x_pos[-1], y_pos[-1], x_max, y_max, knot_slopes[-1], side="above"
    )

    # ------------------   piecewise merge  ----------------------------------
    y = merge3(below_range, above_range, y_below, y_mid, y_above)
    logdet = merge3(below_range, above_range, logdet_below, logdet_mid, logdet_above)

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

    below_range, above_range, p_left, p_right = select_bin(
        y, y_pos, (x_pos, y_pos, knot_slopes)
    )

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
    x_below, logdet_below = linear_tail(
        y,
        y_pos[0],
        x_pos[0],
        y_min,
        x_min,
        1.0 / knot_slopes[0],
        side="below",
        unbounded_logdet=-jnp.log(knot_slopes[0]),
    )

    x_above, logdet_above = linear_tail(
        y,
        y_pos[-1],
        x_pos[-1],
        y_max,
        x_max,
        1.0 / knot_slopes[-1],
        side="above",
        unbounded_logdet=-jnp.log(knot_slopes[-1]),
    )

    x_out = merge3(below_range, above_range, x_below, x_mid, x_above)
    logdet = merge3(below_range, above_range, logdet_below, logdet_mid, logdet_above)

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


def rational_linear_spline_and_logdet(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    """Forward direction returning ``(y, logdet)``."""
    return _rational_linear_spline_fwd(
        x,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


rational_linear_spline.defvalue_and_logdet(rational_linear_spline_and_logdet)


__all__ = [
    "inv_rational_linear_spline",
    "rational_linear_spline",
    "rational_linear_spline_and_logdet",
]
