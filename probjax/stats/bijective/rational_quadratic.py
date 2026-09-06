from functools import partial
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.stats.bijective._spline_common import linear_tail, merge3, select_bin


def _rational_quadratic_spline_fwd(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """Applies a rational-quadratic spline to a scalar, with optional bounded
    linear interpolation outside [x_pos[0], x_pos[-1]].

    Args:
      x: a scalar (0-dimensional array). The scalar `x` can be any real number; it
        will be transformed by the spline if it's in the closed interval
        `[x_pos[0], x_pos[-1]]`.
      x_pos: array of shape [num_bins + 1], the bin boundaries on the x axis.
      y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
      knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
      x_min, x_max, y_min, y_max: optional bounds; if specified, the linear
        interpolation outside the spline domain is bounded within these intervals.

    Returns:
      A tuple of two scalars: (y, logdet),
        where `y` is the spline output,
        and `logdet` is the log of the absolute first derivative at `x`.
    """
    # Identify the regions outside and inside the main spline range, and the
    # bin in which x lies (defaults to the first bin to avoid NaNs).
    below_range, above_range, params_bin_left, params_bin_right = select_bin(
        x, x_pos, (x_pos, y_pos, knot_slopes)
    )

    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])

    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width

    # Normalized position of x inside the bin
    z = (x - x_pos_bin[0]) / bin_width
    z = jnp.clip(z, 0.0, 1.0)  # avoid floating-point issues

    # Rational-quadratic calculation
    sq_z = z * z
    z1mz = z - sq_z  # z(1-z)
    sq_1mz = (1.0 - z) ** 2
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2.0 * bin_slope
    numerator = bin_height * (bin_slope * sq_z + knot_slopes_bin[0] * z1mz)
    denominator = bin_slope + slopes_term * z1mz
    y_unclamped = y_pos_bin[0] + numerator / denominator

    # Log absolute derivative of the spline (inside the bin)
    logdet_unclamped = (
        2.0 * jnp.log(bin_slope)
        + jnp.log(
            knot_slopes_bin[1] * sq_z
            + 2.0 * bin_slope * z1mz
            + knot_slopes_bin[0] * sq_1mz
        )
        - 2.0 * jnp.log(denominator)
    )

    # ------------------------------
    # Below-range: bounded or unbounded linear tail.
    # ------------------------------
    y_below, logdet_below = linear_tail(
        x, x_pos[0], y_pos[0], x_min, y_min, knot_slopes[0], side="below"
    )

    # ------------------------------
    # Above-range: bounded or unbounded linear tail.
    # ------------------------------
    y_above, logdet_above = linear_tail(
        x, x_pos[-1], y_pos[-1], x_max, y_max, knot_slopes[-1], side="above"
    )

    # ------------------------------
    # Merge the three regions:
    #   below_range, inside, above_range
    # ------------------------------
    y = merge3(below_range, above_range, y_below, y_unclamped, y_above)
    logdet = merge3(
        below_range, above_range, logdet_below, logdet_unclamped, logdet_above
    )

    return y, logdet


def _safe_quadratic_root(a: Array, b: Array, c: Array) -> Array:
    """Implement a numerically stable version of the quadratic formula."""
    # This is not a general solution to the quadratic equation, as it assumes
    # b ** 2 - 4. * a * c is known a priori to be positive (and which of the two
    # roots is to be used, see https://arxiv.org/abs/1906.04032).
    # There are two sources of instability:
    # (a) When b ** 2 - 4. * a * c -> 0, sqrt gives NaNs in gradient.
    # We clip sqrt_diff to have the smallest float number.
    sqrt_diff = b**2 - 4.0 * a * c
    safe_sqrt = jnp.sqrt(jnp.clip(sqrt_diff, jnp.finfo(sqrt_diff.dtype).tiny))
    # If sqrt_diff is non-positive, we set sqrt to 0. as it should be positive.
    safe_sqrt = jnp.where(sqrt_diff > 0.0, safe_sqrt, 0.0)
    # (b) When 4. * a * c -> 0. We use the more stable quadratic solution
    # depending on the sign of b.
    # See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf (eq 7 and 8).
    # Solution when b >= 0
    numerator_1 = 2.0 * c
    denominator_1 = -b - safe_sqrt
    # Solution when b < 0
    numerator_2 = -b + safe_sqrt
    denominator_2 = 2 * a
    # Choose the numerically stable solution.
    numerator = jnp.where(b >= 0, numerator_1, numerator_2)
    denominator = jnp.where(b >= 0, denominator_1, denominator_2)
    # (c) Both branches degenerate to 0 / 0 when b == 0 and the discriminant
    # vanishes, which forces c == 0 as well -- and then z = 0 is the root we
    # want. Left unguarded this returns NaN and poisons the whole inverse.
    degenerate = denominator == 0.0
    safe_denominator = jnp.where(degenerate, 1.0, denominator)
    return jnp.where(degenerate, 0.0, numerator / safe_denominator)


def _rational_quadratic_spline_inv(
    y: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Applies the inverse of a rational-quadratic spline to a scalar, with optional
    bounded linear interpolation outside [y_pos[0], y_pos[-1]].

    Args:
      y: a scalar (0-dimensional array). `y` can be any real number; it will
        be transformed by the spline if it's in the closed interval
        [y_pos[0], y_pos[-1]].
      x_pos: array of shape [num_bins + 1], the bin boundaries on the x axis.
      y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
      knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
      x_min, x_max, y_min, y_max: optional bounds; if specified, the linear
        interpolation outside the spline domain is bounded within these intervals.

    Returns:
      A tuple of two scalars: (x, logdet), where:
        x is the inverse spline output,
        logdet is the log of the absolute first derivative of the inverse at `y`.
    """
    # --------------------------------------------------
    # Identify whether y is below, above, or inside the spline range, and
    # the correct bin for y (defaults to the first bin to avoid NaNs).
    # --------------------------------------------------
    below_range, above_range, params_bin_left, params_bin_right = select_bin(
        y, y_pos, (x_pos, y_pos, knot_slopes)
    )

    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])

    # --------------------------------------------------
    # Inverse spline logic (the 'middle' region)
    # --------------------------------------------------
    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width

    # Normalized position w in [0,1]
    w = (y - y_pos_bin[0]) / bin_height
    w = jnp.clip(w, 0.0, 1.0)

    # Solve the quadratic equation a*z^2 + b*z + c = 0 for z
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2.0 * bin_slope
    c = -bin_slope * w
    b = knot_slopes_bin[0] - slopes_term * w
    a = bin_slope - b
    z = _safe_quadratic_root(a, b, c)
    z = jnp.clip(z, 0.0, 1.0)

    # Map z back to x
    x_unclamped = bin_width * z + x_pos_bin[0]

    # Log determinant of the inverse inside the bin
    sq_z = z * z
    z1mz = z - sq_z  # z*(1-z)
    sq_1mz = (1.0 - z) ** 2
    denominator = bin_slope + slopes_term * z1mz
    logdet_unclamped = (
        -2.0 * jnp.log(bin_slope)
        - jnp.log(
            knot_slopes_bin[1] * sq_z
            + 2.0 * bin_slope * z1mz
            + knot_slopes_bin[0] * sq_1mz
        )
        + 2.0 * jnp.log(denominator)
    )

    # --------------------------------------------------
    # Below-range: bounded or unbounded linear tail.
    # Unbounded: x = x_pos[0] + (y - y_pos[0]) / knot_slopes[0].
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Above-range: bounded or unbounded linear tail.
    # Unbounded: x = x_pos[-1] + (y - y_pos[-1]) / knot_slopes[-1].
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Piecewise merge (below, inside, above)
    # --------------------------------------------------
    x = merge3(below_range, above_range, x_below, x_unclamped, x_above)
    logdet = merge3(
        below_range, above_range, logdet_below, logdet_unclamped, logdet_above
    )

    return x, logdet


@partial(custom_inverse, inv_argnum=0)
def rational_quadratic_spline(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    y, _ = _rational_quadratic_spline_fwd(
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


def inv_rational_quadratic_spline(
    y: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    return _rational_quadratic_spline_inv(
        y,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


rational_quadratic_spline.definv_and_logdet(inv_rational_quadratic_spline)


def rational_quadratic_spline_and_logdet(
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
    return _rational_quadratic_spline_fwd(
        x,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


rational_quadratic_spline.defvalue_and_logdet(rational_quadratic_spline_and_logdet)


__all__ = [
    "inv_rational_quadratic_spline",
    "rational_quadratic_spline",
    "rational_quadratic_spline_and_logdet",
]
