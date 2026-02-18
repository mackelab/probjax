from functools import partial
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from probjax.core.custom_primitives.custom_inverse import custom_inverse


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
    # Identify the regions outside and inside the main spline range
    below_range = x <= x_pos[0]
    above_range = x >= x_pos[-1]

    # Identify the correct bin in which x lies
    correct_bin = jnp.logical_and(x >= x_pos[:-1], x < x_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    # If x does not fall into any bin, default to the first bin (avoids NaNs)
    first_bin = jnp.concatenate([
        jnp.array([True]),
        jnp.zeros(len(correct_bin) - 1, dtype=bool),
    ])
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)

    # Collect (x_pos, y_pos, slopes) into a single array so we can dot with the mask
    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

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
    # Below-range: bounded or unbounded?
    # ------------------------------
    # Default unbounded slope below
    slope_below_unbounded = knot_slopes[0]
    y_below_unbounded = (x - x_pos[0]) * slope_below_unbounded + y_pos[0]
    logdet_below_unbounded = jnp.log(slope_below_unbounded)

    # If x_min, y_min are specified, we do a bounded linear mapping.
    # That means for x <= x_min, we clamp to y_min,
    # otherwise linearly interpolate up to x_pos[0], y_pos[0].
    if x_min is not None and y_min is not None:
        # Slope from (x_min -> y_min) to (x_pos[0] -> y_pos[0])
        denom_below = x_pos[0] - x_min
        # Avoid division by zero if x_min == x_pos[0]
        denom_below = jnp.where(denom_below == 0.0, 1e-6, denom_below)
        slope_below_bounded = (y_pos[0] - y_min) / denom_below
        y_below_bounded = y_min + slope_below_bounded * (x - x_min)
        logdet_below_bounded = jnp.log(jnp.abs(slope_below_bounded))

        # Also clamp if x < x_min
        y_below_bounded = jnp.where(x <= x_min, y_min, y_below_bounded)

        # Choose which version (bounded or unbounded) to apply:
        y_below = jnp.where(
            jnp.isnan(slope_below_bounded), y_below_unbounded, y_below_bounded
        )
        logdet_below = jnp.where(
            jnp.isnan(slope_below_bounded), logdet_below_unbounded, logdet_below_bounded
        )
    else:
        # Fall back to the unbounded version
        y_below = y_below_unbounded
        logdet_below = logdet_below_unbounded

    # ------------------------------
    # Above-range: bounded or unbounded?
    # ------------------------------
    # Default unbounded slope above
    slope_above_unbounded = knot_slopes[-1]
    y_above_unbounded = (x - x_pos[-1]) * slope_above_unbounded + y_pos[-1]
    logdet_above_unbounded = jnp.log(slope_above_unbounded)

    # If x_max, y_max are specified, we do a bounded linear mapping.
    # That means for x >= x_max, we clamp to y_max,
    # otherwise linearly interpolate from x_pos[-1], y_pos[-1].
    if x_max is not None and y_max is not None:
        # Slope from (x_pos[-1] -> y_pos[-1]) to (x_max -> y_max)
        denom_above = x_max - x_pos[-1]
        denom_above = jnp.where(denom_above == 0.0, 1e-6, denom_above)
        slope_above_bounded = (y_max - y_pos[-1]) / denom_above
        y_above_bounded = y_pos[-1] + slope_above_bounded * (x - x_pos[-1])
        logdet_above_bounded = jnp.log(jnp.abs(slope_above_bounded))

        # Also clamp if x >= x_max
        y_above_bounded = jnp.where(x >= x_max, y_max, y_above_bounded)

        y_above = jnp.where(
            jnp.isnan(slope_above_bounded), y_above_unbounded, y_above_bounded
        )
        logdet_above = jnp.where(
            jnp.isnan(slope_above_bounded), logdet_above_unbounded, logdet_above_bounded
        )
    else:
        # Fall back to the unbounded version
        y_above = y_above_unbounded
        logdet_above = logdet_above_unbounded

    # ------------------------------
    # Merge the three regions:
    #   below_range, inside, above_range
    # ------------------------------
    y = jnp.where(below_range, y_below, y_unclamped)
    y = jnp.where(above_range, y_above, y)

    logdet = jnp.where(below_range, logdet_below, logdet_unclamped)
    logdet = jnp.where(above_range, logdet_above, logdet)

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
    return numerator / denominator


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
    # Identify whether y is below, above, or inside the spline range
    # --------------------------------------------------
    below_range = y <= y_pos[0]
    above_range = y >= y_pos[-1]

    # --------------------------------------------------
    # Identify the correct bin for y if it's in [y_pos[0], y_pos[-1]]
    # --------------------------------------------------
    correct_bin = jnp.logical_and(y >= y_pos[:-1], y < y_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    # If y does not fall into any bin, default to the first bin (avoids NaNs)
    first_bin = jnp.concatenate([
        jnp.array([True]),
        jnp.zeros(len(correct_bin) - 1, dtype=bool),
    ])
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)

    # Dot-product mask to extract the bin's (x_pos, y_pos, slopes)
    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

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
    # Below-range: bounded or unbounded?
    # --------------------------------------------------
    # Unbounded slope for y < y_pos[0]: x = x_pos[0] + (y - y_pos[0]) / knot_slopes[0]
    slope_below_unbounded = 1.0 / knot_slopes[0]
    x_below_unbounded = x_pos[0] + slope_below_unbounded * (y - y_pos[0])
    logdet_below_unbounded = -jnp.log(
        knot_slopes[0]
    )  # = jnp.log(slope_below_unbounded)

    # If (y_min, x_min) are provided, do a bounded linear mapping from
    # (y_min -> x_min) to (y_pos[0] -> x_pos[0]), and clamp at y <= y_min.
    if y_min is not None and x_min is not None:
        denom_below = y_pos[0] - y_min
        denom_below = jnp.where(denom_below == 0.0, 1e-6, denom_below)
        slope_below_bounded = (x_pos[0] - x_min) / denom_below
        x_below_bounded = x_min + slope_below_bounded * (y - y_min)
        # Clamp x if y <= y_min
        x_below_bounded = jnp.where(y <= y_min, x_min, x_below_bounded)

        logdet_below_bounded = jnp.log(jnp.abs(slope_below_bounded))

        # Decide which version to use (bounded vs unbounded)
        x_below = jnp.where(
            jnp.isnan(slope_below_bounded), x_below_unbounded, x_below_bounded
        )
        logdet_below = jnp.where(
            jnp.isnan(slope_below_bounded), logdet_below_unbounded, logdet_below_bounded
        )
    else:
        # Fallback to unbounded approach
        x_below = x_below_unbounded
        logdet_below = logdet_below_unbounded

    # --------------------------------------------------
    # Above-range: bounded or unbounded?
    # --------------------------------------------------
    # Unbounded slope for y > y_pos[-1]: x = x_pos[-1] + (y - y_pos[-1]) / knot_slopes[-1]
    slope_above_unbounded = 1.0 / knot_slopes[-1]
    x_above_unbounded = x_pos[-1] + slope_above_unbounded * (y - y_pos[-1])
    logdet_above_unbounded = -jnp.log(knot_slopes[-1])

    # If (y_max, x_max) are provided, do a bounded linear mapping from
    # (y_pos[-1] -> x_pos[-1]) to (y_max -> x_max), and clamp at y >= y_max.
    if y_max is not None and x_max is not None:
        denom_above = y_max - y_pos[-1]
        denom_above = jnp.where(denom_above == 0.0, 1e-12, denom_above)
        slope_above_bounded = (x_max - x_pos[-1]) / denom_above
        x_above_bounded = x_pos[-1] + slope_above_bounded * (y - y_pos[-1])
        # Clamp x if y >= y_max
        x_above_bounded = jnp.where(y >= y_max, x_max, x_above_bounded)

        logdet_above_bounded = jnp.log(jnp.abs(slope_above_bounded))

        # Decide which version to use (bounded vs unbounded)
        x_above = jnp.where(
            jnp.isnan(slope_above_bounded), x_above_unbounded, x_above_bounded
        )
        logdet_above = jnp.where(
            jnp.isnan(slope_above_bounded), logdet_above_unbounded, logdet_above_bounded
        )
    else:
        # Fallback to unbounded approach
        x_above = x_above_unbounded
        logdet_above = logdet_above_unbounded

    # --------------------------------------------------
    # Piecewise merge (below, inside, above)
    # --------------------------------------------------
    x = jnp.where(below_range, x_below, x_unclamped)
    x = jnp.where(above_range, x_above, x)

    logdet = jnp.where(below_range, logdet_below, logdet_unclamped)
    logdet = jnp.where(above_range, logdet_above, logdet)

    return x, logdet


def rational_quadratic_spline_and_logdets(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Tuple[ArrayLike, ArrayLike]:
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


__all__ = [
    "rational_quadratic_spline",
    "inv_rational_quadratic_spline",
    "rational_quadratic_spline_and_logdets",
]
