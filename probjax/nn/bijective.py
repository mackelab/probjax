from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.utils.solver import root_scalar


def _normalize_knot_slopes(
    unnormalized_knot_slopes: Array, min_knot_slope: float
) -> Array:
    """Make knot slopes be no less than `min_knot_slope`."""
    # The offset is such that the normalized knot slope will be equal to 1
    # whenever the unnormalized knot slope is equal to 0.
    if min_knot_slope >= 1.0:
        raise ValueError(
            f"The minimum knot slope must be less than 1; got {min_knot_slope}."
        )
    min_knot_slope = jnp.array(min_knot_slope, dtype=unnormalized_knot_slopes.dtype)
    offset = jnp.log(jnp.exp(1.0 - min_knot_slope) - 1.0)
    return jax.nn.softplus(unnormalized_knot_slopes + offset) + min_knot_slope


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
    params: ArrayLike,
    x: ArrayLike,
    range_min_x: float = -1.0,
    range_max_x: float = 1.0,
    range_min_y: float = -1.0,
    range_max_y: float = 1.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)

    # Normalize slopes and bins
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        ((range_max_x - min_bin_size) - range_min_x) + (range_min_x)
    )
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (range_max_y - min_bin_size) - range_min_y
    ) + range_min_y

    if not bounded:
        # Real support
        y, logdet = _rational_quadratic_spline_fwd(x, x_pos, y_pos, knot_slopes)
    else:
        # Bounded support on range
        y, logdet = _rational_quadratic_spline_fwd(
            x,
            x_pos,
            y_pos,
            knot_slopes,
            range_min_x,
            range_max_x,
            range_min_y,
            range_max_y,
        )
    return y, logdet


@partial(custom_inverse, inv_argnum=1)
def rational_quadratic_spline(
    params: ArrayLike,
    x: ArrayLike,
    x_min: float = -10.0,
    x_max: float = 10.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)
    # Normalize slopes and bins
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)


    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        (x_max - min_bin_size) - x_min
    ) + x_min
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (y_max - min_bin_size) - y_min
    ) + y_min

    if not bounded:
        # Real support
        y, _ = _rational_quadratic_spline_fwd(x, x_pos, y_pos, knot_slopes)
    else:
        # Bounded support on range
        y, _ = _rational_quadratic_spline_fwd(
            x,
            x_pos,
            y_pos,
            knot_slopes,
            x_min,
            x_max,
            y_min,
            y_max,
        )
    return y


def inv_rational_quadratic_spline(
    params: ArrayLike,
    x: ArrayLike,
    x_min=-10.0,
    x_max=10.0,
    y_min=-10.0,
    y_max=10.0,
    min_bin_size=1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)
    # Normalize slopes and bins
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)
    # Stay within numerical limits
    x_pos = jnp.clip(x_pos, -6, 6)
    y_pos = jnp.clip(y_pos, -6, 6)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        (x_max - min_bin_size) - x_min
    ) + x_min
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (y_max - min_bin_size) - y_min
    ) + y_min

    if not bounded:
        y, log_det = _rational_quadratic_spline_inv(x, x_pos, y_pos, knot_slopes)
    else:
        y, log_det = _rational_quadratic_spline_inv(
            x,
            x_pos,
            y_pos,
            knot_slopes,
            x_min,
            x_max,
            y_min,
            y_max,
        )
    return y, jnp.squeeze(log_det)


rational_quadratic_spline.definv_and_logdet(inv_rational_quadratic_spline)



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
    first_bin_mask = jnp.concatenate(
        [jnp.array([True]), jnp.zeros(len(correct_bin) - 1, dtype=bool)]
    )
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
    logdet_mid = (
        jnp.log(jnp.abs(bin_slope))
        + jnp.log(alpha)
        - 2.0 * jnp.log(denom)
    )

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
    first_bin_mask = jnp.concatenate(
        [jnp.array([True]), jnp.zeros(len(correct_bin) - 1, dtype=bool)]
    )
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
        logdet_below = jnp.where(
            jnp.isnan(slope_bl), logdet_below, logdet_bounded
        )

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
        logdet_above = jnp.where(
            jnp.isnan(slope_ab), logdet_above, logdet_bounded
        )

    x_out = jnp.where(below_range, x_below, x_mid)
    x_out = jnp.where(above_range, x_above, x_out)

    logdet = jnp.where(below_range, logdet_below, logdet_mid)
    logdet = jnp.where(above_range, logdet_above, logdet)

    return x_out, logdet


@partial(custom_inverse, inv_argnum=1)
def rational_linear_spline(params, x, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0, min_bin_size=1e-4, min_knot_slope=1e-4, bounded=False):
    """Rational linear spline transformation.
    
    Args:
        params: Parameters containing x_pos, y_pos, and knot_slopes
        x: Input values to transform
        x_min, x_max, y_min, y_max: Optional bounds for the transformation
        min_bin_size: Minimum size of each bin
        min_knot_slope: Minimum slope at knot points
        bounded: Whether to use bounded interpolation
        
    Returns:
        Transformed values
    """
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)
    
    # Normalize slopes and bins
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)
    
    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        (x_max - min_bin_size) - x_min
    ) + x_min
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (y_max - min_bin_size) - y_min
    ) + y_min

    if not bounded:
        # Real support
        y, _ = _rational_linear_spline_fwd(x, x_pos, y_pos, knot_slopes)
    else:
        # Bounded support on range
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
    params: ArrayLike,
    x: ArrayLike,
    x_min=-10.0,
    x_max=10.0,
    y_min=-10.0,
    y_max=10.0,
    min_bin_size=1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    """Inverse of the rational linear spline.
    
    Args:
        params: Parameters containing x_pos, y_pos, and knot_slopes
        x: Input values to invert
        x_min, x_max, y_min, y_max: Optional bounds for the transformation
        min_bin_size: Minimum size of each bin
        min_knot_slope: Minimum slope at knot points
        bounded: Whether to use bounded interpolation
        
    Returns:
        Tuple of (inverse values, log determinant)
    """
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)
    
    # Normalize slopes and bins
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)
    
    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        (x_max - min_bin_size) - x_min
    ) + x_min
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (y_max - min_bin_size) - y_min
    ) + y_min

    if not bounded:
        y, log_det = _rational_linear_spline_inv(x, x_pos, y_pos, knot_slopes)
    else:
        y, log_det = _rational_linear_spline_inv(
            x,
            x_pos,
            y_pos,
            knot_slopes,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
    return y, jnp.squeeze(log_det)


# Set up the inverse for rational_linear_spline
rational_linear_spline.definv_and_logdet(inv_rational_linear_spline)



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
    first_bin = jnp.concatenate([jnp.array([True]), jnp.zeros(len(correct_bin) - 1, dtype=bool)])
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
    # slope_below = (y_pos[0] - (y_min if y_min is not None else y_pos[0])) / (
    #     (x_pos[0] - (x_min if x_min is not None else x_pos[0])) + 1e-6
    # ) if (x_min is not None and y_min is not None) else (y_pos[0] - y_pos[0]) / (x_pos[0] - x_pos[0] + 1e-6)
    # # Fallback to unbounded slope = knot slope at left if bounds not provided
    # slope_below = jnp.where(
    #     jnp.isfinite(slope_below) & (x_min is not None) & (y_min is not None),
    #     slope_below,
    #     (y_pos[0] - y_pos[0] + (x - x)) + 0.0  # dummy to keep dtype
    # )
    # Simpler: match your previous tail policy precisely
    slope_below = 1.
    y_below = (x - x_pos[0]) * slope_below + y_pos[0]
    logdet_below = jnp.log(jnp.abs(slope_below))
    if x_min is not None and y_min is not None:
        denom = x_pos[0] - x_min
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_bl = (y_pos[0] - y_min) / denom
        y_lin = y_min + slope_bl * (x - x_min)
        y_lin = jnp.where(x <= x_min, y_min, y_lin)
        y_below = jnp.where(jnp.isnan(slope_bl), y_below, y_lin)
        logdet_below = jnp.where(jnp.isnan(slope_bl), logdet_below, jnp.log(jnp.abs(slope_bl)))

    # ----- upper tail -----
    slope_above = 1.
    y_above = (x - x_pos[-1]) * slope_above + y_pos[-1]
    logdet_above = jnp.log(jnp.abs(slope_above))
    if x_max is not None and y_max is not None:
        denom = x_max - x_pos[-1]
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_ab = (y_max - y_pos[-1]) / denom
        y_lin = y_pos[-1] + slope_ab * (x - x_pos[-1])
        y_lin = jnp.where(x >= x_max, y_max, y_lin)
        y_above = jnp.where(jnp.isnan(slope_ab), y_above, y_lin)
        logdet_above = jnp.where(jnp.isnan(slope_ab), logdet_above, jnp.log(jnp.abs(slope_ab)))

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
    first_bin = jnp.concatenate([jnp.array([True]), jnp.zeros(len(correct_bin) - 1, dtype=bool)])
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
    slope_below = 1.0 #/ s[0]
    x_below = x_pos[0] + slope_below * (y - y_pos[0])
    logdet_below = jnp.log(jnp.abs(slope_below))
    if y_min is not None and x_min is not None:
        denom = y_pos[0] - y_min
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_bl = (x_pos[0] - x_min) / denom
        x_lin = x_min + slope_bl * (y - y_min)
        x_lin = jnp.where(y <= y_min, x_min, x_lin)
        x_below = jnp.where(jnp.isnan(slope_bl), x_below, x_lin)
        logdet_below = jnp.where(jnp.isnan(slope_bl), logdet_below, jnp.log(jnp.abs(slope_bl)))

    # upper tail
    slope_above = 1.0 #/ s[-1]
    x_above = x_pos[-1] + slope_above * (y - y_pos[-1])
    logdet_above = jnp.log(jnp.abs(slope_above))
    if y_max is not None and x_max is not None:
        denom = y_max - y_pos[-1]
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_ab = (x_max - x_pos[-1]) / denom
        x_lin = x_pos[-1] + slope_ab * (y - y_pos[-1])
        x_lin = jnp.where(y >= y_max, x_max, x_lin)
        x_above = jnp.where(jnp.isnan(slope_ab), x_above, x_lin)
        logdet_above = jnp.where(jnp.isnan(slope_ab), logdet_above, jnp.log(jnp.abs(slope_ab)))

    x_out = jnp.where(below_range, x_below, x_mid)
    x_out = jnp.where(above_range, x_above, x_out)

    logdet = jnp.where(below_range, logdet_below, logdet_mid)
    logdet = jnp.where(above_range, logdet_above, logdet)
    return x_out, logdet


@partial(custom_inverse, inv_argnum=1)
def piecewise_affine_spline(
    params: ArrayLike,
    x: ArrayLike,
    x_min: float = -10.0,
    x_max: float = 10.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
    min_bin_size: float = 1e-4,
    bounded: bool = False,
):
    x_pos, y_pos = jnp.split(params, 2, axis=-1)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        ((x_max - min_bin_size) - x_min)
    ) + x_min + 1e-6
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (y_max - min_bin_size) - y_min
    ) + y_min + 1e-6

    if not bounded:
        y, _ = _piecewise_affine_spline_fwd(x, x_pos, y_pos)
    else:
        y, _ = _piecewise_affine_spline_fwd(
            x, x_pos, y_pos, x_min, x_max, y_min, y_max
        )
    return y

def piecewise_affine_spline_inv(params, y, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0, min_bin_size=1e-4, bounded=False):
    x_pos, y_pos = jnp.split(params, 2, axis=-1)

    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        ((x_max - min_bin_size) - x_min)
    ) + x_min + 1e-6
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (y_max - min_bin_size) - y_min
    ) + y_min + 1e-6

    if not bounded:
        x, logdet = _piecewise_affine_spline_inv(y, x_pos, y_pos)
    else:
        x, logdet = _piecewise_affine_spline_inv(
            y, x_pos, y_pos, x_min, x_max, y_min, y_max
        )
    return x, jnp.squeeze(logdet)


piecewise_affine_spline.definv_and_logdet(piecewise_affine_spline_inv)


def _hermite_basis(z):
    """Cubic Hermite basis and derivatives on z in [0,1]."""
    z2 = z * z
    z3 = z2 * z
    # basis
    h00 = 2.0 * z3 - 3.0 * z2 + 1.0
    h10 = z3 - 2.0 * z2 + z
    h01 = -2.0 * z3 + 3.0 * z2
    h11 = z3 - z2
    # derivatives w.r.t z
    dh00 = 6.0 * z2 - 6.0 * z
    dh10 = 3.0 * z2 - 4.0 * z + 1.0
    dh01 = -dh00  # -6 z^2 + 6 z
    dh11 = 3.0 * z2 - 2.0 * z
    return (h00, h10, h01, h11), (dh00, dh10, dh01, dh11)


def _fc_monotone_normalize(bin_slope, m0, m1):
    """
    Fritsch–Carlson per-bin normalization to ensure monotone cubic.
    Works for strictly increasing bins (bin_slope > 0).
    Returns normalized (a, b) where a = m0/bin_slope, b = m1/bin_slope, with constraints.
    """
    eps = 1e-12
    s = bin_slope
    # If s <= 0 (shouldn't happen for increasing y_pos), make it flat
    a = jnp.where(s > 0.0, m0 / (s + eps), 0.0)
    b = jnp.where(s > 0.0, m1 / (s + eps), 0.0)

    # Non-negativity
    a = jnp.maximum(a, 0.0)
    b = jnp.maximum(b, 0.0)

    # If the bin is flat, force a=b=0
    a = jnp.where(s <= 0.0, 0.0, a)
    b = jnp.where(s <= 0.0, 0.0, b)

    # Fritsch–Carlson scaling: ensure a^2 + b^2 <= 9
    sumsq = a * a + b * b
    scale = jnp.where(sumsq > 9.0, 3.0 / jnp.sqrt(sumsq + eps), 1.0)
    a = a * scale
    b = b * scale
    return a, b


def _monotone_hermite_cubic_spline_fwd(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Forward transform using a monotone cubic Hermite spline with FC normalization.
    Returns (y, log|dy/dx|).
    """
    x = jnp.asarray(x)
    x_pos = jnp.asarray(x_pos)
    y_pos = jnp.asarray(y_pos)
    knot_slopes = jnp.asarray(knot_slopes)

    below_range = x <= x_pos[0]
    above_range = x >= x_pos[-1]

    correct_bin = jnp.logical_and(x >= x_pos[:-1], x < x_pos[1:])
    any_bin = jnp.any(correct_bin)
    first_bin = jnp.concatenate([jnp.array([True]), jnp.zeros(len(correct_bin) - 1, dtype=bool)])
    correct_bin = jnp.where(any_bin, correct_bin, first_bin)

    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_l, y_l, m_l = left[0], left[1], left[2]
    x_r, y_r, m_r = right[0], right[1], right[2]

    dx = x_r - x_l
    dy = y_r - y_l
    s = dy / dx  # bin slope (dy/dx), > 0 for monotone

    z = (x - x_l) / dx
    z = jnp.clip(z, 0.0, 1.0)

    # Normalize endpoint slopes to guarantee monotonicity
    a, b = _fc_monotone_normalize(s, m_l, m_r)

    (h00, h10, h01, h11), (dh00, dh10, dh01, dh11) = _hermite_basis(z)

    # y = y_l + dy * [ h01 + a*h10 + b*h11 ]   (since h00 + h01 = 1, and dx*m = (m/s)*dy)
    R = h01 + a * h10 + b * h11
    y_mid = y_l + dy * R

    # dy/dx = s * R'(z),  where R'(z) = dh01 + a*dh10 + b*dh11
    Rp = dh01 + a * dh10 + b * dh11
    # Safety: Rp should be positive; clamp tiny values to preserve gradients
    eps = 1e-12
    logdet_mid = jnp.log(jnp.abs(s)) + jnp.log(jnp.maximum(Rp, eps))

    # -------- tails: identical policy to your RQ/RL code --------
    slope_below = knot_slopes[0]
    y_below = (x - x_pos[0]) * slope_below + y_pos[0]
    logdet_below = jnp.log(jnp.abs(slope_below))
    if x_min is not None and y_min is not None:
        denom = x_pos[0] - x_min
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_bl = (y_pos[0] - y_min) / denom
        y_lin = y_min + slope_bl * (x - x_min)
        y_lin = jnp.where(x <= x_min, y_min, y_lin)
        y_below = jnp.where(jnp.isnan(slope_bl), y_below, y_lin)
        logdet_below = jnp.where(jnp.isnan(slope_bl), logdet_below, jnp.log(jnp.abs(slope_bl)))

    slope_above = knot_slopes[-1]
    y_above = (x - x_pos[-1]) * slope_above + y_pos[-1]
    logdet_above = jnp.log(jnp.abs(slope_above))
    if x_max is not None and y_max is not None:
        denom = x_max - x_pos[-1]
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_ab = (y_max - y_pos[-1]) / denom
        y_lin = y_pos[-1] + slope_ab * (x - x_pos[-1])
        y_lin = jnp.where(x >= x_max, y_max, y_lin)
        y_above = jnp.where(jnp.isnan(slope_ab), y_above, y_lin)
        logdet_above = jnp.where(jnp.isnan(slope_ab), logdet_above, jnp.log(jnp.abs(slope_ab)))

    y = jnp.where(below_range, y_below, y_mid)
    y = jnp.where(above_range, y_above, y)

    logdet = jnp.where(below_range, logdet_below, logdet_mid)
    logdet = jnp.where(above_range, logdet_above, logdet)
    return y, logdet


def _monotone_hermite_cubic_spline_inv(
    y: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    newton_iters: int = 8,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Inverse via clamped Newton on normalized coordinate z in [0,1].
    The cubic in each bin is strictly monotone after FC normalization,
    so a unique root exists and converges quickly.
    Returns (x, log|dx/dy|).
    """
    y = jnp.asarray(y)
    x_pos = jnp.asarray(x_pos)
    y_pos = jnp.asarray(y_pos)
    knot_slopes = jnp.asarray(knot_slopes)

    below_range = y <= y_pos[0]
    above_range = y >= y_pos[-1]

    correct_bin = jnp.logical_and(y >= y_pos[:-1], y < y_pos[1:])
    any_bin = jnp.any(correct_bin)
    first_bin = jnp.concatenate([jnp.array([True]), jnp.zeros(len(correct_bin) - 1, dtype=bool)])
    correct_bin = jnp.where(any_bin, correct_bin, first_bin)

    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_l, y_l, m_l = left[0], left[1], left[2]
    x_r, y_r, m_r = right[0], right[1], right[2]

    dx = x_r - x_l
    dy = y_r - y_l
    s = dy / dx  # > 0

    # Target normalized position w in [0,1]
    w = (y - y_l) / (dy + 1e-12)
    w = jnp.clip(w, 0.0, 1.0)

    # FC-normalized endpoint slope ratios
    a, b = _fc_monotone_normalize(s, m_l, m_r)

    def H_and_dH(z):
        (h00, h10, h01, h11), (dh00, dh10, dh01, dh11) = _hermite_basis(z)
        R = h01 + a * h10 + b * h11
        Rp = dh01 + a * dh10 + b * dh11
        return R, Rp

    # Clamped Newton with bisection fallback inside [0,1]
    def newton(z0):
        def body(carry, _):
            z, lo, hi = carry
            R, Rp = H_and_dH(z)
            f = R - w
            # Newton step
            z_nt = z - f / (Rp + 1e-12)
            # If step leaves bracket, bisect
            out = (z_nt < lo) | (z_nt > hi)
            z_nt = jnp.where(out, 0.5 * (lo + hi), z_nt)
            # Update bracket using sign of f at new z
            R_nt, _ = H_and_dH(z_nt)
            f_nt = R_nt - w
            lo = jnp.where(f_nt < 0.0, z_nt, lo)
            hi = jnp.where(f_nt > 0.0, z_nt, hi)
            return (z_nt, lo, hi), None

        z_init = jnp.clip(w, 0.0, 1.0)  # linear guess
        (z_final, _, _), _ = jax.lax.scan(lambda c, i: body(c, i),
                                          (z_init, jnp.array([0.0]), jnp.array([1.0])),
                                          jnp.arange(newton_iters))
        return z_final

    z = newton(w)
    x_mid = x_l + z * dx

    # log|dx/dy| = -log|dy/dx| = -[ log s + log Rp(z) ]
    _, Rp = H_and_dH(z)
    logdet_mid = -(jnp.log(jnp.abs(s)) + jnp.log(jnp.maximum(Rp, 1e-12)))

    # -------- tails (same as forward) --------
    slope_below = 1.0 / knot_slopes[0]
    x_below = x_pos[0] + slope_below * (y - y_pos[0])
    logdet_below = jnp.log(jnp.abs(slope_below))
    if y_min is not None and x_min is not None:
        denom = y_pos[0] - y_min
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_bl = (x_pos[0] - x_min) / denom
        x_lin = x_min + slope_bl * (y - y_min)
        x_lin = jnp.where(y <= y_min, x_min, x_lin)
        x_below = jnp.where(jnp.isnan(slope_bl), x_below, x_lin)
        logdet_below = jnp.where(jnp.isnan(slope_bl), logdet_below, jnp.log(jnp.abs(slope_bl)))

    slope_above = 1.0 / knot_slopes[-1]
    x_above = x_pos[-1] + slope_above * (y - y_pos[-1])
    logdet_above = jnp.log(jnp.abs(slope_above))
    if y_max is not None and x_max is not None:
        denom = y_max - y_pos[-1]
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_ab = (x_max - x_pos[-1]) / denom
        x_lin = x_pos[-1] + slope_ab * (y - y_pos[-1])
        x_lin = jnp.where(y >= y_max, x_max, x_lin)
        x_above = jnp.where(jnp.isnan(slope_ab), x_above, x_lin)
        logdet_above = jnp.where(jnp.isnan(slope_ab), logdet_above, jnp.log(jnp.abs(slope_ab)))

    x_out = jnp.where(below_range, x_below, x_mid)
    x_out = jnp.where(above_range, x_above, x_out)

    logdet = jnp.where(below_range, logdet_below, logdet_mid)
    logdet = jnp.where(above_range, logdet_above, logdet)
    return x_out, logdet


@partial(custom_inverse, inv_argnum=1)
def monotone_hermite_cubic_spline(params, x, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0, min_bin_size=1e-4, min_knot_slope=1e-4, bounded=False):
    """Monotone Hermite cubic spline transformation.
    
    Args:
        params: Parameters containing x_pos, y_pos, and knot_slopes
        x: Input values to transform
        x_min, x_max, y_min, y_max: Optional bounds for the transformation
        min_bin_size: Minimum size of each bin
        min_knot_slope: Minimum slope at knot points
        bounded: Whether to use bounded interpolation
        
    Returns:
        Transformed values
    """
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)
    
    # Normalize slopes and bins
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)
    
    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        (x_max - min_bin_size) - x_min
    ) + x_min
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (y_max - min_bin_size) - y_min
    ) + y_min

    if not bounded:
        # Real support
        y, _ = _monotone_hermite_cubic_spline_fwd(x, x_pos, y_pos, knot_slopes)
    else:
        # Bounded support on range
        y, _ = _monotone_hermite_cubic_spline_fwd(
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


def inv_monotone_hermite_cubic_spline(
    params: ArrayLike,
    x: ArrayLike,
    x_min=-10.0,
    x_max=10.0,
    y_min=-10.0,
    y_max=10.0,
    min_bin_size=1e-4,
    min_knot_slope: float = 1e-4,
    bounded: bool = False,
):
    """Inverse of the monotone Hermite cubic spline.
    
    Args:
        params: Parameters containing x_pos, y_pos, and knot_slopes
        x: Input values to invert
        x_min, x_max, y_min, y_max: Optional bounds for the transformation
        min_bin_size: Minimum size of each bin
        min_knot_slope: Minimum slope at knot points
        bounded: Whether to use bounded interpolation
        
    Returns:
        Tuple of (inverse values, log determinant)
    """
    x_pos, y_pos, knot_slopes = jnp.split(params, 3, axis=-1)
    
    # Normalize slopes and bins
    knot_slopes = _normalize_knot_slopes(knot_slopes, min_knot_slope)
    
    x_pos = (jnp.cumsum(jax.nn.softmax(x_pos), -1) + min_bin_size) * (
        (x_max - min_bin_size) - x_min
    ) + x_min
    y_pos = (jnp.cumsum(jax.nn.softmax(y_pos), -1) + min_bin_size) * (
        (y_max - min_bin_size) - y_min
    ) + y_min

    if not bounded:
        y, log_det = _monotone_hermite_cubic_spline_inv(x, x_pos, y_pos, knot_slopes)
    else:
        y, log_det = _monotone_hermite_cubic_spline_inv(
            x,
            x_pos,
            y_pos,
            knot_slopes,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
    return y, jnp.squeeze(log_det)


# Set up the inverse for monotone_hermite_cubic_spline
monotone_hermite_cubic_spline.definv_and_logdet(inv_monotone_hermite_cubic_spline)


@partial(custom_inverse, inv_argnum=1)
def learnable_mixture_cdf(
    params: ArrayLike,
    y: ArrayLike,
    min_value=-10.0,
    max_value=10.0,
    **kwargs,
):
    def f(x):
        return _inv_learnable_mixture_cdf(params, x) - y

    x = root_scalar(
        f,
        bracket=(min_value * jnp.ones_like(y), max_value * jnp.ones_like(y)),
        **kwargs,
    )

    return x


def _inv_learnable_mixture_cdf(
    params: ArrayLike,
    x: ArrayLike,
    **kwargs,
):
    x = jnp.asarray(x)
    loc, scale = jnp.split(params, 2, axis=-1)
    scale = jnp.exp(scale)
    x_ks = (x[..., None] - loc) / scale
    cdf = jnp.mean(jax.nn.sigmoid(x_ks), -1)
    out = jax.scipy.stats.norm.ppf(cdf)
    return out


def _inv_and_logdet_learnable_mixture_cdf(params, x, **kwargs):
    _f = jax.vmap(jax.value_and_grad(_inv_learnable_mixture_cdf, argnums=1))
    value, grad = _f(params, x)
    return value, jnp.log(jnp.abs(grad))


learnable_mixture_cdf.definv(_inv_learnable_mixture_cdf)
learnable_mixture_cdf.definv_and_logdet(_inv_and_logdet_learnable_mixture_cdf)


def affine_bijector(
    params: ArrayLike, x: ArrayLike, min_scale=5e-1, max_scale=5.0, **kwargs
):
    loc, scale = jnp.split(params, 2, axis=-1)
    scale = jax.nn.sigmoid(scale) * (max_scale - min_scale) + min_scale

    return loc + scale * x


def additive_bijector(params: ArrayLike, x: ArrayLike, **kwargs):
    return x + params
