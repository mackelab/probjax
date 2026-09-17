"""Shared scaffolding for monotone piecewise spline bijectors.

All spline families in this package (rational-quadratic, rational-linear,
piecewise-affine, monotone Hermite cubic) share the same structure:

- bin selection by dot-product mask over stacked knot parameters,
- one-sided linear tails with optional bounded clamping,
- a three-region (below / inside / above) merge.

Each family keeps only its inside-the-bin ``*_mid`` math; everything else
delegates here so tail, clamp and bin-edge policy cannot drift between
families. Unbounded tail slopes are preserved per family
(e.g. edge knot-slopes, or 1.0 for piecewise-affine).
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def select_bin(v: ArrayLike, mask_pos: ArrayLike, stack):
    """Select the bin containing ``v`` and return edge values.

    Args:
        v: Scalar value to locate.
        mask_pos: Bin boundaries the selection mask is computed on, shape
            ``[num_bins + 1]`` (``x_pos`` forward, ``y_pos`` inverse).
        stack: Knot arrays of shape ``[num_bins + 1]`` gathered at the
            selected bin edges (e.g. ``(x_pos, y_pos, knot_slopes)``).

    Returns:
        ``(below_range, above_range, left, right)`` where ``left``/``right``
        gather the stacked rows at the selected bin edges. Values outside all
        bins default to the first bin (avoids NaNs).
    """
    below_range = v <= mask_pos[0]
    above_range = v >= mask_pos[-1]

    correct_bin = jnp.logical_and(v >= mask_pos[:-1], v < mask_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate([
        jnp.array([True]),
        jnp.zeros(len(correct_bin) - 1, dtype=bool),
    ])
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)

    stacked = jnp.stack(tuple(stack), axis=1)
    left = jnp.sum(correct_bin[:, None] * stacked[:-1], axis=0)
    right = jnp.sum(correct_bin[:, None] * stacked[1:], axis=0)
    return below_range, above_range, left, right


def linear_tail(
    v: ArrayLike,
    anchor_v: ArrayLike,
    anchor_o: ArrayLike,
    bound_v: ArrayLike | None,
    bound_o: ArrayLike | None,
    unbounded_slope: ArrayLike,
    side: str,
    unbounded_logdet: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """One-sided linear tail with optional bounded clamping.

    Args:
        v: Scalar input on the tail side.
        anchor_v: Tail anchor on the input axis (e.g. ``x_pos[0]``).
        anchor_o: Tail anchor on the output axis (e.g. ``y_pos[0]``).
        bound_v: Optional clamp point on the input axis (e.g. ``x_min``).
        bound_o: Optional clamp point on the output axis (e.g. ``y_min``).
        unbounded_slope: Slope used when no bounds are given.
        side: ``"below"`` evaluates the bounded branch from the bound point
            and clamps at ``v <= bound_v``; ``"above"`` evaluates from the
            anchor point and clamps at ``v >= bound_v`` (matching the
            historical per-spline evaluation order bitwise).
        unbounded_logdet: Optional precomputed logdet of the unbounded
            branch; defaults to ``log|unbounded_slope|``.

    Returns:
        ``(value, logdet)`` of the tail branch.
    """
    v_unbounded = anchor_o + unbounded_slope * (v - anchor_v)
    if unbounded_logdet is None:
        unbounded_logdet = jnp.log(jnp.abs(unbounded_slope))
    logdet_unbounded = unbounded_logdet

    if bound_v is not None and bound_o is not None:
        denom = anchor_v - bound_v
        denom = jnp.where(denom == 0.0, 1e-6, denom)
        slope_bounded = (anchor_o - bound_o) / denom
        if side == "below":
            v_bounded = bound_o + slope_bounded * (v - bound_v)
            v_bounded = jnp.where(v <= bound_v, bound_o, v_bounded)
        else:
            v_bounded = anchor_o + slope_bounded * (v - anchor_v)
            v_bounded = jnp.where(v >= bound_v, bound_o, v_bounded)
        logdet_bounded = jnp.log(jnp.abs(slope_bounded))
        v_tail = jnp.where(jnp.isnan(slope_bounded), v_unbounded, v_bounded)
        logdet_tail = jnp.where(
            jnp.isnan(slope_bounded), logdet_unbounded, logdet_bounded
        )
    else:
        v_tail = v_unbounded
        logdet_tail = logdet_unbounded
    return v_tail, logdet_tail


def merge3(
    below_range: ArrayLike,
    above_range: ArrayLike,
    v_below: ArrayLike,
    v_mid: ArrayLike,
    v_above: ArrayLike,
):
    """Merge below/inside/above region values."""
    v = jnp.where(below_range, v_below, v_mid)
    return jnp.where(above_range, v_above, v)
