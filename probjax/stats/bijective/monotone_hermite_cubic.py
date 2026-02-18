from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.core.custom_primitives.custom_inverse import custom_inverse


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
    first_bin = jnp.concatenate([
        jnp.array([True]),
        jnp.zeros(len(correct_bin) - 1, dtype=bool),
    ])
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
        logdet_below = jnp.where(
            jnp.isnan(slope_bl), logdet_below, jnp.log(jnp.abs(slope_bl))
        )

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
        logdet_above = jnp.where(
            jnp.isnan(slope_ab), logdet_above, jnp.log(jnp.abs(slope_ab))
        )

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
    first_bin = jnp.concatenate([
        jnp.array([True]),
        jnp.zeros(len(correct_bin) - 1, dtype=bool),
    ])
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
        (z_final, _, _), _ = jax.lax.scan(
            lambda c, i: body(c, i),
            (z_init, jnp.array(0.0), jnp.array(1.0)),
            jnp.arange(newton_iters),
        )
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
        logdet_below = jnp.where(
            jnp.isnan(slope_bl), logdet_below, jnp.log(jnp.abs(slope_bl))
        )

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
        logdet_above = jnp.where(
            jnp.isnan(slope_ab), logdet_above, jnp.log(jnp.abs(slope_ab))
        )

    x_out = jnp.where(below_range, x_below, x_mid)
    x_out = jnp.where(above_range, x_above, x_out)

    logdet = jnp.where(below_range, logdet_below, logdet_mid)
    logdet = jnp.where(above_range, logdet_above, logdet)
    return x_out, logdet


@partial(custom_inverse, inv_argnum=0)
def monotone_hermite_cubic_spline(
    x: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
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
    y: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    knot_slopes: ArrayLike,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    newton_iters: int = 8,
):
    return _monotone_hermite_cubic_spline_inv(
        y,
        x_pos,
        y_pos,
        knot_slopes,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        newton_iters=newton_iters,
    )


monotone_hermite_cubic_spline.definv_and_logdet(inv_monotone_hermite_cubic_spline)


__all__ = [
    "monotone_hermite_cubic_spline",
    "inv_monotone_hermite_cubic_spline",
]
