"""Piecewise-exponential envelopes of a univariate log-concave density.

The tangents of a concave log density at a set of points lie above it, so their
lower hull is an upper bound and the chords between neighbouring points are a
lower bound (the squeeze). The envelope ``exp(upper)`` is a mixture of
truncated exponentials that can be sampled exactly. :class:`ars` uses it as an
exact rejection sampler and :class:`~probjax.inference.arms` as a Metropolis
proposal; both refine it with the points they evaluate.

Arrays have a fixed length ``max_points`` so refinement compiles; slots past
``num_points`` are padding (``x = inf``) and carry no mass.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array, ArrayLike

# Below this |slope| a segment is treated as flat, avoiding 0/0 in its mass.
_FLAT = 1e-6


class Envelope(NamedTuple):
    """Tangent points of a log-concave density and the hull they define.

    Attributes:
        x: tangent points, sorted, padded with ``inf``.
        h: log density at ``x``.
        grad: its derivative at ``x``.
        z: segment boundaries; segment ``j`` covers ``[z[j], z[j + 1]]``.
        log_mass: log of the envelope's mass on each segment.
        num_points: number of real tangent points.
        lower, upper: the support ``[lower, upper]``.
    """

    x: Array
    h: Array
    grad: Array
    z: Array
    log_mass: Array
    num_points: Array
    lower: Array
    upper: Array


def _segment_log_mass(h, grad, x, start, end):
    """``log ∫_start^end exp(h + grad (t - x)) dt``, stable for infinite ends."""
    width = end - start
    safe_grad = jnp.where(jnp.abs(grad) < _FLAT, 1.0, grad)
    # Factor out the larger endpoint so the remaining log1p term is <= 0.
    rising = h + safe_grad * (end - x) + jnp.log1p(-jnp.exp(-safe_grad * width))
    falling = h + safe_grad * (start - x) + jnp.log1p(-jnp.exp(safe_grad * width))
    curved = jnp.where(safe_grad > 0, rising, falling) - jnp.log(jnp.abs(safe_grad))
    flat = h + jnp.log(width)
    return jnp.where(jnp.abs(grad) < _FLAT, flat, curved)


def _hull(x, h, grad, num_points, lower, upper) -> Envelope:
    max_points = x.shape[0]
    valid = jnp.arange(max_points) < num_points
    # Where neighbouring tangents cross; padding gives NaN, replaced below.
    crossing = (h[1:] - h[:-1] - x[1:] * grad[1:] + x[:-1] * grad[:-1]) / (
        grad[:-1] - grad[1:]
    )
    inner = jnp.where(valid[1:], crossing, upper)
    z = jnp.clip(jnp.concatenate([lower[None], inner, upper[None]]), lower, upper)
    log_mass = _segment_log_mass(h, grad, x, z[:-1], z[1:])
    log_mass = jnp.where(valid, log_mass, -jnp.inf)
    return Envelope(x, h, grad, z, log_mass, num_points, lower, upper)


def _outwards(value_and_grad, point, step, direction, max_doublings=60):
    """Move ``point`` along ``direction`` until the log density decreases."""

    def climbing(carry):
        point, _, doublings = carry
        slope = value_and_grad(point)[1]
        return (direction * slope >= 0) & (doublings < max_doublings)

    def further(carry):
        point, step, doublings = carry
        return point + direction * step, 2.0 * step, doublings + 1

    point, _, _ = jax.lax.while_loop(
        climbing, further, (point, jnp.maximum(step, 1.0), 0)
    )
    return point


def envelope(
    logdensity_fn: Callable,
    points: ArrayLike,
    lower: ArrayLike = -jnp.inf,
    upper: ArrayLike = jnp.inf,
    max_points: int = 50,
) -> Envelope:
    """Build the envelope from tangents at ``points`` (at least two).

    On an unbounded side the outermost tangent must slope towards the mode, or
    the envelope has infinite mass; the outermost points are moved outwards,
    with doubling steps, until it does.
    """
    points = jnp.sort(jnp.asarray(points, dtype=float))
    num_points = points.shape[0]
    if not 2 <= num_points <= max_points:
        raise ValueError("need between two and max_points initial points")
    value_and_grad = jax.value_and_grad(logdensity_fn)
    spread = points[-1] - points[0]
    # A finite bound closes that side, so only unbounded sides are searched.
    right = _outwards(value_and_grad, points[-1], spread, direction=1.0)
    left = _outwards(value_and_grad, points[0], spread, direction=-1.0)
    points = points.at[-1].set(jnp.where(jnp.isinf(upper), right, points[-1]))
    points = points.at[0].set(jnp.where(jnp.isinf(lower), left, points[0]))
    h, grad = jax.vmap(value_and_grad)(points)
    padding = jnp.full(max_points - num_points, jnp.inf)
    return _hull(
        jnp.concatenate([points, padding]),
        jnp.concatenate([h, padding]),
        jnp.concatenate([grad, padding]),
        jnp.asarray(num_points),
        jnp.asarray(lower, dtype=float),
        jnp.asarray(upper, dtype=float),
    )


def refine(env: Envelope, x: Array, h: Array, grad: Array) -> Envelope:
    """Add a tangent at ``x``; a full envelope is returned unchanged."""

    def add(env: Envelope) -> Envelope:
        # The padding slot is +inf, so sorting puts the new point in place.
        xs = env.x.at[env.num_points].set(x)
        order = jnp.argsort(xs)
        return _hull(
            xs[order],
            env.h.at[env.num_points].set(h)[order],
            env.grad.at[env.num_points].set(grad)[order],
            env.num_points + 1,
            env.lower,
            env.upper,
        )

    return jax.lax.cond(env.num_points < env.x.shape[0], add, lambda env: env, env)


def _segment(env: Envelope, x: Array) -> Array:
    index = jnp.searchsorted(env.z, x, side="right") - 1
    return jnp.clip(index, 0, env.num_points - 1)


def upper_hull(env: Envelope, x: Array) -> Array:
    """The envelope's (unnormalised) log density at ``x``."""
    j = _segment(env, x)
    return env.h[j] + env.grad[j] * (x - env.x[j])


def squeeze(env: Envelope, x: Array) -> Array:
    """The chord lower bound at ``x``; ``-inf`` outside the tangent points."""
    i = jnp.searchsorted(env.x, x, side="right") - 1
    inside = (i >= 0) & (i < env.num_points - 1)
    i = jnp.clip(i, 0, env.x.shape[0] - 2)
    left, right = env.x[i], env.x[i + 1]
    chord = ((right - x) * env.h[i] + (x - left) * env.h[i + 1]) / (right - left)
    return jnp.where(inside, chord, -jnp.inf)


def envelope_logpdf(env: Envelope, x: Array) -> Array:
    """Log density of the normalised envelope, i.e. of :func:`sample`."""
    return upper_hull(env, x) - jax.scipy.special.logsumexp(env.log_mass)


def sample(env: Envelope, rng) -> Array:
    """One exact draw from the normalised envelope."""
    segment_key, position_key = jax.random.split(rng)
    j = jax.random.categorical(segment_key, env.log_mass)
    start, end, slope = env.z[j], env.z[j + 1], env.grad[j]
    u = jax.random.uniform(position_key)
    # Inverse CDF of exp(slope * t) on [start, end], anchored at the heavier
    # end so an infinite other end contributes exp(-inf) = 0.
    safe = jnp.where(jnp.abs(slope) < _FLAT, 1.0, slope)
    width = end - start
    rising = end + jnp.log(u + (1.0 - u) * jnp.exp(-safe * width)) / safe
    falling = start + jnp.log((1.0 - u) + u * jnp.exp(safe * width)) / safe
    curved = jnp.where(safe > 0, rising, falling)
    return jnp.where(jnp.abs(slope) < _FLAT, start + u * width, curved)
