"""Shared utilities for statistical distribution implementations."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp

from probjax.utils.jaxutils import where
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = [
    "clip_prob",
    "circular_mean_and_resultant",
    "flatten_by_event_shape",
    "flatten_samples",
    "loc_scale_sample",
    "normalize_sample_weights",
    "mean_and_var_1d",
    "row_mean_and_var",
    "row_mean_and_cov",
    "unflatten_by_event_shape",
    "weighted_mean",
]


def flatten_by_event_shape(
    x: ArrayLike, event_shape: Tuple[int, ...]
) -> Tuple[Array, Tuple[int, ...]]:
    """Flatten the leading dims of ``x`` into one axis, keeping the event dims.

    Returns the flattened array and the leading shape it was flattened from.
    """
    x_arr = jnp.asarray(x)
    event_shape = tuple(event_shape)
    event_ndim = len(event_shape)

    if event_ndim:
        if x_arr.ndim < event_ndim:
            raise ValueError(
                "Input has fewer dimensions than the distribution event shape."
            )
        if tuple(x_arr.shape[-event_ndim:]) != event_shape:
            raise ValueError(
                "Trailing dimensions of the input must match the distribution "
                "event shape."
            )
        leading_shape = tuple(x_arr.shape[:-event_ndim])
    else:
        leading_shape = tuple(x_arr.shape)

    return jnp.reshape(x_arr, (-1,) + event_shape), leading_shape


def unflatten_by_event_shape(
    x_flat: ArrayLike, leading_shape: Tuple[int, ...], event_shape: Tuple[int, ...]
) -> Array:
    """Restore a flattened array to its leading and event dimensions."""
    return jnp.reshape(x_flat, tuple(leading_shape) + tuple(event_shape))


def circular_mean_and_resultant(
    data: ArrayLike, weights: Optional[Array] = None
) -> Tuple[Array, Array]:
    """Circular mean and mean resultant length of ``data``.

    Both the von Mises and the wrapped Cauchy fits need these; the
    ``atan2(sin, cos)`` mean and its ``sqrt(c^2 + s^2)`` resultant.
    ``weights`` must already be normalized (or ``None`` for equal weights).
    """
    data_arr = jnp.asarray(data)
    if weights is None:
        s = jnp.mean(jnp.sin(data_arr))
        c = jnp.mean(jnp.cos(data_arr))
    else:
        s = jnp.sum(weights * jnp.sin(data_arr))
        c = jnp.sum(weights * jnp.cos(data_arr))
    return jnp.arctan2(s, c), jnp.sqrt(c**2 + s**2)


def clip_prob(q: ArrayLike) -> Array:
    """Clip probabilities to ``[tiny, 1 - tiny]`` for the input dtype."""
    q_arr = jnp.asarray(q)
    eps = jnp.finfo(q_arr.dtype).tiny
    return jnp.clip(q_arr, eps, 1.0 - eps)


def loc_scale_sample(
    rng: RngKey,
    sampler: Callable,
    *params: Any,
    shape: Tuple[int, ...] = (),
    loc: ArrayLike = 0.0,
    scale: ArrayLike = 1.0,
) -> Array:
    """Sample ``sampler(rng, *params) * scale + loc`` with broadcast shape.

    Broadcasts ``params``/``loc``/``scale`` to a common event shape and draws
    ``shape + event_shape`` samples from ``sampler``, which must accept a
    ``shape`` keyword (e.g. ``jax.random.normal``).
    """
    loc = jnp.asarray(loc)
    scale = jnp.asarray(scale)
    param_arrays = [jnp.asarray(p) for p in params]
    event_shape = jnp.broadcast_shapes(
        *[p.shape for p in param_arrays], loc.shape, scale.shape
    )
    return sampler(rng, *param_arrays, shape=shape + event_shape) * scale + loc


def flatten_samples(data: ArrayLike) -> Array:
    """Convert observations to a one-dimensional sample vector."""
    return jnp.reshape(jnp.asarray(data), (-1,))


def normalize_sample_weights(
    weights: Optional[ArrayLike],
    *,
    n_samples: int,
    dtype: Any,
    mismatch_message: str = "weights must have the same length as data",
    column: bool = False,
) -> Optional[Array]:
    """Validate, clip, and normalize non-negative sample weights."""
    if weights is None:
        return None

    shape = (n_samples, 1) if column else (n_samples,)
    normalized = jnp.asarray(weights, dtype=dtype).reshape(shape)
    if normalized.shape[0] != n_samples:
        raise ValueError(mismatch_message)

    normalized = jnp.clip(normalized, 0)
    total = jnp.sum(normalized)
    total = where(total > 0, total, jnp.asarray(n_samples, dtype=dtype))
    return normalized / total


def _require_unweighted_fit(weights: Optional[ArrayLike], dist_name: str) -> None:
    """Raise unless ``weights`` is None (weighted fit not implemented)."""
    if weights is not None:
        raise NotImplementedError(
            f"Weighted fitting is not implemented for the {dist_name} distribution."
        )


def mean_and_var_1d(
    data: ArrayLike, weights: Optional[Array] = None
) -> tuple[Array, Array]:
    """Compute (optionally weighted) mean and variance for one-dimensional data."""
    data_arr = flatten_samples(data)
    if weights is None:
        return jnp.mean(data_arr), jnp.var(data_arr)

    mean = jnp.sum(weights * data_arr)
    var = jnp.sum(weights * (data_arr - mean) ** 2)
    return mean, var


def weighted_mean(data: ArrayLike, weights: Optional[ArrayLike] = None) -> Array:
    """Flatten ``data`` and return its (optionally weighted) mean."""
    data_arr = flatten_samples(data)
    if weights is None:
        return jnp.mean(data_arr)
    weights_arr = normalize_sample_weights(
        weights, n_samples=data_arr.shape[0], dtype=data_arr.dtype
    )
    assert weights_arr is not None
    return jnp.sum(weights_arr * data_arr)


def row_mean_and_var(
    data: ArrayLike, weights: Optional[Array] = None
) -> tuple[Array, Array]:
    """Compute per-feature (optionally weighted) mean and variance across rows."""
    data_arr = jnp.asarray(data)
    if weights is None:
        return jnp.mean(data_arr, axis=0), jnp.var(data_arr, axis=0)

    mean = jnp.sum(weights * data_arr, axis=0)
    var = jnp.sum(weights * (data_arr - mean) ** 2, axis=0)
    return mean, var


def row_mean_and_cov(
    data: ArrayLike,
    weights: Optional[Array] = None,
    *,
    unbiased_unweighted: bool = True,
) -> tuple[Array, Array]:
    """Compute per-feature mean and covariance matrix across rows."""
    data_arr = jnp.asarray(data)
    n_samples = int(data_arr.shape[0])

    if weights is None:
        mean = jnp.mean(data_arr, axis=0)
        centered = data_arr - mean
        if unbiased_unweighted:
            denom = jnp.maximum(n_samples - 1, 1)
        else:
            denom = jnp.maximum(n_samples, 1)
        cov = centered.T @ centered / denom
        return mean, cov

    mean = jnp.sum(weights * data_arr, axis=0)
    centered = data_arr - mean
    cov = (centered * weights).T @ centered
    return mean, cov
