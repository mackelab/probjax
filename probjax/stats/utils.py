"""Shared utilities for statistical distribution implementations."""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp

from probjax.utils.typing import Array, ArrayLike

__all__ = [
    "flatten_samples",
    "normalize_sample_weights",
    "mean_and_var_1d",
    "row_mean_and_var",
    "row_mean_and_cov",
]


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
    total = jnp.where(total > 0, total, jnp.asarray(n_samples, dtype=dtype))
    return normalized / total


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
