from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module
from flax.nnx.nn import dtypes, initializers
from flax.nnx.nn.normalization import (
    BatchNorm,
    GroupNorm,
    LayerNorm,
    RMSNorm,
    SpectralNorm,
    WeightNorm,
    _canonicalize_axes,
)
from flax.typing import Array, Axes, Dtype, Initializer, PromoteDtypeFn


def _lp_norm(
    x: Array,
    p: float,
    axes: tuple[int, ...],
    epsilon: float,
) -> Array:
    """Compute the Lp norm along *axes*, with epsilon for stability.

    Returns the norm with the reduced axes kept (keepdims=True) so it
    broadcasts against the original array.
    """
    abs_x = jnp.abs(x)
    if p == 2.0:
        # Fast path: avoids pow, matches Flax _l2_normalize.
        return jnp.sqrt((abs_x * abs_x).sum(axis=axes, keepdims=True) + epsilon)
    if p == 1.0:
        # Fast path: no pow needed.
        return abs_x.sum(axis=axes, keepdims=True) + epsilon
    norm_p = jnp.power(abs_x, p).sum(axis=axes, keepdims=True)
    return jnp.power(norm_p + epsilon, 1.0 / p)


def _apply_scale_bias(
    y: Array,
    x_orig: Array,
    scale: tp.Optional[Array],
    bias: tp.Optional[Array],
    feature_axes: tuple[int, ...],
    dtype: tp.Optional[Dtype],
) -> Array:
    """Apply learned scale/bias and cast to output dtype."""
    feature_shape = [1] * y.ndim
    for ax in feature_axes:
        feature_shape[ax] = y.shape[ax]

    if scale is not None:
        y = y * scale.reshape(feature_shape)
    if bias is not None:
        y = y + bias.reshape(feature_shape)

    args = [x_orig]
    if scale is not None:
        args.append(scale)
    if bias is not None:
        args.append(bias)
    out_dtype = dtypes.canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, out_dtype)


class LpNorm(Module):
    """Lp normalization layer.

    Normalizes the input by dividing by the Lp norm along the reduction axes,
    then optionally applies a learned scale and bias. This generalizes L2
    normalization (p=2) and can approximate LayerNorm-like behaviour for p=2
    without mean centering.

    For numerical stability (following the pattern in Flax's normalization):
      - Computations are promoted to at least float32.
      - A small epsilon is added before taking the p-th root to avoid
        division by zero.

    Args:
        num_features: the number of input features (size of the feature axis).
        p: the exponent for the Lp norm. Must be > 0. Default is 2.0 (L2 norm).
        epsilon: a small float added inside the norm to avoid division by zero.
        dtype: the dtype of the result (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        use_scale: if True, multiply by a learned scale parameter.
        use_bias: if True, add a learned bias parameter.
        scale_init: initializer for the scale parameter.
        bias_init: initializer for the bias parameter.
        reduction_axes: axes over which to compute the Lp norm.
        feature_axes: axes for learned scale/bias parameters.
        promote_dtype: function to promote the dtype of inputs and parameters.
        rngs: rng key.
    """

    def __init__(
        self,
        num_features: int,
        *,
        p: float = 2.0,
        epsilon: float = 1e-6,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        use_scale: bool = True,
        use_bias: bool = False,
        scale_init: Initializer = initializers.ones,
        bias_init: Initializer = initializers.zeros,
        reduction_axes: Axes = -1,
        feature_axes: Axes = -1,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: rnglib.Rngs,
    ):
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")

        feature_shape = (num_features,)

        self.scale: nnx.Param[jax.Array] | None
        if use_scale:
            key = rngs.params()
            self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
        else:
            self.scale = nnx.data(None)

        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            key = rngs.params()
            self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
        else:
            self.bias = nnx.data(None)

        self.num_features = num_features
        self.p = p
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.promote_dtype = promote_dtype

    def __call__(
        self,
        x: Array,
        mask: tp.Optional[jax.Array] = None,
    ) -> Array:
        """Normalize the input by its Lp norm.

        Args:
            x: the input array.
            mask: unused, accepted for API compatibility with other norm layers.

        Returns:
            Normalized input (same shape as input).
        """
        del mask

        scale = self.scale[...] if self.scale else None
        bias = self.bias[...] if self.bias else None
        x, scale, bias = self.promote_dtype((x, scale, bias), dtype=self.dtype)

        # Promote to at least float32 for numerical stability.
        compute_dtype = jnp.promote_types(jnp.result_type(x), jnp.float32)
        x_f = jnp.asarray(x, compute_dtype)

        reduction_axes = _canonicalize_axes(x_f.ndim, self.reduction_axes)
        feature_axes = _canonicalize_axes(x_f.ndim, self.feature_axes)

        norm = _lp_norm(x_f, self.p, reduction_axes, self.epsilon)
        y = x_f / norm

        return _apply_scale_bias(y, x, scale, bias, feature_axes, self.dtype)


class LpNormClip(Module):
    """Lp norm clipping layer.

    Clips the Lp norm of the input to lie within
    ``[min_norm * d^(1/p), max_norm * d^(1/p)]`` where *d* is the number of
    elements along the reduction axes.  The ``d^(1/p)`` scaling (enabled by
    default) makes the bounds *per-element*: with ``max_norm=1`` and ``p=2``
    the effective bound is ``sqrt(d)``, i.e. the norm you'd get if every
    element were exactly 1.  Set ``scale_by_dim=False`` to use the raw
    bounds directly.

    When the norm already falls inside the range the input is passed through
    unchanged; otherwise it is rescaled so that its norm equals the nearest
    bound. An optional learned scale and bias are applied afterwards.

    This is useful for constraining activations (e.g. gradient clipping style
    regularisation on hidden states) while preserving direction.

    For numerical stability (following the pattern in Flax's normalization):
      - Computations are promoted to at least float32.
      - A small epsilon is added before taking the p-th root to avoid
        division by zero.

    Args:
        num_features: the number of input features (size of the feature axis).
        p: the exponent for the Lp norm. Must be > 0. Default is 2.0 (L2 norm).
        min_norm: per-element lower bound on the norm (before ``d^(1/p)``
            scaling). Default is 0.0 (no lower clipping).
        max_norm: per-element upper bound on the norm (before ``d^(1/p)``
            scaling). Use ``float('inf')`` to disable upper clipping.
            Default is 1.0.
        scale_by_dim: if True (default), the effective bounds are
            ``{min,max}_norm * d^(1/p)`` where *d* is the number of elements
            along the reduction axes.  If False, the raw bounds are used.
        epsilon: a small float added inside the norm to avoid division by zero.
        dtype: the dtype of the result (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        use_scale: if True, multiply by a learned scale parameter.
        use_bias: if True, add a learned bias parameter.
        scale_init: initializer for the scale parameter.
        bias_init: initializer for the bias parameter.
        reduction_axes: axes over which to compute the Lp norm.
        feature_axes: axes for learned scale/bias parameters.
        promote_dtype: function to promote the dtype of inputs and parameters.
        rngs: rng key.
    """

    def __init__(
        self,
        num_features: int,
        *,
        p: float = 2.0,
        min_norm: float = 0.0,
        max_norm: float = 1.0,
        scale_by_dim: bool = True,
        epsilon: float = 1e-12,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        use_scale: bool = True,
        use_bias: bool = False,
        scale_init: Initializer = initializers.ones,
        bias_init: Initializer = initializers.zeros,
        reduction_axes: Axes = -1,
        feature_axes: Axes = -1,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: rnglib.Rngs,
    ):
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        if min_norm < 0:
            raise ValueError(f"min_norm must be non-negative, got {min_norm}")
        if max_norm < min_norm:
            raise ValueError(f"max_norm ({max_norm}) must be >= min_norm ({min_norm})")

        feature_shape = (num_features,)

        self.scale: nnx.Param[jax.Array] | None
        if use_scale:
            key = rngs.params()
            self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
        else:
            self.scale = nnx.data(None)

        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            key = rngs.params()
            self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
        else:
            self.bias = nnx.data(None)

        self.num_features = num_features
        self.p = p
        self.min_norm = min_norm
        self.max_norm = max_norm
        self.scale_by_dim = scale_by_dim
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.promote_dtype = promote_dtype

    def __call__(
        self,
        x: Array,
        mask: tp.Optional[jax.Array] = None,
    ) -> Array:
        """Clip the Lp norm of the input.

        The effective bounds are ``{min,max}_norm * d^(1/p)`` when
        ``scale_by_dim=True`` (default), or the raw ``{min,max}_norm`` when
        ``scale_by_dim=False``.

        Args:
            x: the input array.
            mask: unused, accepted for API compatibility with other norm layers.

        Returns:
            Norm-clipped input (same shape as input).
        """
        del mask

        scale = self.scale[...] if self.scale else None
        bias = self.bias[...] if self.bias else None
        x, scale, bias = self.promote_dtype((x, scale, bias), dtype=self.dtype)

        # Promote to at least float32 for numerical stability.
        compute_dtype = jnp.promote_types(jnp.result_type(x), jnp.float32)
        x_f = jnp.asarray(x, compute_dtype)

        reduction_axes = _canonicalize_axes(x_f.ndim, self.reduction_axes)
        feature_axes = _canonicalize_axes(x_f.ndim, self.feature_axes)

        norm = _lp_norm(x_f, self.p, reduction_axes, self.epsilon)

        # Compute effective bounds, optionally scaled by d^(1/p).
        lo = self.min_norm
        hi = self.max_norm
        if self.scale_by_dim:
            d = 1
            for ax in reduction_axes:
                d *= x_f.shape[ax]
            dim_scale = d ** (1.0 / self.p)
            lo = lo * dim_scale
            hi = hi * dim_scale

        # Clip the norm and rescale.  The epsilon inside _lp_norm already
        # ensures norm > 0, so this division is safe.
        clipped_norm = jnp.clip(norm, lo, hi)
        y = x_f * (clipped_norm / norm)

        return _apply_scale_bias(y, x, scale, bias, feature_axes, self.dtype)


__all__ = [
    "LayerNorm",
    "BatchNorm",
    "RMSNorm",
    "GroupNorm",
    "WeightNorm",
    "SpectralNorm",
    "LpNorm",
    "LpNormClip",
]
