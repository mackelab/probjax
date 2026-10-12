import inspect
import warnings
from functools import partial
from typing import Optional, Sequence, Tuple

import flax.nnx as nnx
import jax.numpy as jnp

from probjax.utils.typing import (
    Array,
    ArrayLike,
    DTypeLike,
    ModuleLikeType,
)

# Distinguish an omitted builder from explicit None (which may disable it).
DEFAULT_MODULE = object()


_RNG_SUPPORT_BY_TYPE: dict[type, bool] = {
    nnx.Linear: False,
}


def identity_1x1(_, shape: Sequence[int], dtype=jnp.float32):
    """Kernel init for an all-ones-kernel Conv that starts as identity.

    Shape is (*spatial, C_in, C_out) for any number of spatial dims — 1-D
    convs give (1, C_in, C_out), 2-D give (1, 1, C_in, C_out). If C_in ≠ C_out
    the extra channels are zero-filled.
    """
    k = jnp.zeros(shape, dtype)
    diag = jnp.arange(min(shape[-2], shape[-1]))
    # set W[0, ..., 0, i, i] = 1
    spatial_origin = (0,) * (len(shape) - 2)
    k = k.at[(*spatial_origin, diag, diag)].set(1.0)
    return k


def pad_to_power_of_2(arr: Array, min_size: int = 16, axis=(-1,)) -> Array:
    """Pad the array to the next power of 2 greater than min_size along given axis."""

    def next_power_of_2(x):
        return 1 << (x - 1).bit_length()

    target_shape = list(arr.shape)
    for ax in axis:
        seq_len = arr.shape[ax]
        if seq_len < min_size:
            target_shape[ax] = min_size
        else:
            target_shape[ax] = next_power_of_2(seq_len)

    pad_width = [
        (0, target - current)
        for current, target in zip(arr.shape, target_shape, strict=False)
    ]
    return jnp.pad(arr, pad_width)


def flatten_to_btd(x: ArrayLike) -> Tuple[Array, Tuple[int, ...]]:
    """Flatten leading batch dims to (B, T, D); returns flattened tensor and original shape."""
    x = jnp.asarray(x)
    if x.ndim < 2:
        raise ValueError(
            f"Expected tensor with at least 2 dims (time, dim); got shape {x.shape}"
        )
    orig = tuple(x.shape)
    time_dim = x.shape[-2]
    feature_dim = x.shape[-1]
    x = x.reshape((-1, time_dim, feature_dim))
    return x, orig


def restore_from_btd(x: Array, orig_shape: Optional[tuple[int, ...]]) -> Array:
    """Restore tensor from (B, T, D) back to original leading batch dims."""
    if orig_shape is None:
        return x
    return x.reshape(orig_shape)


def normalize_attn_mask(mask: Array | None) -> Array | None:
    """Normalize attention mask shapes to broadcast with [B, H, T, T]."""
    if mask is None:
        return None
    mask = jnp.asarray(mask)
    if mask.ndim == 2:
        return mask[None, None, :, :]
    if mask.ndim == 3:
        return mask[:, None, :, :]
    if mask.ndim == 4:
        return mask
    raise ValueError(f"Mask must have ndim 2, 3, or 4; got {mask.ndim}.")


def normalize_attn_bias(bias: Array | None) -> Array | None:
    """Normalize attention bias shapes to broadcast with attention logits [B, H, T, T]."""
    if bias is None:
        return None
    bias = jnp.asarray(bias)
    if bias.ndim == 2:
        return bias[None, None, :, :]
    if bias.ndim == 3:
        return bias[:, None, :, :]
    if bias.ndim == 4:
        return bias
    raise ValueError(f"Bias must have ndim 2, 3, or 4; got {bias.ndim}.")


def filter_supported_kwargs(ctor, **kwargs) -> dict:
    """Keep only kwargs the constructor's signature accepts.

    Works for classes, functions, and functools.partial wrappers. Used to
    pass optional metadata (e.g. sharding) to layers that support it while
    remaining compatible with custom layer classes that don't.
    """
    try:
        parameters = inspect.signature(ctor).parameters
    except (ValueError, TypeError):
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters
        and parameters[key].kind != inspect.Parameter.POSITIONAL_ONLY
    }


def filter_precision_kwargs(cls: ModuleLikeType, **kwargs) -> dict:
    """Precision kwargs (``dtype``, ``param_dtype``, ...) to construct ``cls`` with.

    Drops ``None`` values (so ``cls`` keeps its own defaults) and anything
    ``cls`` does not accept, since layer classes are pluggable and e.g.
    ``nnx.LayerNorm`` takes no ``precision``.
    """
    target_cls = cls
    while isinstance(target_cls, partial):
        target_cls = target_cls.func

    # JAX does not support a backward pass with preferred_element_type != input
    # dtype for convolutions (jax-ml/jax#31592), so it is dropped for them.
    if (
        target_cls in (nnx.Conv, nnx.ConvTranspose)
        and kwargs.get("preferred_element_type") is not None
    ):
        warnings.warn(
            f"Ignoring preferred_element_type={kwargs['preferred_element_type']!r} for "
            f"{target_cls.__name__}: JAX does not support a backward pass with "
            "preferred_element_type != input dtype (see jax-ml/jax#31592).",
            UserWarning,
            stacklevel=2,
        )
        kwargs.pop("preferred_element_type")

    try:
        accepted = inspect.signature(target_cls.__init__).parameters.keys()
    except (ValueError, TypeError):
        accepted = kwargs.keys()
    return {k: v for k, v in kwargs.items() if v is not None and k in accepted}


def cast_output(x: ArrayLike, dtype: DTypeLike | None) -> Array:
    """Cast ``x`` to ``dtype`` (a layer's ``preferred_element_type``); no-op for None."""
    return x if dtype is None else x.astype(dtype)


def module_accepts_rng(module) -> bool:
    # Functions, bound methods and partials can have different signatures even
    # when they share a type. Inspect those directly, without a per-type cache.
    variable_signature = inspect.isroutine(module) or isinstance(module, partial)
    module_type = type(module)
    if not variable_signature and module_type in _RNG_SUPPORT_BY_TYPE:
        return _RNG_SUPPORT_BY_TYPE[module_type]
    try:
        params = inspect.signature(module).parameters.values()
        supports_rng = any(
            (p.name == "rng" and p.kind != inspect.Parameter.POSITIONAL_ONLY)
            or p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params
        )
    except (TypeError, ValueError):
        supports_rng = True
    if not variable_signature:
        _RNG_SUPPORT_BY_TYPE[module_type] = supports_rng
    return supports_rng
