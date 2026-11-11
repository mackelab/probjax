from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from probjax.utils.guards import ensure_tuple


def _normalize_chunk_shape(chunk_shape: Union[int, Sequence[int]]) -> tuple:
    """Shared ``chunkify``/``unchunkify`` chunk-shape validation."""
    return ensure_tuple("chunk_shape", chunk_shape, positive=True)


Transform = Union[Callable[[Any], Any], Sequence[Callable[[Any], Any]]]


def as_transform_list(transforms: Transform | None) -> list[Callable[[Any], Any]]:
    """Normalize a transform argument to a list of callables.

    Parameters
    ----------
    transforms : callable | Sequence[callable] | None
        A single callable, a sequence of callables, or ``None``.

    Returns
    -------
    list[Callable]
        ``[fn]`` for a callable, ``list(transforms)`` for a sequence, and
        ``[]`` for ``None`` or an empty sequence.
    """
    if callable(transforms):
        return [transforms]
    return list(transforms) if transforms else []


def compose(*fns: Callable[[Any], Any] | Sequence[Callable[[Any], Any]] | None):
    """Compose callables into a single callable applied left to right.

    Parameters
    ----------
    *fns : callable | Sequence[callable] | None
        Callables or sequences of callables (flattened in order). ``None``
        entries are skipped.

    Returns
    -------
    Callable
        ``x -> fn_n(...fn_2(fn_1(x)))``; the identity if no callables given.
    """
    flat = [f for fn in fns for f in as_transform_list(fn)]

    def composed(x):
        for f in flat:
            x = f(x)
        return x

    return composed


def chunkify(
    x: "jax.Array",
    chunk_shape: Union[int, Sequence[int]],
    *,
    channel_axis: Optional[int] = -1,
) -> "jax.Array":
    """
    Partition an array with spatial structure into a sequence of transformer tokens.

    Parameters
    ----------
    x:
        Input array of shape ``(*batch_dims, *spatial_dims, channels)``. The channel
        dimension can be omitted by setting ``channel_axis=None``.
    chunk_shape:
        Size of each chunk along the spatial dimensions. Can be an ``int`` (applied
        uniformly) or an ``Iterable[int]`` with length equal to the number of spatial
        dimensions.
    channel_axis:
        Axis that stores per-location features (e.g. RGB channels). Set to ``None`` if
        the input does not have a dedicated channel dimension. Defaults to the last
        axis.

    Returns
    -------
    jax.Array
        Array of shape ``(*batch_dims, num_chunks, chunk_volume * channels)`` where
        ``num_chunks`` is the product over the number of chunks per spatial dimension
        and ``chunk_volume`` is the product over ``chunk_shape``. When a spatial
        dimension is not divisible by the corresponding ``chunk_shape``, the input is
        padded with zeros at the end to the nearest multiple before chunking.
    """

    x = jnp.asarray(x)
    chunk_shape = _normalize_chunk_shape(chunk_shape)

    spatial_ndim = len(chunk_shape)

    if channel_axis is None:
        x = jnp.expand_dims(x, axis=-1)
        channel_axis = x.ndim - 1
    else:
        channel_axis = int(channel_axis)
        if not (-x.ndim <= channel_axis < x.ndim):
            raise ValueError(
                f"channel_axis={channel_axis} is out of bounds for array with "
                f"{x.ndim} dimensions."
            )
        channel_axis = channel_axis % x.ndim
        if channel_axis != x.ndim - 1:
            x = jnp.moveaxis(x, channel_axis, -1)

    if x.ndim < spatial_ndim + 1:
        raise ValueError(
            f"Input must have at least {spatial_ndim} spatial dims plus channels, "
            f"got shape {x.shape}."
        )

    spatial_start = x.ndim - spatial_ndim - 1
    batch_shape = x.shape[:spatial_start]
    spatial_shape = x.shape[spatial_start:-1]
    channel_dim = x.shape[-1]

    if len(spatial_shape) != spatial_ndim:
        raise ValueError(
            f"Expected {spatial_ndim} spatial dimensions, got {len(spatial_shape)}."
        )

    pad_widths = [(0, 0)] * x.ndim
    padded_spatial_shape = []
    chunk_counts = []
    pad_required = False
    for idx, (size, chunk) in enumerate(zip(spatial_shape, chunk_shape, strict=False)):
        padded_size = ((size + chunk - 1) // chunk) * chunk
        pad_after = padded_size - size
        if pad_after:
            pad_required = True
            pad_widths[spatial_start + idx] = (0, pad_after)
        padded_spatial_shape.append(padded_size)
        chunk_counts.append(padded_size // chunk)
    if pad_required:
        x = jnp.pad(x, pad_widths)
        spatial_shape = tuple(padded_spatial_shape)
    else:
        spatial_shape = tuple(spatial_shape)

    chunk_counts = tuple(chunk_counts)

    # Reshape to interleave chunk counts and chunk sizes, keeping batch dims in front.
    reshaped_shape: list[int] = list(batch_shape)
    for n_chunks, chunk_size in zip(chunk_counts, chunk_shape, strict=False):
        reshaped_shape.extend([n_chunks, chunk_size])
    reshaped_shape.append(channel_dim)
    tokens = x.reshape(reshaped_shape)

    batch_ndim = len(batch_shape)
    count_axes = [batch_ndim + 2 * idx for idx in range(spatial_ndim)]
    chunk_axes = [axis + 1 for axis in count_axes]
    perm = (
        list(range(batch_ndim))
        + count_axes
        + chunk_axes
        + [batch_ndim + 2 * spatial_ndim]
    )
    tokens = tokens.transpose(perm)

    total_chunks = int(np.prod(chunk_counts)) if chunk_counts else 1
    chunk_volume = int(np.prod(chunk_shape)) if chunk_shape else 1
    output_shape = batch_shape + (total_chunks, chunk_volume * channel_dim)

    return tokens.reshape(output_shape)


def unchunkify(
    tokens: "jax.Array",
    chunk_shape: Union[int, Sequence[int]],
    *,
    spatial_shape: Sequence[int],
    channel_axis: Optional[int] = -1,
) -> "jax.Array":
    """
    Invert ``chunkify`` by folding a token sequence back into its spatial layout.

    Parameters
    ----------
    tokens:
        Array of shape ``(*batch_dims, num_chunks, chunk_volume * channels)`` produced
        by :func:`chunkify`.
    chunk_shape:
        Chunk shape passed to :func:`chunkify`.
    spatial_shape:
        Spatial shape of the original tensor prior to chunking. Each entry must be
        positive. Spatial dimensions that are not divisible by ``chunk_shape`` are
        reconstructed with zero-padding that is removed before returning.
    channel_axis:
        Original channel axis supplied to :func:`chunkify`. Use ``None`` if the input
        tensor had no explicit channel dimension.

    Returns
    -------
    jax.Array
        Array of shape ``(*batch_dims, *spatial_shape, channels)`` (or without the
        channel dimension when ``channel_axis=None``).
    """

    tokens = jnp.asarray(tokens)
    chunk_shape = _normalize_chunk_shape(chunk_shape)

    spatial_shape = ensure_tuple("spatial_shape", spatial_shape, positive=True)

    spatial_ndim = len(chunk_shape)
    if len(spatial_shape) != spatial_ndim:
        raise ValueError(
            f"spatial_shape has {len(spatial_shape)} dimensions but chunk_shape "
            f"requires {spatial_ndim}."
        )

    chunk_counts = []
    padded_spatial_shape = []
    for s, c in zip(spatial_shape, chunk_shape, strict=False):
        chunk_count = (s + c - 1) // c
        chunk_counts.append(chunk_count)
        padded_spatial_shape.append(chunk_count * c)
    chunk_counts = tuple(chunk_counts)
    padded_spatial_shape = tuple(padded_spatial_shape)

    batch_shape = tokens.shape[:-2]
    total_chunks = tokens.shape[-2]
    chunk_volume = int(np.prod(chunk_shape)) if chunk_shape else 1

    if chunk_volume == 0:
        raise ValueError("chunk_shape must define a strictly positive chunk volume.")
    if tokens.shape[-1] % chunk_volume:
        raise ValueError(
            f"Last dimension ({tokens.shape[-1]}) is not divisible by chunk_volume "
            f"{chunk_volume}."
        )
    channel_dim = tokens.shape[-1] // chunk_volume

    expected_chunks = int(np.prod(chunk_counts)) if chunk_counts else 1
    if total_chunks != expected_chunks:
        raise ValueError(
            f"Token dimension ({total_chunks}) is incompatible with chunk counts "
            f"{chunk_counts}."
        )

    reshaped = tokens.reshape(batch_shape + chunk_counts + chunk_shape + (channel_dim,))

    batch_ndim = len(batch_shape)
    perm = list(range(batch_ndim))
    perm.extend(
        axis
        for idx in range(spatial_ndim)
        for axis in (batch_ndim + idx, batch_ndim + spatial_ndim + idx)
    )
    perm.append(batch_ndim + 2 * spatial_ndim)
    interleaved = reshaped.transpose(perm)

    result = interleaved.reshape(batch_shape + padded_spatial_shape + (channel_dim,))

    if padded_spatial_shape != spatial_shape:
        slices = [slice(None)] * result.ndim
        for axis_idx, size in enumerate(spatial_shape):
            slices[batch_ndim + axis_idx] = slice(0, size)
        result = result[tuple(slices)]

    if channel_axis is None:
        return result[..., 0]

    channel_axis = int(channel_axis)
    channel_axis = channel_axis % result.ndim
    if channel_axis != result.ndim - 1:
        result = jnp.moveaxis(result, -1, channel_axis)
    return result
