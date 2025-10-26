# indexed_async_dataloader_v6.py
import asyncio
import collections
import itertools
import queue
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import prefetch_to_device


# ------------------------- small helpers ----------------------------- #
def _tree_to_jnp(batch):
    return jax.tree_util.tree_map(jnp.asarray, batch)


def _shard(batch, n_dev):
    def split(x):
        b = x.shape[0]
        if b % n_dev:
            raise ValueError(f"Batch {b} not divisible by {n_dev}")
        return x.reshape((n_dev, b // n_dev) + x.shape[1:])

    return jax.tree_util.tree_map(split, batch)


def _prefetch_single(iterator, size, device):
    dq = collections.deque()
    _put = lambda x: jax.device_put(x, device)
    _fill = lambda n: [
        dq.append(jax.tree_util.tree_map(_put, d))
        for d in itertools.islice(iterator, n)
    ]
    _fill(size)
    while dq:
        yield dq.popleft()
        _fill(1)


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
        and ``chunk_volume`` is the product over ``chunk_shape``.
    """

    x = jnp.asarray(x)
    if isinstance(chunk_shape, int):
        if chunk_shape <= 0:
            raise ValueError("chunk_shape must be positive.")
        chunk_shape = (chunk_shape,)
    else:
        chunk_shape = tuple(int(cs) for cs in chunk_shape)
        if not chunk_shape:
            raise ValueError("chunk_shape must be non-empty.")
        if any(cs <= 0 for cs in chunk_shape):
            raise ValueError("All entries in chunk_shape must be positive.")

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

    for i, (size, chunk) in enumerate(zip(spatial_shape, chunk_shape, strict=False)):
        if size % chunk:
            raise ValueError(
                f"Spatial dimension {i} with size {size} is not divisible by "
                f"chunk size {chunk}."
            )

    chunk_counts = tuple(
        size // chunk for size, chunk in zip(spatial_shape, chunk_shape, strict=False)
    )

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
        divisible by the corresponding entry in ``chunk_shape``.
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
    if isinstance(chunk_shape, int):
        if chunk_shape <= 0:
            raise ValueError("chunk_shape must be positive.")
        chunk_shape = (chunk_shape,)
    else:
        chunk_shape = tuple(int(cs) for cs in chunk_shape)
        if not chunk_shape:
            raise ValueError("chunk_shape must be non-empty.")
        if any(cs <= 0 for cs in chunk_shape):
            raise ValueError("All entries in chunk_shape must be positive.")

    spatial_shape = tuple(int(s) for s in spatial_shape)
    if any(s <= 0 for s in spatial_shape):
        raise ValueError("All entries in spatial_shape must be positive.")

    spatial_ndim = len(chunk_shape)
    if len(spatial_shape) != spatial_ndim:
        raise ValueError(
            f"spatial_shape has {len(spatial_shape)} dimensions but chunk_shape "
            f"requires {spatial_ndim}."
        )

    chunk_counts = []
    for s, c in zip(spatial_shape, chunk_shape, strict=False):
        if s % c:
            raise ValueError(
                "Each entry in spatial_shape must be divisible by the matching "
                "chunk_shape."
            )
        chunk_counts.append(s // c)
    chunk_counts = tuple(chunk_counts)

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

    full_spatial_shape = tuple(
        n * c for n, c in zip(chunk_counts, chunk_shape, strict=False)
    )
    result = interleaved.reshape(batch_shape + full_spatial_shape + (channel_dim,))

    if channel_axis is None:
        return result[..., 0]

    channel_axis = int(channel_axis)
    channel_axis = channel_axis % result.ndim
    if channel_axis != result.ndim - 1:
        result = jnp.moveaxis(result, -1, channel_axis)
    return result


# --------------------------------------------------------------------- #


Transform = Union[Callable[[Any], Any], Sequence[Callable[[Any], Any]]]


class DataLoader:
    """
    Infinite, non-blocking, resource-safe JAX DataLoader **with on-the-fly transforms**.

    Parameters added
    ----------------
    host_transforms   : callable | Sequence[callable] | None
        Applied on the **CPU** worker thread immediately after `dataset[idxs]`.
    device_transforms : callable | Sequence[callable] | None
        Applied **after** the batch has been moved to accelerator memory
        (and sharded, if `shard=True`).  Pass JIT-compiled functions for
        best speed (`@jax.jit` or `@jax.pmap` when multi-device).
    """

    # ------------------------- init ----------------------------------- #
    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        loop: bool = True,
        host_transforms: Optional[Transform] = None,
        device_transforms: Optional[Transform] = None,
        num_prefetch_host: int = 16,
        min_fill: float = 0.5,
        num_prefetch_device: int = 2,
        shard: bool = False,
        devices: Optional[Sequence[jax.Device]] = None,
        num_async_workers: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not (0.0 < min_fill <= 1.0):
            raise ValueError("min_fill must be in (0,1].")

        self._ds, self._N, self._bsz = dataset, len(dataset), batch_size
        self._drop_last, self._loop = drop_last, loop

        self._rng = np.random.default_rng(seed) if shuffle else None

        # -------- transforms ---------- #
        self._host_tfns = (
            list(host_transforms)
            if isinstance(host_transforms, (list, tuple))
            else ([host_transforms] if host_transforms else [])
        )
        self._device_tfns = (
            list(device_transforms)
            if isinstance(device_transforms, (list, tuple))
            else ([device_transforms] if device_transforms else [])
        )

        # -------- queues / buffers ----- #
        self._q = queue.Queue(num_prefetch_host)
        self._min_fill, self._prefetch_dev = min_fill, max(1, num_prefetch_device)

        # -------- device config -------- #
        self._shard_flag = shard
        self._devices = list(devices) if devices else jax.local_devices()
        self._n_dev = len(self._devices)

        # -------- infra ---------------- #
        self._executor = ThreadPoolExecutor(max_workers=num_async_workers)
        self._stop_event = threading.Event()
        self._producer_th = threading.Thread(target=self._producer_main, daemon=True)
        self._producer_th.start()
        self._closed = False
        weakref.finalize(self, self._finalizer)

    # ---------------- epoch index generator --------------------------- #
    def _index_batches(self):
        # This can be overwritten for more complicated indexing
        while True:
            idx = np.arange(self._N, dtype=np.int64)
            if self._rng is not None:
                self._rng.shuffle(idx)
            if self._drop_last:
                idx = idx[: (len(idx) // self._bsz) * self._bsz]
            for i in range(0, len(idx), self._bsz):
                yield idx[i : i + self._bsz]
            if not self._loop:
                break

    def _fetch_batch(self, idxs):
        # This can be overwritten for more complicated dataset indexing
        batch = self._ds[idxs]
        return batch

    # ---------------- background producer ----------------------------- #
    def _producer_main(self):
        asyncio.run(self._fill_queue())

    async def _fill_queue(self):
        try:
            for idxs in self._index_batches():
                if self._stop_event.is_set():
                    break
                fut = asyncio.get_running_loop().run_in_executor(
                    self._executor, self._process_batch, idxs
                )
                batch = await fut
                self._q.put(batch)  # blocks if queue full
            self._q.put(None)
        except Exception:
            self._q.put(None)
            raise

    def _process_batch(self, idxs):
        batch = self._fetch_batch(idxs)
        for fn in self._host_tfns:
            batch = fn(batch)
        batch = _tree_to_jnp(batch)
        return batch

    # ---------------- host iterator w/ recycling ---------------------- #
    def _host_iter(self):
        min_size = int(self._q.maxsize * self._min_fill)
        while True:
            batch = self._q.get()
            if batch is None:
                raise StopIteration
            while self._q.qsize() < min_size and not self._q.full():
                try:
                    self._q.put_nowait(batch)
                except queue.Full:
                    break
            yield batch

    # ---------------- main iterator API ------------------------------- #
    def __iter__(self):
        host_it = self._host_iter()
        if self._shard_flag and self._n_dev > 1:
            host_it = (_shard(b, self._n_dev) for b in host_it)

        self._dev_it = (
            _prefetch_single(host_it, self._prefetch_dev, self._devices[0])
            if self._n_dev == 1 and not self._shard_flag
            else prefetch_to_device(host_it, self._prefetch_dev, self._devices)
        )
        return self

    def __next__(self):
        batch = next(self._dev_it)
        for fn in self._device_tfns:
            batch = fn(batch)
        return batch

    def __len__(self):
        return (
            (self._N // self._bsz)
            if self._drop_last
            else (self._N + self._bsz - 1) // self._bsz
        )

    # ---------------- clean-up / context manager ----------------------- #
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self._q.put(None)
        if self._producer_th.is_alive():
            self._producer_th.join(timeout=1.0)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _finalizer(self):
        try:
            self.close()
        except Exception:
            pass

    def __del__(self):
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
