# indexed_async_dataloader_v6.py
import asyncio
import collections
import contextlib
import itertools
import queue
import threading
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import prefetch_to_device
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding

from probjax.utils.typing import Device


# ------------------------- small helpers ----------------------------- #
def _tree_to_jnp(batch, host_device: Device):
    """Materialize batch leaves as JAX arrays on `host_device`."""

    def to_host(x):
        return jax.device_put(x, host_device)

    return jax.tree_util.tree_map(to_host, batch)


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


def _prefetch_sharding(iterator, size, sharding: Sharding):
    dq = collections.deque()
    _put = lambda x: jax.device_put(x, sharding)
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
        and ``chunk_volume`` is the product over ``chunk_shape``. When a spatial
        dimension is not divisible by the corresponding ``chunk_shape``, the input is
        padded with zeros at the end to the nearest multiple before chunking.
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


# --------------------------------------------------------------------- #


Transform = Union[Callable[[Any], Any], Sequence[Callable[[Any], Any]]]

_STOP = object()


class _WorkerError:
    __slots__ = ("exc", "tb")

    def __init__(self, exc: BaseException, tb: str):
        self.exc = exc
        self.tb = tb


class DataLoader:
    """
    Infinite, non-blocking, resource-safe JAX DataLoader **with on-the-fly transforms**.

    Parameters added
    ----------------
    host_transforms   : callable | Sequence[callable] | None
        Applied on the **CPU** worker thread immediately after `dataset[idxs]`.
    device_transforms : callable | Sequence[callable] | None
        Applied **after** the batch has been moved to accelerator memory
        (and sharded, if `shard=True`). Pass JIT-compiled functions for best speed.
    host_device       : jax.Device | None
        Device that stores producer-side batches before they are prefetched.
    sharding          : jax.sharding.Sharding | None
        Optional global sharding to apply when moving batches to device.
        Mutually exclusive with shard=True and mesh/batch_spec.
    mesh              : jax.sharding.Mesh | None
        Mesh used to build a NamedSharding when batch_spec is provided.
    batch_spec        : jax.sharding.PartitionSpec | None
        PartitionSpec for the batch when mesh is provided.
    max_in_flight      : int | None
        Number of in-flight CPU batch jobs scheduled via `run_in_executor`.
        Keeping this >1 enables actual async pipelining. Order is preserved.
        Defaults to `num_async_workers` (min 1).
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
        devices: Optional[Sequence[Device]] = None,
        num_async_workers: int = 1,
        host_device: Optional[Device] = None,
        max_in_flight: Optional[int] = None,
        sharding: Sharding | None = None,
        mesh: Mesh | None = None,
        batch_spec: PartitionSpec | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not (0.0 < min_fill <= 1.0):
            raise ValueError("min_fill must be in (0,1].")
        if num_prefetch_host <= 0:
            raise ValueError("num_prefetch_host must be positive.")
        if num_async_workers <= 0:
            raise ValueError("num_async_workers must be positive.")
        if max_in_flight is not None and max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive or None.")

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
        if shard and (sharding is not None or mesh is not None or batch_spec is not None):
            raise ValueError("Use either shard=True or explicit sharding/mesh, not both.")
        self._shard_flag = shard
        self._devices = list(devices) if devices else jax.local_devices()
        self._n_dev = len(self._devices)
        if host_device is None:
            try:
                cpu_devices = jax.devices("cpu")
            except RuntimeError:
                cpu_devices = []
            host_device = cpu_devices[0] if cpu_devices else jax.devices()[0]
        self._host_device = host_device

        self._sharding = self._resolve_sharding(sharding, mesh, batch_spec)

        # -------- infra ---------------- #
        self._num_async_workers = int(num_async_workers)
        self._max_in_flight = (
            max(1, self._num_async_workers)
            if max_in_flight is None
            else int(max_in_flight)
        )

        self._executor = ThreadPoolExecutor(max_workers=self._num_async_workers)
        self._stop_event = threading.Event()

        # worker error propagation (producer thread -> consumer thread)
        self._worker_error: Optional[_WorkerError] = None
        self._worker_error_lock = threading.Lock()

        # asyncio loop/task owned by producer thread
        self._loop_ref: Optional[asyncio.AbstractEventLoop] = None
        self._task_ref: Optional[asyncio.Task] = None

        self._producer_th = threading.Thread(target=self._producer_main, daemon=True)
        self._producer_th.start()

        self._closed = False
        self._iter_ref = None
        self._iter_token = None

        # IMPORTANT: don't pass a bound method to weakref.finalize (can keep self alive)
        self._finalizer_ref = weakref.finalize(
            self, DataLoader._finalize, weakref.ref(self)
        )

    @staticmethod
    def _resolve_sharding(
        sharding: Sharding | None,
        mesh: Mesh | None,
        batch_spec: PartitionSpec | None,
    ) -> Sharding | None:
        if sharding is not None:
            if mesh is not None or batch_spec is not None:
                raise ValueError("Provide either sharding or mesh+batch_spec, not both.")
            return sharding
        if mesh is None and batch_spec is None:
            return None
        if mesh is None or batch_spec is None:
            raise ValueError("mesh and batch_spec must be provided together.")
        return NamedSharding(mesh, batch_spec)

    @staticmethod
    def _finalize(self_ref: "weakref.ReferenceType[DataLoader]"):
        obj = self_ref()
        if obj is not None:
            with contextlib.suppress(Exception):
                obj.close()

    # ---------------- epoch index generator --------------------------- #
    def _index_batches(self):
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
        batch = self._ds[idxs]
        return batch

    # ---------------- internal queue utilities ------------------------ #
    def _drain_queue(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _force_put(self, item, *, clear: bool = False) -> None:
        """
        Ensure `item` gets into the queue without blocking. Optionally clear the queue.
        This is used ONLY for terminal signaling (_STOP) so dropping queued batches is OK.
        """
        if clear:
            self._drain_queue()
        while True:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                # Make room by dropping one element.
                with contextlib.suppress(queue.Empty):
                    self._q.get_nowait()

    def _set_worker_error(self, exc: BaseException) -> None:
        err = _WorkerError(exc, traceback.format_exc())
        with self._worker_error_lock:
            self._worker_error = err

    def _get_worker_error(self) -> Optional[_WorkerError]:
        with self._worker_error_lock:
            return self._worker_error

    # ---------------- background producer ----------------------------- #
    def _producer_main(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop_ref = loop
        self._task_ref = loop.create_task(self._fill_queue())
        try:
            loop.run_until_complete(self._task_ref)
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _fill_queue(self):
        """
        Async producer:
        - schedules CPU work via run_in_executor
        - maintains up to `max_in_flight` in-flight tasks (ORDER PRESERVED)
        - pushes batches into a bounded host queue without blocking the event loop
        - on error, stores traceback and wakes consumer via _STOP
        """
        pending: collections.deque[asyncio.Task] = collections.deque()

        async def run_one(idxs):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._process_batch, idxs)

        async def put_batch_nonblocking(item) -> bool:
            # Never block the event loop; wait for space or stop.
            while not self._stop_event.is_set():
                try:
                    self._q.put_nowait(item)
                    return True
                except queue.Full:
                    await asyncio.sleep(0.005)
            return False

        try:
            for idxs in self._index_batches():
                if self._stop_event.is_set():
                    break

                # schedule next compute
                pending.append(asyncio.create_task(run_one(idxs)))

                # keep pipeline bounded; preserve order by awaiting oldest
                if len(pending) >= self._max_in_flight:
                    oldest = pending.popleft()
                    try:
                        batch = await oldest
                    except asyncio.CancelledError:
                        break
                    except BaseException as e:
                        self._set_worker_error(e)
                        self._stop_event.set()
                        break

                    if not await put_batch_nonblocking(batch):
                        break

            # drain remaining tasks (in order) if not stopping
            while pending and not self._stop_event.is_set():
                oldest = pending.popleft()
                try:
                    batch = await oldest
                except asyncio.CancelledError:
                    break
                except BaseException as e:
                    self._set_worker_error(e)
                    self._stop_event.set()
                    break
                if not await put_batch_nonblocking(batch):
                    break

        except asyncio.CancelledError:
            # Expected during close()
            pass
        except BaseException as e:
            self._set_worker_error(e)
        finally:
            # Cancel anything still pending
            while pending:
                t = pending.popleft()
                t.cancel()

            # Unblock consumer immediately. Clear queue so _STOP always lands.
            self._force_put(_STOP, clear=True)

    def _process_batch(self, idxs):
        batch = self._fetch_batch(idxs)
        for fn in self._host_tfns:
            batch = fn(batch)
        batch = _tree_to_jnp(batch, self._host_device)
        return batch

    # ---------------- host iterator w/ recycling ---------------------- #
    def _host_iter(self):
        maxsize = getattr(self._q, "maxsize", 0) or 0
        min_size = int(maxsize * self._min_fill) if maxsize > 0 else 0

        while True:
            item = self._q.get()

            if item is _STOP:
                err = self._get_worker_error()
                if err is not None:
                    raise RuntimeError(
                        f"DataLoader worker failed:\n{err.tb}"
                    ) from err.exc
                raise StopIteration

            batch = item

            # recycle if below threshold (your original behavior)
            while min_size > 0 and self._q.qsize() < min_size and not self._q.full():
                try:
                    self._q.put_nowait(batch)
                except queue.Full:
                    break

            yield batch

    # ---------------- main iterator API ------------------------------- #
    def _ensure_iter(self):
        if self._closed:
            return
        if self._iter_ref is None:
            token = object()
            self._iter_token = token
            self._iter_ref = self._iter_gen(token)

    def __iter__(self):
        self._ensure_iter()
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        self._ensure_iter()
        return next(self._iter_ref)

    def _iter_gen(self, token):
        host_it = self._host_iter()
        if self._shard_flag:
            host_it = (_shard(b, self._n_dev) for b in host_it)

        if self._sharding is not None:
            dev_it = _prefetch_sharding(host_it, self._prefetch_dev, self._sharding)
        else:
            dev_it = (
                _prefetch_single(host_it, self._prefetch_dev, self._devices[0])
                if self._n_dev == 1 and not self._shard_flag
                else prefetch_to_device(host_it, self._prefetch_dev, self._devices)
            )

        # generator wrapper ensures close() runs on exception unwind (CPython refcount)
        def gen():
            try:
                for batch in dev_it:
                    for fn in self._device_tfns:
                        batch = fn(batch)
                    yield batch
            finally:
                # Avoid closing the loader if a newer iterator replaced this one.
                if self._iter_token is token:
                    self.close()

        return gen()

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
        self._iter_ref = None
        self._iter_token = None

        self._stop_event.set()

        # Unblock consumer immediately (and avoid deadlock if queue is full).
        self._force_put(_STOP, clear=True)

        # Cancel the asyncio producer task thread-safely.
        loop = self._loop_ref
        task = self._task_ref
        if loop is not None and task is not None:

            def _cancel_task():
                if not task.done():
                    task.cancel()

            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(_cancel_task)

        if self._producer_th.is_alive():
            self._producer_th.join(timeout=1.0)

        # Don't wait: prevents hanging on long-running host transforms / dataset.
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
