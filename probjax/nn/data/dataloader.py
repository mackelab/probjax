import asyncio
import collections
import contextlib
import queue
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence, Union

import jax
import numpy as np
from flax.jax_utils import prefetch_to_device
from jax.sharding import Mesh, NamedSharding

from probjax.nn.data.datasets import SimulationDataset
from probjax.nn.data.prefetch import (
    _WorkerError,
    _prefetch_sharding,
    _prefetch_single,
)
from probjax.nn.data.transforms import Transform, as_transform_list
from probjax.utils.guards import normalize_devices
from probjax.utils.typing import Array, Device, PyTree


# ------------------------- small helpers ----------------------------- #
def _tree_to_jnp(batch: PyTree[Array], host_device: Device) -> PyTree[Array]:
    """Materialize batch leaves as JAX arrays on `host_device`."""

    def to_host(x):
        return jax.device_put(x, host_device)

    return jax.tree_util.tree_map(to_host, batch)


def _tree_to_host(batch: PyTree[Array]) -> PyTree[Array]:
    """Normalize batch leaves to host-backed arrays."""

    return jax.tree_util.tree_map(np.asarray, batch)


def _shard(batch: PyTree[Array], n_dev: int) -> PyTree[Array]:
    def split(x):
        b = x.shape[0]
        if b % n_dev:
            raise ValueError(f"Batch {b} not divisible by {n_dev}")
        return x.reshape((n_dev, b // n_dev) + x.shape[1:])

    return jax.tree_util.tree_map(split, batch)


_STOP = object()


class _HostProducer:
    """Producer side of a `DataLoader`: dataset access, index batching, host
    transforms, the host queue and the threads that fill it.

    Deliberately holds no reference to the `DataLoader`. Its threads (and the
    device-prefetch thread, which drives `_host_iter`) keep it alive, so a
    reference back would keep a dropped loader alive for the process lifetime.
    The loader only references this object and calls `close` from a
    `weakref.finalize`.
    """

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        drop_last: bool,
        rng: Optional[np.random.Generator],
        loop: bool,
        host_tfns: list,
        queue_size: int,
        min_fill: float,
        num_async_workers: int,
        max_in_flight: int,
        has_named_sharding: bool,
        host_device: Device,
    ):
        self._ds, self._N, self._bsz = dataset, len(dataset), batch_size
        self._drop_last, self._loop, self._rng = drop_last, loop, rng
        self._host_tfns = host_tfns
        self._q = queue.Queue(queue_size)
        self._min_fill = min_fill
        self._has_named_sharding = has_named_sharding
        self._host_device = host_device
        self._max_in_flight = max_in_flight

        self._executor = ThreadPoolExecutor(max_workers=num_async_workers)
        self._stop_event = threading.Event()

        # worker error propagation (producer thread -> consumer thread)
        self._worker_error: Optional[_WorkerError] = None
        self._worker_error_lock = threading.Lock()

        # asyncio loop/task owned by producer thread
        self._loop_ref: Optional[asyncio.AbstractEventLoop] = None
        self._task_ref: Optional[asyncio.Task] = None

        # Device-side iterator of the current epoch stream, owned here so
        # close() can stop its thread without the DataLoader.
        self.dev_it = None

        self._closed = False
        self._producer_th = threading.Thread(target=self._producer_main, daemon=True)
        self._producer_th.start()

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
        # NOTE: no traceback formatting here (see _WorkerError); this runs on
        # the producer event loop and must return immediately.
        # After close() the executor is shut down under the producer; the
        # resulting errors are an artefact of stopping, not worker failures.
        if self._stop_event.is_set():
            return
        err = _WorkerError(exc)
        with self._worker_error_lock:
            self._worker_error = err

    def _get_worker_error(self) -> Optional[_WorkerError]:
        with self._worker_error_lock:
            return self._worker_error

    def _raise_if_worker_failed(self) -> None:
        err = self._get_worker_error()
        if err is not None:
            raise RuntimeError(
                f"DataLoader worker failed:\n{err.traceback_text()}"
            ) from err.exc

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

            if self._stop_event.is_set() or self._get_worker_error() is not None:
                # Abnormal end (error / close): unblock the consumer
                # immediately. Clear queue so _STOP always lands.
                self._force_put(_STOP, clear=True)
            else:
                # Clean exhaustion: queue _STOP behind the remaining batches
                # so no produced data is lost. Bail out to a clearing put if
                # a concurrent close() asks us to stop.
                while True:
                    if self._stop_event.is_set():
                        self._force_put(_STOP, clear=True)
                        break
                    try:
                        self._q.put(_STOP, timeout=0.05)
                        break
                    except queue.Full:
                        continue

    def _process_batch(self, idxs):
        batch = self._fetch_batch(idxs)
        for fn in self._host_tfns:
            batch = fn(batch)
        if self._has_named_sharding:
            batch = _tree_to_host(batch)
        else:
            batch = _tree_to_jnp(batch, self._host_device)
        return batch

    # ---------------- host iterator w/ recycling ---------------------- #
    def _host_iter(self):
        maxsize = getattr(self._q, "maxsize", 0) or 0
        # Recycling duplicates batches to keep the queue topped up; it is only
        # sound for an endless stream, so it is off for finite epochs.
        min_size = int(maxsize * self._min_fill) if maxsize > 0 and self._loop else 0

        while True:
            item = self._q.get()

            if item is _STOP:
                self._raise_if_worker_failed()
                # NOTE: plain `return`, not `raise StopIteration`: this is a
                # generator body, where raising StopIteration becomes a
                # RuntimeError (PEP 479).
                return

            batch = item

            # recycle if below threshold (your original behavior)
            recycled = False
            while min_size > 0 and self._q.qsize() < min_size and not self._q.full():
                try:
                    self._q.put_nowait(batch)
                    recycled = True
                except queue.Full:
                    break
            if recycled:
                # The producer is lagging (or dead and the error is still
                # being recorded). Yield the GIL so it gets a timeslice to
                # catch up or deliver its error instead of being starved by
                # this hot loop.
                time.sleep(0)

            yield batch

    # ---------------- clean-up ---------------------------------------- #
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        self._stop_event.set()

        # Unblock consumer immediately (and avoid deadlock if queue is full).
        with contextlib.suppress(Exception):
            self._force_put(_STOP, clear=True)

        # Shut down the device-side prefetch thread, which owns its own
        # stop event unreachable from here otherwise.
        dev_it, self.dev_it = self.dev_it, None
        close_dev_it = getattr(dev_it, "close", None)
        if close_dev_it is not None:
            with contextlib.suppress(Exception):
                close_dev_it()

        # Cancel the asyncio producer task thread-safely.
        loop, task = self._loop_ref, self._task_ref
        if loop is not None and task is not None:

            def _cancel_task():
                if not task.done():
                    task.cancel()

            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(_cancel_task)

        th = self._producer_th
        if th.is_alive() and threading.current_thread() is not th:
            with contextlib.suppress(Exception):
                th.join(timeout=1.0)

        # Don't wait: prevents hanging on long-running host transforms / dataset.
        with contextlib.suppress(Exception):
            self._executor.shutdown(wait=False, cancel_futures=True)


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
    sharding          : jax.sharding.Sharding | PyTree[jax.sharding.Sharding] | None
        Optional explicit sharding tree to apply when moving batches to device.
        Must match the batch pytree structure. Mutually exclusive with
        shard=True and mesh/batch_spec.
    mesh              : jax.sharding.Mesh | None
        Mesh used to build a NamedSharding when batch_spec is provided.
    batch_spec        : jax.sharding.PartitionSpec | PyTree[jax.sharding.PartitionSpec] | None
        Explicit pytree of PartitionSpecs for the batch when mesh is provided.
        This must match the batch pytree structure; leaf specs are not inferred.
    loop              : bool
        Endless stream (reshuffled epochs) if True; a single epoch, then
        stop, if False.
    min_fill          : float in (0, 1]
        Queue-refill threshold. When the host queue (and the sharded device
        prefetch queue) holds fewer than ``min_fill`` of its capacity, the
        batch just served is re-queued so the consumer never starves. This
        recycling repeats batches, so it only applies when ``loop=True``; with
        ``loop=False`` it is disabled and exactly one epoch is served.
    max_in_flight      : int | None
        Number of in-flight CPU batch jobs scheduled via `run_in_executor`.
        Keeping this >1 enables actual async pipelining. Order is preserved.
        Defaults to `num_async_workers` (min 1).
    """

    # ------------------------- init ----------------------------------- #
    def __init__(
        self,
        dataset: Union[SimulationDataset, Sequence[PyTree[Array]]],
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
        sharding: Any = None,
        mesh: Mesh | None = None,
        batch_spec: Any = None,
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

        self._N, self._bsz = len(dataset), batch_size
        self._drop_last, self._loop = drop_last, loop

        # -------- transforms ---------- #
        self._device_tfns: list[Callable[[Any], Any]] = as_transform_list(
            device_transforms
        )

        self._min_fill, self._prefetch_dev = min_fill, max(1, num_prefetch_device)

        # -------- device config -------- #
        if shard and (
            sharding is not None or mesh is not None or batch_spec is not None
        ):
            raise ValueError(
                "Use either shard=True or explicit sharding/mesh, not both."
            )
        self._shard_flag = shard
        self._devices = list(devices) if devices else jax.local_devices()
        self._n_dev = len(self._devices)
        if host_device is None:
            host_device = normalize_devices(None)[0]
        self._host_device = host_device

        self._sharding = self._resolve_sharding(sharding, mesh, batch_spec)
        self._has_named_sharding = any(
            isinstance(s, NamedSharding)
            for s in jax.tree_util.tree_leaves(self._sharding)
        )

        # -------- infra ---------------- #
        self._num_async_workers = int(num_async_workers)
        self._max_in_flight = (
            max(1, self._num_async_workers)
            if max_in_flight is None
            else int(max_in_flight)
        )

        self._producer = _HostProducer(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            rng=np.random.default_rng(seed) if shuffle else None,
            loop=loop,
            host_tfns=as_transform_list(host_transforms),
            queue_size=num_prefetch_host,
            min_fill=min_fill,
            num_async_workers=self._num_async_workers,
            max_in_flight=self._max_in_flight,
            has_named_sharding=self._has_named_sharding,
            host_device=self._host_device,
        )

        self._closed = False
        self._iter_ref = None
        self._iter_token = None

        # Finalize through the producer, not the loader: threads reference the
        # producer, never the loader, so dropping the loader (once gc breaks
        # its cycle with the iterator generator) stops them promptly. The
        # finalizer also registers an atexit hook, so close() runs at
        # interpreter exit regardless.
        self._finalizer_ref = weakref.finalize(self, self._producer.close)

    @staticmethod
    def _resolve_sharding(
        sharding: Any,
        mesh: Mesh | None,
        batch_spec: Any,
    ) -> Any:
        if sharding is not None:
            if mesh is not None or batch_spec is not None:
                raise ValueError(
                    "Provide either sharding or mesh+batch_spec, not both."
                )
            return sharding
        if mesh is None and batch_spec is None:
            return None
        if mesh is None or batch_spec is None:
            raise ValueError("mesh and batch_spec must be provided together.")
        return jax.tree_util.tree_map(
            lambda spec: NamedSharding(mesh, spec), batch_spec
        )

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
        # Fail fast: surface background errors immediately instead of serving
        # recycled/prefetched stale batches while the error propagates. Close
        # first: the suspended iterator generator below is never resumed, so
        # it cannot clean up by itself.
        err = self._producer._get_worker_error()
        if err is not None:
            self.close()
            raise RuntimeError(
                f"DataLoader worker failed:\n{err.traceback_text()}"
            ) from err.exc
        self._ensure_iter()
        iter_ref = self._iter_ref
        if iter_ref is None:
            raise RuntimeError("The data loader is closed.")
        return next(iter_ref)

    def _iter_gen(self, token):
        producer = self._producer
        host_it = producer._host_iter()
        if self._shard_flag:
            host_it = (_shard(b, self._n_dev) for b in host_it)

        if self._sharding is not None:
            dev_it = _prefetch_sharding(
                host_it,
                self._prefetch_dev,
                self._sharding,
                # Recycling only suits endless streams (see `min_fill`).
                min_fill=self._min_fill if self._loop else 0.0,
            )
        else:
            dev_it = (
                _prefetch_single(host_it, self._prefetch_dev, self._devices[0])
                if self._n_dev == 1 and not self._shard_flag
                else prefetch_to_device(host_it, self._prefetch_dev, self._devices)
            )

        # Track the device iterator on the producer so close() can shut down its
        # background thread directly, even if this generator is stuck in a
        # reference cycle and its finally block never runs in time.
        producer.dev_it = dev_it

        # generator wrapper ensures close() runs on exception unwind (CPython refcount)
        def gen():
            try:
                for batch in dev_it:
                    for fn in self._device_tfns:
                        batch = fn(batch)
                    yield batch
            finally:
                close_dev_it = getattr(dev_it, "close", None)
                if close_dev_it is not None:
                    with contextlib.suppress(Exception):
                        close_dev_it()
                if producer.dev_it is dev_it:
                    producer.dev_it = None
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
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._iter_ref = None
        self._iter_token = None
        self._producer.close()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
