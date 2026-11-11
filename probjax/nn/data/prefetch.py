import collections
import contextlib
import itertools
import queue
import threading
import time
import traceback
from typing import Callable, cast

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from probjax.utils.typing import Array, PyTree


class _WorkerError:
    """A background-thread failure.

    The exception is captured immediately (cheap) but the traceback text is
    formatted lazily on first use: formatting walks the stack with a cold
    linecache and can take tens of milliseconds, which must never block the
    producer from signalling the error and stopping.
    """

    __slots__ = ("exc", "tb")

    def __init__(self, exc: BaseException):
        self.exc = exc
        self.tb: str | None = None

    def traceback_text(self) -> str:
        tb = self.tb
        if tb is None:
            tb = "".join(
                traceback.format_exception(
                    type(self.exc), self.exc, self.exc.__traceback__
                )
            )
            self.tb = tb
        return tb


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


def _force_put(q: queue.Queue, item) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                q.get_nowait()


def _prefetch_worker(
    source, put_fn, q: queue.Queue, stop_event, sentinel, error: list
) -> None:
    """Thread body of `_AsyncPrefetchIterator`.

    A module-level function that receives only what it needs (never the
    iterator object), so a dropped iterator can be collected and its
    ``__del__`` can stop this thread.
    """
    try:
        for item in source:
            if stop_event.is_set():
                break
            prefetched = put_fn(item)
            # Keep device placement asynchronous so prefetching can overlap
            # with consumer work instead of synchronizing on every batch.
            while not stop_event.is_set():
                try:
                    # Coarse poll: close() unblocks via _force_put.
                    q.put(prefetched, timeout=0.05)
                    break
                except queue.Full:
                    continue
    except BaseException as exc:
        # NOTE: no traceback formatting here (see _WorkerError); record
        # immediately so the consumer fails fast.
        error[0] = _WorkerError(exc)
    finally:
        while True:
            try:
                q.put(sentinel, timeout=0.05)
                break
            except queue.Full:
                if stop_event.is_set():
                    _force_put(q, sentinel)
                    break


class _AsyncPrefetchIterator:
    def __init__(
        self,
        iterator,
        size: int,
        put_fn: Callable[[PyTree[Array]], PyTree[Array]],
        *,
        queue_size: int | None = None,
        min_fill: float = 0.5,
    ):
        maxsize = max(1, queue_size or size)
        self._queue = queue.Queue(maxsize)
        self._min_size = int(maxsize * min_fill) if maxsize > 0 else 0
        self._stop_event = threading.Event()
        self._sentinel = object()
        self._error_box: list[_WorkerError | None] = [None]
        self._done = False
        self._thread = threading.Thread(
            target=_prefetch_worker,
            args=(
                iter(iterator),
                put_fn,
                self._queue,
                self._stop_event,
                self._sentinel,
                self._error_box,
            ),
            daemon=True,
        )
        self._thread.start()

    @property
    def _error(self) -> _WorkerError | None:
        return self._error_box[0]

    def _raise_if_worker_failed(self) -> None:
        err = self._error
        if err is not None:
            raise RuntimeError(
                f"Device prefetch worker failed:\n{err.traceback_text()}"
            ) from err.exc

    def __iter__(self):
        return self

    def __next__(self):
        # Once exhausted or closed there is no producer left to put anything:
        # a further queue.get() would block forever.
        if self._done:
            raise StopIteration
        # Fail fast: surface background errors immediately instead of serving
        # recycled batches while the sentinel propagates.
        self._raise_if_worker_failed()
        item = self._queue.get()
        if item is self._sentinel:
            self._done = True
            self.close()
            self._raise_if_worker_failed()
            raise StopIteration
        recycled = False
        while (
            self._min_size > 0
            and self._queue.qsize() < self._min_size
            and not self._queue.full()
        ):
            try:
                self._queue.put_nowait(item)
                recycled = True
            except queue.Full:
                break
        if recycled:
            # The producer is lagging (or dead and the error is still being
            # recorded). Yield the GIL so it gets a timeslice to catch up or
            # deliver its error instead of being starved by this hot loop.
            time.sleep(0)
        return item

    def close(self) -> None:
        self._done = True
        self._stop_event.set()
        _force_put(self._queue, self._sentinel)
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()


def _prefetch_sharding(iterator, size, sharding, min_fill: float = 0.5):
    """Device-prefetch with `sharding`. ``min_fill=0`` disables recycling
    (required for finite streams, where a recycled batch is a duplicate)."""

    def _put(x, leaf_sharding):
        if isinstance(leaf_sharding, NamedSharding):
            spec = getattr(
                leaf_sharding,
                "spec",
                getattr(leaf_sharding, "partition_spec", PartitionSpec()),
            )
            return multihost_utils.host_local_array_to_global_array(
                np.asarray(x), cast(Mesh, leaf_sharding.mesh), spec
            )
        return jax.device_put(x, leaf_sharding)

    def _put_ready(data):
        prefetched = jax.tree_util.tree_map(_put, data, sharding)
        return jax.block_until_ready(prefetched)

    return _AsyncPrefetchIterator(
        iterator,
        size,
        _put_ready,
        # Keep a little extra headroom so the consumer is less likely to see
        # the sharding worker mid-transfer when batches are expensive.
        queue_size=max(2, size * 2),
        min_fill=min_fill,
    )
