# indexed_async_dataloader_v6.py
import asyncio, itertools, queue, threading, collections, weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator, Optional, Sequence, Any, Callable, Union

import numpy as np
import jax, jax.numpy as jnp
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
    _fill = lambda n: [dq.append(jax.tree_util.tree_map(_put, d)) for d in itertools.islice(iterator, n)]
    _fill(size)
    while dq:
        yield dq.popleft()
        _fill(1)
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
        dataset: Iterable[Any],
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
        shard: bool = True,
        devices: Optional[Sequence[jax.Device]] = None,
        num_async_workers: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not (0.0 < min_fill <= 1.0):
            raise ValueError("min_fill must be in (0,1].")

        self._ds, self._N, self._bsz = dataset, len(dataset), batch_size
        self._drop_last, self._loop = drop_last, loop

        self._rng = (
            np.random.default_rng(seed) if shuffle else None
        )

        # -------- transforms ---------- #
        self._host_tfns = (
            list(host_transforms) if isinstance(host_transforms, (list, tuple))
            else ([host_transforms] if host_transforms else [])
        )
        self._device_tfns = (
            list(device_transforms) if isinstance(device_transforms, (list, tuple))
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

    # ---------------- background producer ----------------------------- #
    def _producer_main(self):
        asyncio.run(self._fill_queue())

    async def _fill_queue(self):
        try:
            for idxs in self._index_batches():
                if self._stop_event.is_set():
                    break
                fut = asyncio.get_running_loop().run_in_executor(
                    self._executor, self._fetch_batch, idxs
                )
                batch = await fut
                self._q.put(batch)  # blocks if queue full
            self._q.put(None)
        except Exception:
            self._q.put(None)
            raise

    def _fetch_batch(self, idxs):
        batch = self._ds[idxs]
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
                try: self._q.put_nowait(batch)
                except queue.Full: break
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
        return (self._N // self._bsz) if self._drop_last else (self._N + self._bsz - 1) // self._bsz

    # ---------------- clean-up / context manager ----------------------- #
    def close(self):
        if self._closed: return
        self._closed = True
        self._stop_event.set()
        self._q.put(None)
        if self._producer_th.is_alive():
            self._producer_th.join(timeout=1.0)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _finalizer(self):
        try: self.close()
        except Exception: pass

    def __del__(self): self._finalizer()
    def __enter__(self): return self
    def __exit__(self, *exc): self.close(); return False
