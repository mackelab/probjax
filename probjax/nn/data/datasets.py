import contextlib
import math
import threading
import time
import weakref
from typing import Any, Callable, Sequence, Union

import jax
import numpy as np

from probjax.nn.data.prefetch import _WorkerError
from probjax.utils.guards import normalize_devices, normalize_rng
from probjax.utils.typing import Array, Device, PyTree, RngKey

# Datasets


def _simulation_producer(
    self_ref: "weakref.ReferenceType[SimulationDataset]",
    stop_event: threading.Event,
    condition: threading.Condition,
) -> None:
    """Thread body of `SimulationDataset`'s producer.

    Module-level and given only a weak reference: the dataset is dereferenced
    for one unit of work at a time and never held across a wait, so a dropped
    dataset is collected (its ``__del__`` then stops this thread) instead of
    being kept alive by its own producer.
    """
    while True:
        with condition:
            while True:
                if stop_event.is_set():
                    return
                ds = self_ref()
                if ds is None:
                    return
                if ds._pending_refresh >= ds._batch_size:
                    ds._pending_refresh -= ds._batch_size
                    break
                del ds  # don't keep the dataset alive while waiting
                condition.wait(timeout=0.1)
        keep_going = ds._produce_one(stop_event)
        del ds
        if not keep_going:
            return


class SimulationDataset:
    """Fixed-size dataset whose samples are refreshed asynchronously.

    Parameters
    ----------
    simulator_fn : Callable
        A function that takes a RNG key and returns a simulation output.
    simulation_batch_size : int
        Number of simulations to run per batch.
    rng : RngKey
        JAX random key for reproducibility.
    simulation_devices : Device | Sequence[Device]
        Device(s) to run simulations on. If multiple devices are provided,
        simulations are parallelized across them using pmap.
        Defaults to CPU when available.
    simulation_batch_mode : {"vmap", "map"}
        Batch execution mode for simulations on each device.
        ``"vmap"`` uses vectorized execution; ``"map"`` uses ``jax.lax.map``.
    jit_simulator : bool
        Whether to JIT compile the simulator. Default True.
    buffer_size : int
        Size of the data buffer. Default 8192.
    """

    def __init__(
        self,
        simulator_fn: Callable[..., Any],
        *,
        simulation_batch_size: int = 128,
        rng: RngKey,
        simulation_devices: Union[Device, Sequence[Device]] = None,
        simulation_batch_mode: str = "vmap",
        jit_simulator: bool = True,
        buffer_size: int = 8192,
    ) -> None:
        self._simulation_devices = normalize_devices(simulation_devices)

        self._n_sim_devices = len(self._simulation_devices)
        self._simulation_device = self._simulation_devices[0]
        self._simulation_backend = self._simulation_device.platform

        self._simulator_fn = simulator_fn
        self._batch_size = int(simulation_batch_size)
        self._simulation_batch_mode = str(simulation_batch_mode).lower()
        if self._simulation_batch_mode not in {"vmap", "map"}:
            raise ValueError("simulation_batch_mode must be one of {'vmap', 'map'}.")

        if self._batch_size % self._n_sim_devices != 0:
            raise ValueError(
                f"simulation_batch_size ({self._batch_size}) must be divisible "
                f"by the number of simulation devices ({self._n_sim_devices})"
            )
        self._batch_size_per_device = self._batch_size // self._n_sim_devices

        # Round buffer up to a whole number of batches for clean ring writes.
        self._buffer_batches = max(1, math.ceil(int(buffer_size) / self._batch_size))
        self._dataset_size = self._buffer_batches * self._batch_size

        self._initial_rng = normalize_rng(rng)
        self._sim_rng = self._initial_rng
        # Keep RNG state on the simulation device so split/dispatch follows it.
        self._sim_rng = self._pin_sim_key(self._sim_rng)

        # One event per producer generation (see `_start_producer`); an event
        # that was set is never cleared, so a slow old producer cannot be
        # revived by a restart.
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        # Guards `_sim_rng`, which the producer and reset() both advance.
        self._rng_lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        self._batched_simulator = self._build_batched_simulator(jit_simulator)
        self._producer: threading.Thread | None = None

        self._buffer: Any | None = None
        self._buffer_leaves: Sequence[np.ndarray] = ()
        self._tree_def = None
        self._write_ptr = 0
        self._pending_refresh = 0
        # Background producer failure. Served stale data forever if ignored,
        # so __getitem__ re-raises it (fail fast) instead.
        self._producer_error: _WorkerError | None = None

        self._stats = {
            "batches_produced": 0,
            "production_time": 0.0,
            "samples_written": 0,
            "batches_requested": 0,
            "samples_requested": 0,
            "batches_consumed": 0,
            "samples_served": 0,
        }

        self._initialise_buffer()
        self._start_producer()

    def __del__(self) -> None:
        """Cleanup: stop producer thread when object is deleted."""
        # Suppress exceptions during cleanup to avoid errors in __del__.
        with contextlib.suppress(Exception):
            self.close()

    def __len__(self) -> int:
        return self._dataset_size

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> PyTree[Array]:
        idxs, squeeze = self._normalise_indices(index)
        with self._lock:
            err = self._producer_error
            if err is not None:
                raise RuntimeError(
                    f"SimulationDataset producer failed:\n{err.traceback_text()}"
                ) from err.exc
            if self._buffer is None:
                raise RuntimeError("Simulation buffer not initialised.")
            leaves = [leaf[idxs] for leaf in self._buffer_leaves]
            self._stats["batches_requested"] += 1
            self._stats["samples_requested"] += idxs.shape[0]
        batch = jax.tree_util.tree_unflatten(self._tree_def, leaves)
        if squeeze:
            batch = jax.tree_util.tree_map(lambda x: x[0], batch)
        batch = self._convert_for_consumer(batch)
        self._request_refresh(idxs.shape[0])
        return batch

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
            batches = max(1, int(stats.get("batches_produced", 0)))
            stats["avg_production_time_per_batch"] = stats["production_time"] / batches
            return stats

    def reset(self, *, seed: int | None = None, rng: RngKey | None = None) -> None:
        if rng is not None and seed is not None:
            raise ValueError("Provide either rng or seed, not both.")
        if rng is None:
            rng = self._initial_rng if seed is None else jax.random.PRNGKey(int(seed))
        else:
            rng = normalize_rng(rng)
        # Stop the producer first: a producer finishing a batch must not
        # advance (and thereby clobber) the rng requested here.
        self._stop_producer()
        self._initial_rng = normalize_rng(rng)
        with self._rng_lock:
            self._sim_rng = self._pin_sim_key(self._initial_rng)
        with self._lock:
            self._pending_refresh = 0
        self._initialise_buffer()
        self._start_producer()

    def close(self) -> None:
        """Stop the producer thread and clean up resources."""
        self._stop_producer()
        with self._lock:
            self._buffer = None
            self._buffer_leaves = ()
            self._tree_def = None

    def set_data(self, data: Any, *, start_producer: bool = False) -> None:
        """Replace the internal buffer with user-provided data.

        Parameters
        ----------
        data : Any
            A PyTree of arrays with a leading sample dimension. Leaves must be
            array-like and broadcast-consistent in their first dimension.
        start_producer : bool, default False
            If True, (re)start the background producer after setting the buffer.
            By default we keep the dataset static and the producer stopped.
        """
        # Stop producer while we mutate the buffer
        self._stop_producer()

        # Convert to host NumPy, validate tree & batch dimension
        data_host = jax.tree_util.tree_map(
            lambda x: np.asarray(jax.device_get(x)), data
        )
        leaves = jax.tree_util.tree_leaves(data_host)
        if not leaves:
            raise ValueError("set_data: Provided data has no leaves.")

        # Infer N (samples) and validate consistent leading dim
        try:
            N = int(leaves[0].shape[0])
        except Exception as e:
            raise ValueError("set_data: Could not infer leading dimension.") from e
        for i, lf in enumerate(leaves[1:], start=1):
            if lf.shape[0] != N:
                raise ValueError(
                    f"set_data: Leaf 0 has N={N} but leaf {i} has N={lf.shape[0]}."
                )

        if N <= 0:
            raise ValueError("set_data: Need at least one sample.")

        # Round up to a whole number of batches
        batch_size = self._batch_size
        buffer_batches = max(1, math.ceil(N / batch_size))
        dataset_size = buffer_batches * batch_size

        # Build new buffer with rounded size and copy data (pad by wrap if needed)
        tree_def = jax.tree_util.tree_structure(data_host)
        new_buffer = jax.tree_util.tree_map(
            lambda x: np.empty((dataset_size,) + x.shape[1:], dtype=x.dtype),
            data_host,
        )
        new_buffer_leaves = jax.tree_util.tree_leaves(new_buffer)

        if dataset_size == N:
            # Exact fit
            for buf_leaf, data_leaf in zip(new_buffer_leaves, leaves, strict=True):
                buf_leaf[:] = data_leaf
        else:
            # Copy the N samples, then pad by wrapping from the start
            for buf_leaf, data_leaf in zip(new_buffer_leaves, leaves, strict=True):
                buf_leaf[:N] = data_leaf
                remaining = dataset_size - N
                if remaining > 0:
                    # Wrap (repeat from the start) to fill the last partial batch
                    wrap_src = (
                        data_leaf[: remaining % N if N != 0 else 0]
                        if remaining > N
                        else data_leaf[:remaining]
                    )
                    # If remaining > N, tile then slice (avoids large loops)
                    if remaining > N:
                        reps = (remaining + N - 1) // N
                        tiled = np.concatenate([data_leaf] * reps, axis=0)
                        buf_leaf[N:] = tiled[:remaining]
                    else:
                        buf_leaf[N:] = wrap_src

        # Reset internal state and stats
        with self._lock:
            self._buffer = new_buffer
            self._buffer_leaves = new_buffer_leaves
            self._tree_def = tree_def
            self._write_ptr = 0
            self._pending_refresh = 0
            self._producer_error = None

            # Update size bookkeeping to match the new buffer
            self._buffer_batches = buffer_batches
            self._dataset_size = dataset_size

            # Reset (only) counters that logically depend on production
            self._stats.update({
                "batches_produced": 0,
                "production_time": 0.0,
                "samples_written": dataset_size,
                "batches_requested": 0,
                "samples_requested": 0,
                "batches_consumed": 0,
                "samples_served": 0,
            })

        # Optionally restart the producer (kept off by default for a fixed dataset)
        if start_producer:
            self._start_producer()

    # --- internal helpers ------------------------------------------------------------

    def _initialise_buffer(self) -> None:
        self._write_ptr = 0

        batch, duration = self._produce_batch()
        batch_host = self._to_host(batch)
        batch_leaves = jax.tree_util.tree_leaves(batch_host)

        tree_def = jax.tree_util.tree_structure(batch_host)
        buffer = jax.tree_util.tree_map(
            lambda x: np.empty(
                (self._dataset_size,) + np.asarray(x).shape[1:],
                dtype=np.asarray(x).dtype,
            ),
            batch_host,
        )
        buffer_leaves = jax.tree_util.tree_leaves(buffer)

        if not buffer_leaves:
            raise RuntimeError("Simulator returned an empty batch.")

        for buf_leaf, data_leaf in zip(buffer_leaves, batch_leaves, strict=True):
            reshaped = buf_leaf.reshape(
                (self._buffer_batches, self._batch_size) + data_leaf.shape[1:]
            )
            reshaped[:] = data_leaf

        with self._lock:
            self._buffer = buffer
            self._buffer_leaves = buffer_leaves
            self._tree_def = tree_def
            self._producer_error = None
            self._stats["batches_produced"] += 1
            self._stats["production_time"] += duration
            self._stats["samples_written"] += self._batch_size
            # Request refresh of entire buffer so producer starts working immediately
            self._pending_refresh = self._dataset_size

        with self._condition:
            self._condition.notify_all()

    def _start_producer(self) -> None:
        if (
            self._producer is not None
            and self._producer.is_alive()
            and not self._stop_event.is_set()
        ):
            return
        # Fresh event per generation: the thread only ever sees its own.
        self._stop_event = threading.Event()
        self._producer = threading.Thread(
            target=_simulation_producer,
            args=(weakref.ref(self), self._stop_event, self._condition),
            daemon=True,
        )
        self._producer.start()

    def _stop_producer(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        producer = self._producer
        if (
            producer is not None
            and producer.is_alive()
            and threading.current_thread() is not producer
        ):
            producer.join(timeout=1.0)
        # A producer that outlives the join (stuck in a slow simulator) is
        # stale: its event is set, so it exits without touching the buffer.
        self._producer = None

    def _produce_one(self, stop_event: threading.Event) -> bool:
        """Simulate one batch and write it. False if the producer should exit."""
        try:
            produced = self._produce_batch(stop_event)
        except Exception as exc:
            # Record the failure so the next __getitem__ raises instead of
            # serving stale data forever, then exit the thread cleanly.
            # NOTE: no traceback formatting here (see _WorkerError).
            with self._lock:
                if self._producer_error is None and not stop_event.is_set():
                    self._producer_error = _WorkerError(exc)
            stop_event.set()
            return False
        if produced is None:
            return False
        batch, duration = produced

        batch_host = self._to_host(batch)
        with self._lock:
            # Stale producer (reset/set_data/close ran meanwhile): drop it.
            if stop_event.is_set():
                return False
            if self._buffer is None:
                return True
            self._write_batch(batch_host)
            self._stats["batches_produced"] += 1
            self._stats["production_time"] += duration
        return True

    def _produce_batch(
        self, stop_event: threading.Event | None = None
    ) -> tuple[Any, float] | None:
        """Simulate one batch; None if `stop_event` is set (stale producer)."""
        with self._rng_lock:
            if stop_event is not None and stop_event.is_set():
                return None
            rng = self._pin_sim_key(self._sim_rng)
            self._sim_rng, batch_key = jax.random.split(rng)
        sample_keys = jax.random.split(batch_key, self._batch_size)
        if self._n_sim_devices > 1:
            keys_per_device = sample_keys.reshape(
                (self._n_sim_devices, self._batch_size_per_device)
                + sample_keys.shape[1:]
            )
        else:
            keys_per_device = jax.device_put(sample_keys, self._simulation_device)
        start = time.perf_counter()

        if self._n_sim_devices > 1:
            batch = self._batched_simulator(keys_per_device)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((self._batch_size,) + x.shape[2:]), batch
            )
        else:
            batch = self._batched_simulator(keys_per_device)

        duration = time.perf_counter() - start
        return batch, duration

    def _build_batched_simulator(self, jit_simulator: bool) -> Callable[[Any], Any]:
        def single_call(keys: RngKey) -> Any:
            return self._simulator_fn(keys)

        if self._simulation_batch_mode == "map":

            def per_device_batched(keys):
                return jax.lax.map(single_call, keys)

        else:
            per_device_batched = jax.vmap(single_call)

        if self._n_sim_devices > 1:
            batched = jax.pmap(
                per_device_batched,
                devices=self._simulation_devices,
            )
        else:
            batched = per_device_batched

        if jit_simulator:
            if self._n_sim_devices > 1:
                return jax.jit(batched, backend=self._simulation_backend)
            return jax.jit(batched, device=self._simulation_device)
        return batched

    def _pin_sim_key(self, rng) -> RngKey:
        return jax.device_put(rng, self._simulation_device)

    def _to_host(self, batch: Any) -> Any:
        return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), batch)

    def _write_batch(self, batch_host: Any) -> None:
        if self._buffer is None:
            raise RuntimeError("Simulation buffer not initialised.")
        indices = self._reserve_indices(self._batch_size)
        batch_leaves = jax.tree_util.tree_leaves(batch_host)
        for buf_leaf, data_leaf in zip(self._buffer_leaves, batch_leaves, strict=True):
            buf_leaf[indices] = data_leaf
        self._stats["samples_written"] += len(indices)

    def _reserve_indices(self, count: int) -> np.ndarray:
        start = self._write_ptr
        end = (start + count) % self._dataset_size
        if count <= 0:
            return np.empty((0,), dtype=np.int64)
        if start < end or end == 0:
            idxs = np.arange(start, start + count, dtype=np.int64) % self._dataset_size
        else:
            first = np.arange(start, self._dataset_size, dtype=np.int64)
            second = np.arange(0, end, dtype=np.int64)
            idxs = np.concatenate([first, second])
        self._write_ptr = end
        return idxs

    def _convert_for_consumer(self, batch: Any) -> Any:
        return batch

    def _normalise_indices(self, index: Any) -> tuple[np.ndarray, bool]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._dataset_size)
            idxs = np.arange(start, stop, step, dtype=np.int64)
            squeeze = False
        elif isinstance(index, (list, tuple, np.ndarray)):
            arr = np.asarray(index, dtype=np.int64)
            idxs = np.mod(arr, self._dataset_size)
            squeeze = False
        else:
            idx = int(index)
            idxs = np.array([(idx % self._dataset_size)], dtype=np.int64)
            squeeze = True
        return idxs, squeeze

    def _request_refresh(self, count: int) -> None:
        if count <= 0:
            return
        with self._condition:
            self._pending_refresh += count
            self._condition.notify_all()

    def mark_batch_served(self, sample_count: int) -> None:
        with self._lock:
            self._stats["batches_consumed"] += 1
            self._stats["samples_served"] += int(sample_count)
