# indexed_async_dataloader_v6.py
import asyncio
import collections
import contextlib
import itertools
import math
import queue
import threading
import time
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import prefetch_to_device
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding

from probjax.utils.typing import Device, RngKey

if TYPE_CHECKING:
    pass


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

    if isinstance(sharding, NamedSharding):
        base_spec = getattr(
            sharding, "spec", getattr(sharding, "partition_spec", PartitionSpec())
        )
        base_spec_entries = tuple(base_spec)
        sharding_by_ndim: dict[int, NamedSharding] = {}

        def _put(x):
            ndim = np.ndim(x)
            leaf_sharding = sharding_by_ndim.get(ndim)
            if leaf_sharding is None:
                if ndim <= len(base_spec_entries):
                    leaf_spec = PartitionSpec(*base_spec_entries[:ndim])
                else:
                    leaf_spec = PartitionSpec(
                        *base_spec_entries,
                        *([None] * (ndim - len(base_spec_entries))),
                    )
                leaf_sharding = NamedSharding(sharding.mesh, leaf_spec)
                sharding_by_ndim[ndim] = leaf_sharding
            return jax.device_put(x, leaf_sharding)

    else:
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


# Datasets


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
        if simulation_devices is None:
            try:
                cpu_devices = jax.devices("cpu")
            except Exception:
                cpu_devices = []
            simulation_devices = cpu_devices[0] if cpu_devices else jax.devices()[0]

        if isinstance(simulation_devices, (list, tuple)):
            self._simulation_devices = list(simulation_devices)
        else:
            self._simulation_devices = [simulation_devices]

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

        self._initial_rng = rng
        self._sim_rng = jax.random.PRNGKey(0) if isinstance(rng, int) else rng
        # Keep RNG state on the simulation device so split/dispatch follows it.
        self._sim_rng = self._pin_sim_key(self._sim_rng)

        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

        self._batched_simulator = self._build_batched_simulator(jit_simulator)
        self._producer: threading.Thread | None = None

        self._buffer: Any | None = None
        self._buffer_leaves: Sequence[np.ndarray] = ()
        self._tree_def = None
        self._write_ptr = 0
        self._pending_refresh = 0

        self._stats = {
            "batches_produced": 0,
            "production_time": 0.0,
            "samples_written": 0,
            "batches_requested": 0,
            "samples_requested": 0,
        }

        self._initialise_buffer()
        self._start_producer()

    def __del__(self) -> None:
        """Cleanup: stop producer thread when object is deleted."""
        try:
            self.close()
        except Exception:
            # Suppress exceptions during cleanup to avoid errors in __del__
            pass

    def __len__(self) -> int:
        return self._dataset_size

    def __getitem__(self, index: Any) -> Any:
        idxs, squeeze = self._normalise_indices(index)
        with self._lock:
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
            if seed is None:
                rng = self._initial_rng
            else:
                rng = jax.random.PRNGKey(int(seed))
        else:
            rng = jax.random.PRNGKey(int(rng)) if isinstance(rng, int) else rng
        self._initial_rng = rng
        self._sim_rng = jax.random.PRNGKey(0) if isinstance(rng, int) else rng
        self._sim_rng = self._pin_sim_key(self._sim_rng)
        self._stop_producer()
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

    def set_data(self, data: Any) -> None:
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
            })

        # Optionally restart the producer (kept off by default for a fixed dataset)
        self._start_producer()

    # --- internal helpers ------------------------------------------------------------

    def _initialise_buffer(self) -> None:
        self._stop_event.clear()
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
            self._stats["batches_produced"] += 1
            self._stats["production_time"] += duration
            self._stats["samples_written"] += self._batch_size
            # Request refresh of entire buffer so producer starts working immediately
            self._pending_refresh = self._dataset_size

        with self._condition:
            self._condition.notify_all()

    def _start_producer(self) -> None:
        if self._producer is not None and self._producer.is_alive():
            return
        self._stop_event.clear()
        self._producer = threading.Thread(target=self._producer_main, daemon=True)
        self._producer.start()

    def _stop_producer(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._producer is not None and self._producer.is_alive():
            self._producer.join(timeout=1.0)
        self._producer = None

    def _producer_main(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while (
                    not self._stop_event.is_set()
                    and self._pending_refresh < self._batch_size
                ):
                    self._condition.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                self._pending_refresh -= self._batch_size
            try:
                batch, duration = self._produce_batch()
            except Exception:
                self._stop_event.set()
                raise

            batch_host = self._to_host(batch)
            with self._lock:
                if self._buffer is None:
                    continue
                self._write_batch(batch_host)
                self._stats["batches_produced"] += 1
                self._stats["production_time"] += duration

    def _produce_batch(self) -> tuple[Any, float]:
        self._sim_rng = self._pin_sim_key(self._sim_rng)
        self._sim_rng, batch_key = jax.random.split(self._sim_rng)
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
        def single_call(key: RngKey) -> Any:
            return self._simulator_fn(key)

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

    def _pin_sim_key(self, key: RngKey) -> RngKey:
        return jax.device_put(key, self._simulation_device)

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
        PartitionSpec for the batch when mesh is provided. For PyTree leaves with
        different rank, the spec is adapted per leaf: shorter leaves use the first
        ``leaf.ndim`` entries, and longer leaves append replicated dimensions
        (``None`` entries).
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
                raise ValueError(
                    "Provide either sharding or mesh+batch_spec, not both."
                )
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
