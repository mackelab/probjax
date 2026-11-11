import gc
import threading
import time
import weakref

import jax
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from probjax.nn import DataLoader
from probjax.nn.data import compose, datasets, prefetch


def _sharding_spec_of(array):
    sharding = array.sharding
    return getattr(sharding, "spec", getattr(sharding, "partition_spec", P()))


def _require_two_gpus():
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("test requires at least 2 GPU devices")
    if len(gpu_devices) < 2:
        pytest.skip("test requires at least 2 GPU devices")
    return gpu_devices[:2]


class _TreeDataset:
    def __init__(self, n_samples=8):
        self._tokens = np.arange(n_samples * 3 * 2).reshape(n_samples, 3, 2)
        self._features = np.arange(n_samples * 4).reshape(n_samples, 4)
        self._labels = np.arange(n_samples)
        self._extra = np.arange(n_samples * 2 * 2 * 2).reshape(n_samples, 2, 2, 2)
        self._constant = np.asarray(3.14)

    def __len__(self):
        return self._tokens.shape[0]

    def __getitem__(self, idxs):
        return {
            "tokens": self._tokens[idxs],
            "features": self._features[idxs],
            "labels": self._labels[idxs],
            "extra": self._extra[idxs],
            "constant": self._constant,
        }


class _IndexDataset:
    def __init__(self, n_samples=6):
        self._values = np.arange(n_samples, dtype=np.int32)

    def __len__(self):
        return self._values.shape[0]

    def __getitem__(self, idxs):
        return self._values[idxs]


class _FeatureDataset:
    def __init__(self, n_samples=4096, width=4096):
        self._values = np.arange(n_samples * width, dtype=np.float32).reshape(
            n_samples, width
        )

    def __len__(self):
        return self._values.shape[0]

    def __getitem__(self, idxs):
        return self._values[idxs]


class _FeatureTreeDataset:
    def __init__(self, n_samples=4096, width=4096):
        self._features = np.arange(n_samples * width, dtype=np.float32).reshape(
            n_samples, width
        )
        self._aux = np.arange(n_samples * 512, dtype=np.float32).reshape(n_samples, 512)
        self._labels = np.arange(n_samples, dtype=np.int32)

    def __len__(self):
        return self._features.shape[0]

    def __getitem__(self, idxs):
        return {
            "features": self._features[idxs],
            "aux": self._aux[idxs],
            "labels": self._labels[idxs],
        }


def test_dataloader_mesh_batch_spec_uses_explicit_spec_tree():
    mesh = jax.make_mesh((1,), ("data",), devices=jax.devices()[:1])
    dataset = _TreeDataset()
    batch_spec = {
        "tokens": P("data", None, None),
        "features": P("data", None),
        "labels": P("data"),
        "extra": P("data", None, None, None),
        "constant": P(),
    }

    with DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        loop=False,
        num_prefetch_host=2,
        num_prefetch_device=1,
        mesh=mesh,
        batch_spec=batch_spec,
    ) as loader:
        batch = next(iter(loader))

    assert _sharding_spec_of(batch["tokens"]) == P("data", None, None)
    assert _sharding_spec_of(batch["features"]) == P("data", None)
    assert _sharding_spec_of(batch["labels"]) == P("data")
    assert _sharding_spec_of(batch["extra"]) == P("data", None, None, None)
    assert _sharding_spec_of(batch["constant"]) == P()


def test_dataloader_mesh_sharding_keeps_batches_host_local_in_worker():
    mesh = jax.make_mesh((1,), ("data",), devices=jax.devices()[:1])
    dataset = _IndexDataset()

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        loop=False,
        num_prefetch_host=2,
        num_prefetch_device=1,
        mesh=mesh,
        batch_spec=P("data"),
    )
    try:
        batch = loader._producer._process_batch(np.array([0, 1]))
    finally:
        loader.close()

    assert isinstance(batch, np.ndarray)


def test_prefetch_sharding_uses_host_local_array_to_global_array(monkeypatch):
    mesh = jax.make_mesh((1,), ("data",), devices=jax.devices()[:1])
    sharding = {"x": jax.sharding.NamedSharding(mesh, P("data", None))}
    calls = []

    def fake_host_local_array_to_global_array(x, global_mesh, pspecs):
        calls.append((np.asarray(x).copy(), global_mesh, pspecs))
        return ("global", np.asarray(x).shape, pspecs)

    def fail_device_put(x, device=None, **kwargs):
        if isinstance(device, jax.sharding.NamedSharding):
            raise AssertionError(
                "NamedSharding should use host_local_array_to_global_array"
            )
        return x

    monkeypatch.setattr(
        prefetch.multihost_utils,
        "host_local_array_to_global_array",
        fake_host_local_array_to_global_array,
    )
    monkeypatch.setattr(prefetch.jax, "device_put", fail_device_put)

    out = next(
        prefetch._prefetch_sharding(
            iter([{"x": np.arange(4, dtype=np.int32).reshape(2, 2)}]),
            1,
            sharding,
        )
    )

    assert out["x"] == ("global", (2, 2), P("data", None))
    assert len(calls) == 1


def test_prefetch_sharding_serves_ready_batches(monkeypatch):
    mesh = jax.make_mesh((1,), ("data",), devices=jax.devices()[:1])
    sharding = {"x": jax.sharding.NamedSharding(mesh, P("data", None))}
    ready_calls = []

    def fake_host_local_array_to_global_array(x, global_mesh, pspecs):
        return ("global", int(np.asarray(x).reshape(-1)[0]))

    def fake_block_until_ready(x):
        ready_calls.append(x)
        return x

    monkeypatch.setattr(
        prefetch.multihost_utils,
        "host_local_array_to_global_array",
        fake_host_local_array_to_global_array,
    )
    monkeypatch.setattr(prefetch.jax, "block_until_ready", fake_block_until_ready)

    it = prefetch._prefetch_sharding(
        iter([
            {"x": np.array([[0, 1]], dtype=np.int32)},
            {"x": np.array([[2, 3]], dtype=np.int32)},
            {"x": np.array([[4, 5]], dtype=np.int32)},
        ]),
        2,
        sharding,
    )
    try:
        # All three batches and the terminal marker fit in the queue.
        it._thread.join(timeout=10)
        assert not it._thread.is_alive(), "prefetch worker did not finish"
        assert len(ready_calls) == 3
        _ = next(it)
        batch = next(it)
    finally:
        it.close()

    assert batch["x"] == ("global", 2)
    assert len(ready_calls) >= 2


def test_prefetch_sharding_recycles_ready_batches_when_worker_lags(monkeypatch):
    mesh = jax.make_mesh((1,), ("data",), devices=jax.devices()[:1])
    sharding = {"x": jax.sharding.NamedSharding(mesh, P("data", None))}
    worker_waiting = threading.Event()
    release_worker = threading.Event()
    consumer_done = threading.Event()
    results = []

    def fake_host_local_array_to_global_array(x, global_mesh, pspecs):
        return ("global", int(np.asarray(x).reshape(-1)[0]))

    def fake_block_until_ready(x):
        if x["x"] == ("global", 2):
            worker_waiting.set()
            assert release_worker.wait(10), "test did not release the worker"
        return x

    monkeypatch.setattr(
        prefetch.multihost_utils,
        "host_local_array_to_global_array",
        fake_host_local_array_to_global_array,
    )
    monkeypatch.setattr(prefetch.jax, "block_until_ready", fake_block_until_ready)

    it = prefetch._prefetch_sharding(
        iter([
            {"x": np.array([[0, 1]], dtype=np.int32)},
            {"x": np.array([[2, 3]], dtype=np.int32)},
        ]),
        1,
        sharding,
    )

    def consume():
        results.extend([next(it), next(it)])
        consumer_done.set()

    consumer = threading.Thread(target=consume, daemon=True)
    try:
        assert worker_waiting.wait(10), "worker did not reach the second transfer"
        consumer.start()
        # Both reads must finish while the second transfer is still blocked.
        # This checks independence from the worker, without sub-ms timing gates.
        assert consumer_done.wait(5), "consumer blocked on the unfinished transfer"
        assert [batch["x"] for batch in results] == [("global", 0), ("global", 0)]
        release_worker.set()
        remaining = list(it)
        assert remaining[-1]["x"] == ("global", 2)
        assert {batch["x"] for batch in remaining} <= {("global", 0), ("global", 2)}
    finally:
        release_worker.set()
        it.close()
        if consumer.ident is not None:
            consumer.join(timeout=10)


def test_dataloader_mesh_sharding_overlaps_expensive_host_transform():
    mesh = jax.make_mesh((1,), ("data",), devices=jax.devices()[:1])
    dataset = _IndexDataset(n_samples=16)
    transform_calls = []

    def slow_host_transform(batch):
        transform_calls.append(int(np.asarray(batch).reshape(-1)[0]))
        time.sleep(0.05)
        return batch

    with DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        loop=True,
        host_transforms=slow_host_transform,
        num_prefetch_host=4,
        num_prefetch_device=2,
        min_fill=0.01,
        num_async_workers=2,
        max_in_flight=2,
        mesh=mesh,
        batch_spec=P("data", None),
    ) as loader:
        it = iter(loader)
        time.sleep(0.18)
        first = next(it)
        first.block_until_ready()
        start = time.perf_counter()
        batch = next(it)
        batch.block_until_ready()
        elapsed = time.perf_counter() - start

    assert batch.shape == (2,)
    assert elapsed < 0.01
    assert len(transform_calls) >= 2


def test_dataloader_named_sharding_2gpu_grid_stays_nonblocking_with_expensive_host_transform():
    mesh = jax.make_mesh((2,), ("data",), devices=_require_two_gpus())
    dataset = _FeatureTreeDataset(n_samples=4096, width=4096)
    transform_calls = []
    measured_times = []

    def slow_host_transform(batch):
        batch = jax.tree_util.tree_map(np.asarray, batch)
        transform_calls.append(float(batch["features"].reshape(-1)[0]))
        time.sleep(0.5)
        return {
            "features": batch["features"] * 1.5,
            "aux": batch["aux"] + 2.0,
            "labels": batch["labels"] + 1,
        }

    with DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        loop=True,
        host_transforms=slow_host_transform,
        num_prefetch_host=4,
        num_prefetch_device=2,
        min_fill=0.01,
        num_async_workers=2,
        max_in_flight=2,
        mesh=mesh,
        batch_spec={
            "features": P("data", None),
            "aux": P("data", None),
            "labels": P("data"),
        },
    ) as loader:
        it = iter(loader)
        time.sleep(2.5)
        first = next(it)
        first["features"].block_until_ready()
        for _ in range(5):
            start = time.perf_counter()
            batch = next(it)
            batch["features"].block_until_ready()
            measured_times.append(time.perf_counter() - start)

    assert batch["features"].shape == (512, 4096)
    assert batch["aux"].shape == (512, 512)
    assert batch["labels"].shape == (512,)
    assert _sharding_spec_of(batch["features"]) == P("data", None)
    assert _sharding_spec_of(batch["aux"]) == P("data", None)
    assert _sharding_spec_of(batch["labels"]) == P("data")
    assert max(measured_times) < 0.005
    assert len(transform_calls) >= 6


def test_simulation_dataset_producer_failure_raises_with_cause():
    """A dead background producer must fail fast, not serve stale data forever."""
    calls = {"n": 0}

    def flaky_simulator(key):
        calls["n"] += 1
        if calls["n"] > 4:
            raise ValueError("boom from simulator")
        return {"x": jax.numpy.ones((2,))}

    ds = datasets.SimulationDataset(
        flaky_simulator,
        simulation_batch_size=4,
        rng=jax.random.PRNGKey(0),
        buffer_size=8,
        jit_simulator=False,
    )
    try:
        with pytest.raises(
            RuntimeError, match="SimulationDataset producer failed"
        ) as exc_info:
            for _ in range(200):
                ds[np.array([0, 1])]
                time.sleep(0.001)
    finally:
        ds.close()
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_simulation_dataset_recovers_after_reset():
    """reset() clears a recorded producer failure and restarts production."""
    state = {"calls": 0, "fail": True}

    def flaky_simulator(key):
        state["calls"] += 1
        if state["fail"] and state["calls"] > 4:
            raise ValueError("boom from simulator")
        return {"x": jax.numpy.ones((2,))}

    ds = datasets.SimulationDataset(
        flaky_simulator,
        simulation_batch_size=4,
        rng=jax.random.PRNGKey(0),
        buffer_size=8,
        jit_simulator=False,
    )
    try:
        with pytest.raises(RuntimeError, match="SimulationDataset producer failed"):
            for _ in range(200):
                ds[np.array([0, 1])]
                time.sleep(0.001)
        state["fail"] = False
        ds.reset()
        batch = ds[np.array([0, 1])]
        assert np.asarray(batch["x"]).shape == (2, 2)
    finally:
        ds.close()


def test_dataloader_host_transform_failure_raises_promptly():
    """Worker errors must surface within a bounded number of takes, not be
    masked indefinitely by recycled/prefetched batches."""
    calls = {"n": 0}

    def flaky_transform(batch):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ValueError("boom from host transform")
        return batch

    with DataLoader(
        _IndexDataset(n_samples=32),
        batch_size=4,
        shuffle=False,
        loop=True,
        host_transforms=flaky_transform,
        num_prefetch_host=4,
        num_prefetch_device=2,
        num_async_workers=2,
        max_in_flight=2,
    ) as loader:
        it = iter(loader)
        with pytest.raises(RuntimeError, match="DataLoader worker failed") as exc_info:
            for _ in range(20000):
                next(it)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_dataloader_loop_false_exhaustion_yields_exact_epoch():
    """loop=False must terminate with exactly one epoch when recycling is off
    (num_prefetch_host=1 rounds the min_fill threshold to 0)."""
    with DataLoader(
        _IndexDataset(n_samples=8),
        batch_size=4,
        shuffle=False,
        loop=False,
        num_prefetch_host=1,
        num_prefetch_device=1,
    ) as loader:
        batches = list(loader)
    assert len(batches) == 2
    np.testing.assert_array_equal(np.asarray(batches[0]), [0, 1, 2, 3])
    np.testing.assert_array_equal(np.asarray(batches[1]), [4, 5, 6, 7])


def test_async_prefetch_iterator_surfaces_worker_error():
    """A failing put_fn must surface as RuntimeError with the original cause."""
    calls = {"n": 0}

    def flaky_put(x):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("boom in put_fn")
        return x

    it = prefetch._AsyncPrefetchIterator(
        iter([1, 2, 3, 4]), 1, flaky_put, queue_size=4, min_fill=0.5
    )
    try:
        with pytest.raises(
            RuntimeError, match="Device prefetch worker failed"
        ) as exc_info:
            for _ in range(100):
                next(it)
    finally:
        it.close()
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_compose_applies_transforms_left_to_right():
    f = compose(lambda x: x + 1, lambda x: x * 10)
    assert f(1) == 20


def test_compose_accepts_sequences_and_none():
    assert compose([lambda x: x + 1, lambda x: x * 10])(1) == 20
    assert compose(None)(5) == 5
    assert compose()(5) == 5


def test_dataloader_host_transforms_list_applies_in_order():
    with DataLoader(
        _IndexDataset(n_samples=4),
        batch_size=4,
        shuffle=False,
        loop=False,
        host_transforms=[lambda x: x + 1, lambda x: x * 10],
        num_prefetch_host=1,
        num_prefetch_device=1,
    ) as loader:
        (batch,) = list(loader)
    np.testing.assert_array_equal(np.asarray(batch), [10, 20, 30, 40])


# ----------------------- regression tests (review fixes) ----------------------- #


class _SlowDataset(_IndexDataset):
    def __init__(self, n_samples=8, delay=0.02):
        super().__init__(n_samples)
        self._delay = delay

    def __getitem__(self, idxs):
        time.sleep(self._delay)
        return super().__getitem__(idxs)


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        gc.collect()
        time.sleep(0.02)
    return predicate()


def _thread_count():
    return threading.active_count()


def test_dataloader_loop_false_slow_dataset_default_prefetch_exact_epoch():
    """Recycling must not duplicate batches of a finite epoch."""
    with DataLoader(
        _SlowDataset(n_samples=16, delay=0.05), batch_size=4, shuffle=False, loop=False
    ) as loader:
        batches = [np.asarray(b) for b in loader]
    assert len(batches) == 4
    for i, b in enumerate(batches):
        np.testing.assert_array_equal(b, np.arange(4 * i, 4 * i + 4))


def test_dataloader_loop_false_sharding_path_exact_epoch():
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    with DataLoader(
        _SlowDataset(n_samples=16, delay=0.05),
        batch_size=4,
        shuffle=False,
        loop=False,
        sharding=sharding,
    ) as loader:
        batches = [np.asarray(b) for b in loader]
    assert len(batches) == 4
    for i, b in enumerate(batches):
        np.testing.assert_array_equal(b, np.arange(4 * i, 4 * i + 4))


def test_dataloader_loop_true_still_recycles_on_host_queue():
    loader = DataLoader(
        _SlowDataset(n_samples=8, delay=0.05),
        batch_size=4,
        shuffle=False,
        loop=True,
        num_prefetch_host=8,
    )
    try:
        it = iter(loader)
        # 8 batches in well under the time the slow dataset needs to produce
        # them: only possible if served batches are recycled.
        start = time.monotonic()
        for _ in range(8):
            next(it)
        assert time.monotonic() - start < 0.3
    finally:
        loader.close()


@pytest.mark.parametrize("use_sharding", [False, True])
def test_dropped_dataloader_releases_threads(use_sharding):
    kwargs = {}
    if use_sharding:
        kwargs["sharding"] = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    baseline = _thread_count()
    refs = []
    for _ in range(3):
        dl = DataLoader(_SlowDataset(n_samples=32), batch_size=4, **kwargs)
        next(iter(dl))
        refs.append(weakref.ref(dl))
        del dl
        gc.collect()
    assert _wait_for(lambda: all(r() is None for r in refs)), "loader leaked"
    assert _wait_for(lambda: _thread_count() <= baseline), (
        f"threads leaked: {_thread_count()} > {baseline}"
    )


def test_dropped_simulation_dataset_releases_thread():
    def sim(key):
        return {"x": jax.random.normal(key, (2,))}

    baseline = _thread_count()
    refs = []
    for _ in range(3):
        ds = datasets.SimulationDataset(
            sim, simulation_batch_size=4, rng=jax.random.PRNGKey(0), buffer_size=8
        )
        ds[np.arange(4)]
        refs.append(weakref.ref(ds))
        del ds
        gc.collect()
    assert _wait_for(lambda: all(r() is None for r in refs)), "dataset leaked"
    assert _wait_for(lambda: _thread_count() <= baseline)


def test_async_prefetch_iterator_dropped_stops_thread():
    baseline = _thread_count()
    it = prefetch._AsyncPrefetchIterator(iter(range(10**9)), 1, lambda x: x)
    next(it)
    ref = weakref.ref(it)
    del it
    gc.collect()
    assert _wait_for(lambda: ref() is None)
    assert _wait_for(lambda: _thread_count() <= baseline)


def test_async_prefetch_iterator_next_after_exhaustion_or_close_stops():
    it = prefetch._AsyncPrefetchIterator(iter([1, 2]), 1, lambda x: x, queue_size=4)
    assert list(it) == [1, 2]
    for _ in range(3):
        with pytest.raises(StopIteration):
            next(it)
    it2 = prefetch._AsyncPrefetchIterator(iter(range(100)), 1, lambda x: x)
    it2.close()
    with pytest.raises(StopIteration):
        next(it2)


def _gated_simulator():
    """Simulator that blocks producer threads (not the main thread) on a
    per-call event, to hold a producer mid-batch deterministically."""
    entered, events = [], [threading.Event() for _ in range(8)]

    def sim(key):
        if threading.current_thread() is not threading.main_thread():
            idx = len(entered)
            entered.append(threading.current_thread())
            events[idx].wait(10)
        return {"x": jax.random.normal(key, (2,))}

    return sim, entered, events


def test_simulation_dataset_reset_stale_producer_cannot_write_or_clobber_seed():
    sim, entered, events = _gated_simulator()
    ref_sim, _, ref_events = _gated_simulator()
    ds = ref = None
    try:
        ref = datasets.SimulationDataset(
            ref_sim,
            simulation_batch_size=4,
            rng=jax.random.PRNGKey(7),
            buffer_size=4,
            jit_simulator=False,
        )
        ds = datasets.SimulationDataset(
            sim,
            simulation_batch_size=4,
            rng=jax.random.PRNGKey(0),
            buffer_size=4,
            jit_simulator=False,
        )
        assert _wait_for(lambda: len(entered) == 1)  # producer mid-simulation
        ds.reset(seed=7)  # joins for 1s, then abandons the stuck producer
        assert _wait_for(lambda: len(entered) == 2)  # new producer mid-batch
        stale = entered[0]
        np.testing.assert_array_equal(
            ds._buffer["x"], ref._buffer["x"]
        )  # same first contents as a fresh dataset with that seed
        before = ds._buffer["x"].copy()
        produced = ds.get_stats()["batches_produced"]
        events[0].set()  # let the stale producer finish
        stale.join(timeout=5)
        assert not stale.is_alive()
        np.testing.assert_array_equal(ds._buffer["x"], before)
        assert ds.get_stats()["batches_produced"] == produced
    finally:
        for e in events + ref_events:
            e.set()
        for d in (ds, ref):
            if d is not None:
                d.close()


def test_simulation_dataset_mark_batch_served_counts():
    ds = datasets.SimulationDataset(
        lambda key: {"x": jax.random.normal(key, (2,))},
        simulation_batch_size=4,
        rng=jax.random.PRNGKey(0),
        buffer_size=8,
    )
    try:
        ds.mark_batch_served(4)
        ds.mark_batch_served(3)
        stats = ds.get_stats()
        assert stats["batches_consumed"] == 2
        assert stats["samples_served"] == 7
        ds.set_data({"x": np.zeros((8, 2), np.float32)})
        assert ds.get_stats()["batches_consumed"] == 0
        ds.mark_batch_served(1)
        assert ds.get_stats()["samples_served"] == 1
    finally:
        ds.close()


def test_simulation_dataset_set_data_is_static_by_default():
    ds = datasets.SimulationDataset(
        lambda key: {"x": jax.random.normal(key, (2,))},
        simulation_batch_size=4,
        rng=jax.random.PRNGKey(0),
        buffer_size=8,
    )
    try:
        data = {"x": np.arange(16, dtype=np.float32).reshape(8, 2)}
        ds.set_data(data)
        for _ in range(20):
            ds[np.arange(4)]
            time.sleep(0.005)
        np.testing.assert_array_equal(ds[np.arange(8)]["x"], data["x"])
        ds.set_data(data, start_producer=True)
        assert ds._producer is not None and ds._producer.is_alive()
    finally:
        ds.close()


def test_dataloader_close_does_not_record_spurious_worker_error():
    loader = DataLoader(
        _SlowDataset(n_samples=64, delay=0.01),
        batch_size=4,
        num_async_workers=2,
        max_in_flight=4,
    )
    next(iter(loader))
    producer = loader._producer
    loader.close()
    time.sleep(0.3)
    assert producer._get_worker_error() is None
