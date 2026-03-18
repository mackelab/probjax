import time

import jax
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

import probjax.nn.io_util as io_util
from probjax.nn import DataLoader


def _sharding_spec_of(array):
    sharding = array.sharding
    return getattr(sharding, "spec", getattr(sharding, "partition_spec", P()))


def _require_two_gpus():
    gpu_devices = jax.devices("gpu")
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
        self._aux = np.arange(n_samples * 512, dtype=np.float32).reshape(
            n_samples, 512
        )
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
        batch = loader._process_batch(np.array([0, 1]))
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
            raise AssertionError("NamedSharding should use host_local_array_to_global_array")
        return x

    monkeypatch.setattr(
        io_util.multihost_utils,
        "host_local_array_to_global_array",
        fake_host_local_array_to_global_array,
    )
    monkeypatch.setattr(io_util.jax, "device_put", fail_device_put)

    out = next(
        io_util._prefetch_sharding(
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
        time.sleep(0.05)
        return x

    monkeypatch.setattr(
        io_util.multihost_utils,
        "host_local_array_to_global_array",
        fake_host_local_array_to_global_array,
    )
    monkeypatch.setattr(io_util.jax, "block_until_ready", fake_block_until_ready)

    it = io_util._prefetch_sharding(
        iter(
            [
                {"x": np.array([[0, 1]], dtype=np.int32)},
                {"x": np.array([[2, 3]], dtype=np.int32)},
                {"x": np.array([[4, 5]], dtype=np.int32)},
            ]
        ),
        2,
        sharding,
    )
    try:
        time.sleep(0.12)
        _ = next(it)
        start = time.perf_counter()
        batch = next(it)
        elapsed = time.perf_counter() - start
    finally:
        it.close()

    assert batch["x"] == ("global", 2)
    assert elapsed < 5e-4
    assert len(ready_calls) >= 2


def test_prefetch_sharding_recycles_ready_batches_when_worker_lags(monkeypatch):
    mesh = jax.make_mesh((1,), ("data",), devices=jax.devices()[:1])
    sharding = {"x": jax.sharding.NamedSharding(mesh, P("data", None))}

    def fake_host_local_array_to_global_array(x, global_mesh, pspecs):
        return ("global", int(np.asarray(x).reshape(-1)[0]))

    def fake_block_until_ready(x):
        time.sleep(0.05)
        return x

    monkeypatch.setattr(
        io_util.multihost_utils,
        "host_local_array_to_global_array",
        fake_host_local_array_to_global_array,
    )
    monkeypatch.setattr(io_util.jax, "block_until_ready", fake_block_until_ready)

    it = io_util._prefetch_sharding(
        iter(
            [
                {"x": np.array([[0, 1]], dtype=np.int32)},
                {"x": np.array([[2, 3]], dtype=np.int32)},
            ]
        ),
        1,
        sharding,
    )
    try:
        time.sleep(0.06)
        first = next(it)
        start = time.perf_counter()
        recycled = next(it)
        elapsed = time.perf_counter() - start
        time.sleep(0.06)
        third = next(it)
        fourth = next(it)
    finally:
        it.close()

    assert first["x"] == ("global", 0)
    assert recycled["x"] == ("global", 0)
    assert {third["x"], fourth["x"]} == {("global", 0), ("global", 2)}
    assert elapsed < 5e-4


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
