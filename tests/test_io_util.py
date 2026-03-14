import jax
import numpy as np
from jax.sharding import PartitionSpec as P

from probjax.nn import DataLoader


def _sharding_spec_of(array):
    sharding = array.sharding
    return getattr(sharding, "spec", getattr(sharding, "partition_spec", P()))


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
