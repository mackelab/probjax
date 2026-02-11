from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import inspect
from typing import Sequence

import jax
from flax import nnx
from jax.sharding import Mesh, PartitionSpec


@dataclass(frozen=True)
class LinearShardingSpec:
    kernel: PartitionSpec | None = None
    bias: PartitionSpec | None = None
    activation: PartitionSpec | None = None


@dataclass(frozen=True)
class MLPShardingSpec:
    mesh: Mesh | None = None
    per_layer: Sequence[LinearShardingSpec | None] | LinearShardingSpec | None = None
    default: LinearShardingSpec | None = None


DEFAULT_LINEAR_SHARDING = LinearShardingSpec(
    kernel=PartitionSpec(None, "model"),
    bias=PartitionSpec("model",),
    activation=PartitionSpec("data", "model"),
)

DEFAULT_MHA_SHARDING = LinearShardingSpec(
    kernel=PartitionSpec(None, "model"),
    bias=PartitionSpec("model",),
    activation=PartitionSpec("data", None, "model"),
)

DEFAULT_TRANSFORMER_HIDDEN_ACTIVATION = PartitionSpec("data", None, "model")
DEFAULT_TRANSFORMER_INPUT_ACTIVATION = PartitionSpec("data", None, None)

TRANSFORMER_MLP_COLUMN_SHARDING = LinearShardingSpec(
    kernel=PartitionSpec(None, "model"),
    bias=PartitionSpec("model",),
    activation=DEFAULT_TRANSFORMER_HIDDEN_ACTIVATION,
)
TRANSFORMER_MLP_ROW_SHARDING = LinearShardingSpec(
    kernel=PartitionSpec("model", None),
    bias=PartitionSpec("model",),
    activation=DEFAULT_TRANSFORMER_HIDDEN_ACTIVATION,
)

DEFAULT_SPATIAL_ACTIVATION = PartitionSpec("data", None, None, "model")


def normalize_mlp_sharding(
    sharding: Mesh | MLPShardingSpec | None, num_layers: int
):
    if sharding is None:
        return None, None, None

    if isinstance(sharding, Mesh):
        mesh = sharding
        default_spec = DEFAULT_LINEAR_SHARDING
        per_layer = None
    elif isinstance(sharding, MLPShardingSpec):
        mesh = sharding.mesh
        default_spec = sharding.default or DEFAULT_LINEAR_SHARDING
        per_layer = sharding.per_layer
    else:
        raise TypeError(
            "sharding must be a jax.sharding.Mesh or MLPShardingSpec, "
            f"got {type(sharding)}"
        )

    if per_layer is None:
        per_layer_specs = [default_spec for _ in range(num_layers)]
    elif isinstance(per_layer, LinearShardingSpec):
        per_layer_specs = [per_layer for _ in range(num_layers)]
    else:
        if len(per_layer) != num_layers:
            raise ValueError(
                f"sharding per_layer must have length {num_layers}, got {len(per_layer)}"
            )
        per_layer_specs = [
            spec if spec is not None else default_spec for spec in per_layer
        ]
    return mesh, default_spec, per_layer_specs


def linear_sharding_kwargs(ctor, spec: LinearShardingSpec | None) -> dict:
    if spec is None:
        return {}
    kwargs = {}
    if spec.kernel is not None:
        init_fn = _resolve_init_fn(ctor, "kernel_init", nnx.initializers.lecun_normal())
        kwargs["kernel_init"] = nnx.with_partitioning(init_fn, spec.kernel)
    if spec.bias is not None:
        init_fn = _resolve_init_fn(ctor, "bias_init", nnx.initializers.zeros)
        kwargs["bias_init"] = nnx.with_partitioning(init_fn, spec.bias)
    return _filter_constructor_kwargs(ctor, **kwargs)


def make_sharded_linear_ctor(base_ctor, sharding_kwargs):
    def ctor(in_features, out_features):
        return base_ctor(in_features, out_features, **sharding_kwargs)

    return ctor


def filter_sharding_kwargs(ctor, **kwargs):
    return _filter_constructor_kwargs(ctor, **kwargs)


def _filter_constructor_kwargs(ctor, **kwargs):
    try:
        param_names = inspect.signature(ctor).parameters.keys()
    except (ValueError, TypeError):
        return kwargs
    return {key: kwargs[key] for key in kwargs if key in param_names}


def _resolve_init_fn(ctor, name: str, default_fn):
    if isinstance(ctor, partial) and ctor.keywords and name in ctor.keywords:
        return ctor.keywords[name]
    return default_fn

