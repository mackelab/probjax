"""Composed neural-network architectures (no training semantics).

Architectures here are pure :class:`flax.nnx.Module` instances —
``MLP``, ``Transformer``, ``UNet``, ``ResNet``, ``LRUModel``,
``DeepSet``, ``MaskedMLP``, ``Sequential``. Generative-model wrappers
(diffusion families, normalizing flows) live in
:mod:`probjax.nn.diffusion` and :mod:`probjax.nn.flows`.
"""

from probjax.nn.nets.lru import LRUModel
from probjax.nn.nets.simple import (
    MLP,
    DeepSet,
    MaskedMLP,
    ResNet,
    Sequential,
)
from probjax.nn.nets.transformer import Transformer
from probjax.nn.nets.unets import UNet
from probjax.nn.sharding import (
    LinearShardingCfg,
    LinearShardingSpec,
    MLPShardingSpec,
    NormShardingSpec,
    ShardingCfg,
    SpatialShardingCfg,
    TransformerShardingCfg,
)

__all__ = [
    "DeepSet",
    "LRUModel",
    "MLP",
    "MaskedMLP",
    "ResNet",
    "Sequential",
    "Transformer",
    "UNet",
    # sharding (re-exported here for convenience)
    "LinearShardingCfg",
    "LinearShardingSpec",
    "MLPShardingSpec",
    "NormShardingSpec",
    "ShardingCfg",
    "SpatialShardingCfg",
    "TransformerShardingCfg",
]
