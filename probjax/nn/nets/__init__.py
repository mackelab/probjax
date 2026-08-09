"""Composed neural-network architectures (no training semantics).

Architectures here are pure :class:`flax.nnx.Module` instances —
``MLP``, ``Transformer``, ``UNet``, ``ResNet``, ``LRUModel``,
``DeepSet``, ``MaskedMLP``, ``Sequential``. Generative-model wrappers
(diffusion families, normalizing flows) live in
:mod:`probjax.nn.generative` and :mod:`probjax.nn.generative.nflows`.
"""

from probjax.nn.nets.lru import LRUModel
from probjax.nn.nets.simple import (
    MLP,
    DeepSet,
    MaskedMLP,
    ResNet,
    Sequential,
)
from probjax.nn.nets.time import TimeMLP
from probjax.nn.nets.transformer import Transformer
from probjax.nn.nets.unets import UNet
__all__ = [
    "DeepSet",
    "LRUModel",
    "MLP",
    "MaskedMLP",
    "ResNet",
    "Sequential",
    "TimeMLP",
    "Transformer",
    "UNet",
]
