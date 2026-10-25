"""Autoregressive density models: ``p(x) = prod_i p(x_i | x_<i)``.

Two choices make a model -- the *head* (what each conditional is) and the
*conditioner* (the causal network predicting the head's parameters):

    >>> Autoregressive(4, rngs=nnx.Rngs(0))                       # MADE
    >>> Autoregressive(4, MixtureHead(10), rngs=nnx.Rngs(0))      # KDE-like
    >>> Autoregressive(
    ...     196, CategoricalHead(256), rngs=nnx.Rngs(0),
    ...     conditioner=TransformerARConditionerConfig(),
    ... )
"""

from probjax.nn.generative.autoregressive.conditioners import (
    ARConditionerConfig,
    MLPARConditionerConfig,
    SSMARConditionerConfig,
    TransformerARConditionerConfig,
)
from probjax.nn.generative.autoregressive.heads import (
    CategoricalHead,
    Head,
    HistogramHead,
    LocScaleHead,
    MixtureHead,
    NormalHead,
    SplineHead,
)
from probjax.nn.generative.autoregressive.model import Autoregressive

__all__ = [
    "Autoregressive",
    # heads
    "Head",
    "CategoricalHead",
    "HistogramHead",
    "LocScaleHead",
    "MixtureHead",
    "NormalHead",
    "SplineHead",
    # conditioners
    "ARConditionerConfig",
    "MLPARConditionerConfig",
    "SSMARConditionerConfig",
    "TransformerARConditionerConfig",
]
