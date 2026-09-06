"""Autoregressive density models over univariate ``probjax.stats`` families.

``p(x) = prod_i p(x_i | x_<i)`` with each conditional a univariate distribution
whose parameters a masked network predicts. Hand the model a distribution class
and everything else follows from it -- ``dist.parameters`` gives the parameter
names, order and constraints, and ``logpdf`` / ``_rvs_impl`` give the density
and the sampler:

    >>> from probjax.stats import norm
    >>> model = Autoregressive(4, norm, rngs=nnx.Rngs(0))

For heads flexible enough to model arbitrary conditionals, the same mechanism
takes the distributions added for the purpose -- ``mixture_kernel`` (KDE-like),
``histogram`` / ``tailed_histogram``, and ``spline_normal`` -- via the
convenience constructors on :class:`ARFamily`, or the named presets below.
"""

from probjax.nn.generative.autoregressive.config import (
    ARConditionerConfig,
    ARFamily,
    MLPARConditionerConfig,
    SSMARConditionerConfig,
    TransformerARConditionerConfig,
)
from probjax.nn.generative.autoregressive.model import (
    Autoregressive,
    CategoricalAutoregressive,
    HistogramAutoregressive,
    MADE,
    MixtureAutoregressive,
    SplineAutoregressive,
    made,
)

__all__ = [
    # family + conditioner configs
    "ARFamily",
    "ARConditionerConfig",
    "MLPARConditionerConfig",
    "SSMARConditionerConfig",
    "TransformerARConditionerConfig",
    # models
    "Autoregressive",
    "MADE",
    "MixtureAutoregressive",
    "SplineAutoregressive",
    "HistogramAutoregressive",
    "CategoricalAutoregressive",
    "made",
]
