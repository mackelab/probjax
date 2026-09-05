"""Variational inference, and using it to help the samplers.

:func:`flow_vi` fits a normalizing flow to an unnormalized target by
reparameterised reverse KL, in the shape of ``blackjax.vi.meanfield_vi``.
:func:`neutra` then reparameterises the target through that flow so any existing
MCMC kernel can sample it in better-conditioned coordinates.
"""

from probjax.inference.vi.flow_vi import (
    FlowVIInfo,
    FlowVIState,
    rebuild,
)
from probjax.inference.vi.flow_vi import (
    as_top_level_api as flow_vi,
)
from probjax.inference.vi.neutra import NeuTraTransform, neutra

__all__ = [
    "FlowVIInfo",
    "FlowVIState",
    "NeuTraTransform",
    "flow_vi",
    "neutra",
    "rebuild",
]
