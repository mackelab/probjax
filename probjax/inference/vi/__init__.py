"""Variational inference, and using it to help the samplers.

:func:`flow_vi` fits a learned density to an unnormalized target by
reparameterised reverse KL. :func:`neutra` then reparameterises the target
through a fitted flow so any existing MCMC kernel can sample it in
better-conditioned coordinates.
"""

from probjax.inference.vi.flow_vi import FlowVIInfo, FlowVIState, flow_vi
from probjax.inference.vi.neutra import NeuTraTransform, neutra

__all__ = [
    "FlowVIInfo",
    "FlowVIState",
    "NeuTraTransform",
    "flow_vi",
    "neutra",
]
