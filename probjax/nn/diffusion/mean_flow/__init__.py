"""Mean flow-matching generative models (paired-flow training).

Reuses the configuration types from
:mod:`probjax.nn.diffusion.flow_matching.config`; mean-flow-specific
training configs live in :mod:`probjax.nn.diffusion.mean_flow.config`.
"""

from probjax.nn.diffusion.mean_flow.config import (
    FlowPairTrainingConfigProtocol,
    SigmoidPairFlowTrainingConfig,
)
from probjax.nn.diffusion.mean_flow.loss import (
    build_mean_flow_matching_loss,
    build_mean_flow_matching_loss_from_schedule,
)
from probjax.nn.diffusion.mean_flow.model import (
    LinearMeanFlow,
    MeanFlowMatcher,
)

__all__ = [
    "LinearMeanFlow",
    "MeanFlowMatcher",
    "build_mean_flow_matching_loss",
    "build_mean_flow_matching_loss_from_schedule",
    "FlowPairTrainingConfigProtocol",
    "SigmoidPairFlowTrainingConfig",
]
