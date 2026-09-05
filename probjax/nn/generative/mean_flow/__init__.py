"""Mean flow-matching generative models (paired-flow training).

Reuses the configuration types from
:mod:`probjax.nn.generative.flow_matching.config`; mean-flow-specific
training configs live in :mod:`probjax.nn.generative.mean_flow.config`.
"""

from probjax.nn.generative.mean_flow.config import (
    FlowPairTrainingConfigProtocol,
    SigmoidPairFlowTrainingConfig,
)
from probjax.nn.generative.mean_flow.model import (
    LinearMeanFlow,
    MeanFlowMatcher,
)
from probjax.nn.losses.mean_flow import (
    build_mean_flow_matching_loss,
    build_mean_flow_matching_loss_from_schedule,
)

__all__ = [
    "LinearMeanFlow",
    "MeanFlowMatcher",
    "build_mean_flow_matching_loss",
    "build_mean_flow_matching_loss_from_schedule",
    "FlowPairTrainingConfigProtocol",
    "SigmoidPairFlowTrainingConfig",
]
