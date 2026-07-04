"""Continuous flow-matching generative models.

Public API: ``FlowMatcher`` / ``LinearFlow`` model classes, the loss builder
``build_flow_matching_loss``, and the configuration types from
:mod:`probjax.nn.diffusion.flow_matching.config`.
"""

from probjax.nn.diffusion.flow_matching.config import (
    AutodiffInterpolationSchedule,
    CosineInterpolationSchedule,
    FlowPreconditioningProtocol,
    FlowSolverConfigProtocol,
    FlowTrainingConfigProtocol,
    GaussianFlowPreconditioning,
    GeneralInterpolationSchedule,
    InterpolationScheduleProtocol,
    LinearFlowSolverConfig,
    LinearInterpolationSchedule,
    LogitNormalFlowTrainingConfig,
    QuadraticInterpolationSchedule,
    RhoFlowSolverConfig,
    UniformFlowTrainingConfig,
)
from probjax.nn.diffusion.flow_matching.loss import build_flow_matching_loss
from probjax.nn.diffusion.flow_matching.model import FlowMatcher, LinearFlow

__all__ = [
    "FlowMatcher",
    "LinearFlow",
    "build_flow_matching_loss",
    # config types
    "AutodiffInterpolationSchedule",
    "CosineInterpolationSchedule",
    "FlowPreconditioningProtocol",
    "FlowSolverConfigProtocol",
    "FlowTrainingConfigProtocol",
    "GaussianFlowPreconditioning",
    "GeneralInterpolationSchedule",
    "InterpolationScheduleProtocol",
    "LinearFlowSolverConfig",
    "LinearInterpolationSchedule",
    "LogitNormalFlowTrainingConfig",
    "QuadraticInterpolationSchedule",
    "RhoFlowSolverConfig",
    "UniformFlowTrainingConfig",
]
