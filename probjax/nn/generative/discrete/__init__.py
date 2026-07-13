"""Multinomial / categorical diffusion models for discrete data."""

from probjax.nn.generative.discrete.config import (
    CategoricalEDMPreconditioning,
    CategoricalPreconditioningProtocol,
    CategoricalScheduleProtocol,
    CategoricalTrainingConfigProtocol,
    ImportanceContinuousTimeTrainingConfig,
    MultinomialDiffusionSchedule,
    UniformContinuousTimeTrainingConfig,
)
from probjax.nn.losses.multinomial import (
    build_time_dependent_multinomial_diffusion_loss,
)
from probjax.nn.generative.discrete.model import (
    MultinomialCosineDM,
    MultinomialDiffusion,
    MultinomialLogSNRDM,
)

__all__ = [
    "MultinomialCosineDM",
    "MultinomialDiffusion",
    "MultinomialLogSNRDM",
    "build_time_dependent_multinomial_diffusion_loss",
    # configs
    "CategoricalEDMPreconditioning",
    "CategoricalPreconditioningProtocol",
    "CategoricalScheduleProtocol",
    "CategoricalTrainingConfigProtocol",
    "ImportanceContinuousTimeTrainingConfig",
    "MultinomialDiffusionSchedule",
    "UniformContinuousTimeTrainingConfig",
]
