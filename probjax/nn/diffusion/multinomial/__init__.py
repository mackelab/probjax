"""Multinomial / categorical diffusion models for discrete data."""

from probjax.nn.diffusion.multinomial.config import (
    CategoricalEDMPreconditioning,
    CategoricalPreconditioningProtocol,
    CategoricalScheduleProtocol,
    CategoricalTrainingConfigProtocol,
    ImportanceContinuousTimeTrainingConfig,
    MultinomialDiffusionSchedule,
    UniformContinuousTimeTrainingConfig,
)
from probjax.nn.diffusion.multinomial.loss import (
    build_time_dependent_multinomial_diffusion_loss,
)
from probjax.nn.diffusion.multinomial.model import (
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
