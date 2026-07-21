"""Denoising-diffusion / score-based generative models.

Includes the model class hierarchy (``DiffusionDenoiser``, ``EDM``, ``VE``,
``VP``, ``CosineDM``), their training losses (denoising MSE +
denoising-score-matching variants), and configuration types.
"""

from probjax.nn.generative.diffusion.model import (
    EDM,
    VE,
    VP,
    CosineDM,
    DiffusionDenoiser,
)
from probjax.nn.losses.denoising import (
    build_denoising_loss,
    build_time_dependent_denoising_loss,
)
from probjax.nn.losses.denoising_score_matching import (
    build_denoising_score_matching_loss,
    build_time_dependent_denoising_score_matching_loss,
)

__all__ = [
    "CosineDM",
    "DiffusionDenoiser",
    "EDM",
    "VE",
    "VP",
    "build_denoising_loss",
    "build_denoising_score_matching_loss",
    "build_time_dependent_denoising_loss",
    "build_time_dependent_denoising_score_matching_loss",
]
