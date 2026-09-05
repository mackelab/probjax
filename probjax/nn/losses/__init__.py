"""Functional training-loss builders.

Every loss is a higher-order ``build_*`` function: it takes a ``model_fn``
callable (typically a bound method of an ``nnx.Module``) plus schedule /
weighting options and returns a ``loss_fn``. The generative model classes'
``model.loss(rng, data)`` methods are thin conveniences that delegate here.
"""

from probjax.nn.losses.denoising import (
    build_denoising_loss,
    build_time_dependent_denoising_loss,
)
from probjax.nn.losses.denoising_score_matching import (
    build_denoising_score_matching_loss,
    build_time_dependent_denoising_score_matching_loss,
)
from probjax.nn.losses.flow_matching import build_flow_matching_loss
from probjax.nn.losses.mean_flow import (
    build_mean_flow_matching_loss,
    build_mean_flow_matching_loss_from_schedule,
)
from probjax.nn.losses.multinomial import (
    build_time_dependent_multinomial_diffusion_loss,
)
from probjax.nn.losses.score_matching import (
    build_score_matching_loss,
    build_time_dependent_score_matching_loss,
)
from probjax.nn.losses.sliced_score_matching import (
    build_sliced_score_matching_loss,
    build_time_dependent_sliced_score_matching_loss,
)
from probjax.nn.losses.target_score_matching import (
    build_target_score_matching_loss,
    build_time_dependent_target_score_matching_loss,
)

__all__ = [
    "build_denoising_loss",
    "build_denoising_score_matching_loss",
    "build_flow_matching_loss",
    "build_mean_flow_matching_loss",
    "build_mean_flow_matching_loss_from_schedule",
    "build_score_matching_loss",
    "build_sliced_score_matching_loss",
    "build_target_score_matching_loss",
    "build_time_dependent_denoising_loss",
    "build_time_dependent_denoising_score_matching_loss",
    "build_time_dependent_multinomial_diffusion_loss",
    "build_time_dependent_score_matching_loss",
    "build_time_dependent_sliced_score_matching_loss",
    "build_time_dependent_target_score_matching_loss",
]
