"""Generative-model families organized one-folder-per-method.

Each subpackage bundles its model class, training loss, and configuration
in one place:

* :mod:`probjax.nn.diffusion.ddpm` — denoising-diffusion / score-based
  models (``DiffusionDenoiser``, ``EDM``, ``VE``, ``VP``, ``CosineDM``).
* :mod:`probjax.nn.diffusion.flow_matching` — continuous flow matching
  (``FlowMatcher``, ``LinearFlow``).
* :mod:`probjax.nn.diffusion.mean_flow` — mean / paired flow matching.
* :mod:`probjax.nn.diffusion.multinomial` — categorical / multinomial
  diffusion for discrete data.

Cross-family losses (vanilla score matching, sliced score matching,
target score matching) live in :mod:`probjax.nn.losses`.
"""

from probjax.nn.diffusion.ddpm import (
    EDM,
    VE,
    VP,
    CosineDM,
    DiffusionDenoiser,
    build_denoising_loss,
    build_denoising_score_matching_loss,
    build_time_dependent_denoising_loss,
    build_time_dependent_denoising_score_matching_loss,
)
from probjax.nn.diffusion.flow_matching import (
    AutodiffInterpolationSchedule,
    CosineInterpolationSchedule,
    FlowMatcher,
    FlowPreconditioningProtocol,
    FlowSolverConfigProtocol,
    FlowTrainingConfigProtocol,
    GaussianFlowPreconditioning,
    GeneralInterpolationSchedule,
    InterpolationScheduleProtocol,
    LinearFlow,
    LinearFlowSolverConfig,
    LinearInterpolationSchedule,
    LogitNormalFlowTrainingConfig,
    QuadraticInterpolationSchedule,
    RhoFlowSolverConfig,
    UniformFlowTrainingConfig,
    build_flow_matching_loss,
)
from probjax.nn.diffusion.mean_flow import (
    FlowPairTrainingConfigProtocol,
    LinearMeanFlow,
    MeanFlowMatcher,
    SigmoidPairFlowTrainingConfig,
    build_mean_flow_matching_loss,
    build_mean_flow_matching_loss_from_schedule,
)
from probjax.nn.diffusion.multinomial import (
    CategoricalEDMPreconditioning,
    CategoricalPreconditioningProtocol,
    CategoricalScheduleProtocol,
    CategoricalTrainingConfigProtocol,
    ImportanceContinuousTimeTrainingConfig,
    MultinomialCosineDM,
    MultinomialDiffusion,
    MultinomialDiffusionSchedule,
    MultinomialLogSNRDM,
    UniformContinuousTimeTrainingConfig,
    build_time_dependent_multinomial_diffusion_loss,
)
