"""Generative-model families organized one-folder-per-method.

Each subpackage bundles its model class and configuration in one place:

* :mod:`probjax.nn.generative.flows` — normalizing flows
  (``NormalizingFlow``, ``realnvp``, ``maf``, ``nsf``, …) plus the
  bijective primitives and conditioner networks they're built from.
* :mod:`probjax.nn.generative.diffusion` — denoising-diffusion / score-based
  models (``DiffusionDenoiser``, ``EDM``, ``VE``, ``VP``, ``CosineDM``).
* :mod:`probjax.nn.generative.flow_matching` — continuous flow matching
  (``FlowMatcher``, ``LinearFlow``).
* :mod:`probjax.nn.generative.mean_flow` — mean / paired flow matching.
* :mod:`probjax.nn.generative.discrete` — categorical / multinomial
  diffusion for discrete data.

All families satisfy :class:`GenerativeModelProtocol`: they train via
``loss(rng, data)`` and expose themselves as distributions via
``as_distribution()``. The functional ``build_*`` loss builders live in
:mod:`probjax.nn.losses`.
"""

from probjax.nn.generative.protocols import GenerativeModelProtocol
from probjax.nn.generative.sampling import BuiltSampler
from probjax.nn.generative.flows import (
    AdditiveAutoregressiveFlow,
    AdditiveCouplingFlow,
    AffineAutoregressiveFlow,
    AffineCouplingFlow,
    BernsteinPolynomialFlow,
    GaussianizationFlow,
    NeuralAutoregressiveFlow,
    NeuralSplineFlow,
    NormalizingFlow,
    SplineAutoregressiveFlow,
    SplineCouplingFlow,
    SumOfSquaresPolynomialFlow,
    UnconstrainedNeuralAutoregressiveFlow,
    bpf,
    gf,
    maf,
    naf,
    nice,
    nsf,
    realnvp,
    sospf,
    unaf,
)
from probjax.nn.generative.diffusion import (
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
from probjax.nn.generative.flow_matching import (
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
from probjax.nn.generative.mean_flow import (
    FlowPairTrainingConfigProtocol,
    LinearMeanFlow,
    MeanFlowMatcher,
    SigmoidPairFlowTrainingConfig,
    build_mean_flow_matching_loss,
    build_mean_flow_matching_loss_from_schedule,
)
from probjax.nn.generative.discrete import (
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
