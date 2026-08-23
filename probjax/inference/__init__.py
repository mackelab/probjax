"""Inference algorithms for probabilistic models.

This module provides various inference algorithms including:

- MCMC (Markov Chain Monte Carlo) sampling
- SMC (Sequential Monte Carlo) methods
- Filtering and smoothing for state space models
- Rejection sampling

Kernels are pure transitions. Adaptors fit their parameters and runners provide
the standard compiled execution paths.
"""

from probjax.inference.adaptation import adapt, adapt_step, as_warmup, compose_adaptors
from probjax.inference.base import (
    AdaptationResult,
    AdaptationTrace,
    Adaptor,
    FilteringResult,
    Kernel,
    MCMCResult,
    SMCResult,
    Warmup,
    WarmupResult,
)

# Filtering imports
from probjax.inference.filtering import (
    FilterAPI,
    FilterInfo,
    FilterKernel,
    FilterState,
    ParticleFilter,
    extended_kalman_filter,
    kalman_filter,
    make_continuous_transition,
    make_filter_api,
    make_linearized_observation,
    make_linearized_transition,
    particle_smoother,
    rank_reduced_kalman_filter,
    rauch_tung_stribel_smoother,
    smooth,
    sq_kalman_filter,
    ukf,
)

# MCMC imports
from probjax.inference.mcmc import (
    a2rms,
    adjusted_mclmc,
    adjusted_mclmc_dynamic,
    arms,
    covariance_adaptor,
    dynamic_hmc,
    elliptical_slice,
    gauss_rwmh,
    gaussian_imh,
    hmc,
    imh,
    latent_slice,
    mala,
    mass_matrix_adaptor,
    mclmc,
    mclmc_warmup,
    mh,
    nuts,
    pathfinder_warmup,
    pseudo_marginal,
    sghmc,
    sgld,
    sgnht,
    slice,
    slice_step_size_adaptor,
    step_size_adaptor,
    window_warmup,
)
from probjax.inference.mcmc.base import MarkovKernel, Params, State
from probjax.inference.mcmc_runner import MCMC

# Rejection sampling
from probjax.inference.rejection import (
    ARSState,
    RejectionSampler,
    ars,
    init_ars_state,
    update_ars_state,
)

# SMC imports
from probjax.inference.smc import (
    GeometricPath,
    PartialPosteriorsPath,
    acceptance_rate_adaptor,
    adaptive_persistent_smc,
    adaptive_persistent_smc_kernel,
    adaptive_smc,
    adaptive_smc_kernel,
    particle_adaptor,
    path_smc,
    persistent_smc,
    persistent_smc_kernel,
    smc,
    tuning,
)
from probjax.inference.smc_runner import SMC
from probjax.inference.vi import (
    FlowVIInfo,
    FlowVIState,
    NeuTraTransform,
    flow_vi,
    neutra,
)

__all__ = [
    "Kernel",
    "Adaptor",
    "AdaptationResult",
    "AdaptationTrace",
    "Warmup",
    "WarmupResult",
    "adapt",
    "adapt_step",
    "as_warmup",
    "compose_adaptors",
    "MCMCResult",
    "SMCResult",
    "FilteringResult",
    "step_size_adaptor",
    "mass_matrix_adaptor",
    "covariance_adaptor",
    "slice_step_size_adaptor",
    "window_warmup",
    "pathfinder_warmup",
    "flow_vi",
    "neutra",
    "FlowVIState",
    "FlowVIInfo",
    "NeuTraTransform",
    "mclmc_warmup",
    # MCMC kernels
    "hmc",
    "nuts",
    "mala",
    "mh",
    "gauss_rwmh",
    "imh",
    "gaussian_imh",
    "slice",
    "elliptical_slice",
    "latent_slice",
    "mclmc",
    "adjusted_mclmc",
    "adjusted_mclmc_dynamic",
    "dynamic_hmc",
    "arms",
    "a2rms",
    "pseudo_marginal",
    "sgld",
    "sghmc",
    "sgnht",
    # MCMC base classes
    "MarkovKernel",
    "Params",
    "State",
    # MCMC runner
    "MCMC",
    # SMC
    "smc",
    "adaptive_smc",
    "adaptive_smc_kernel",
    "persistent_smc",
    "persistent_smc_kernel",
    "adaptive_persistent_smc",
    "adaptive_persistent_smc_kernel",
    "path_smc",
    "GeometricPath",
    "PartialPosteriorsPath",
    "tuning",
    "particle_adaptor",
    "acceptance_rate_adaptor",
    # SMC runner
    "SMC",
    # Filtering
    "kalman_filter",
    "extended_kalman_filter",
    "ukf",
    "sq_kalman_filter",
    "rank_reduced_kalman_filter",
    "ParticleFilter",
    "particle_smoother",
    "rauch_tung_stribel_smoother",
    "smooth",
    "make_filter_api",
    "make_linearized_transition",
    "make_linearized_observation",
    "make_continuous_transition",
    "FilterAPI",
    "FilterInfo",
    "FilterKernel",
    "FilterState",
    # Rejection sampling
    "ars",
    "ARSState",
    "init_ars_state",
    "update_ars_state",
    "RejectionSampler",
]
