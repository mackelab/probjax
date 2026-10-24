"""Inference algorithms for probabilistic models.

This module provides various inference algorithms including:

- MCMC (Markov Chain Monte Carlo) sampling
- SMC (Sequential Monte Carlo) methods
- Filtering and smoothing for state space models
- Rejection sampling

Kernels are pure transitions. Adaptors fit their parameters and runners provide
the standard compiled execution paths.
"""

from probjax.inference.adaptation import compose_adaptors, pooled
from probjax.inference.base import (
    Adaptor,
    Kernel,
    RunResult,
    SMCResult,
    Warmup,
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
    mean_adaptor,
    mh,
    neural_imh,
    neural_imh_warmup,
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

# Rejection sampling
from probjax.inference.rejection import (
    ARSState,
    RejectionSampler,
    ars,
    init_ars_state,
    update_ars_state,
)
from probjax.inference.runner import run, step, warmup

# SMC imports
from probjax.inference.smc import (
    GeometricPath,
    PartialPosteriorsPath,
    PretrainedTwistInfo,
    PretrainedTwistState,
    RDSMCInfo,
    RDSMCState,
    acceptance_rate_adaptor,
    adaptive_persistent_smc,
    adaptive_smc,
    particle_adaptor,
    path_smc,
    persistent_smc,
    run_pretrained_diffusion_smc,
    run_rdsmc,
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
    "Warmup",
    "RunResult",
    "run",
    "step",
    "warmup",
    "compose_adaptors",
    "pooled",
    "SMCResult",
    "step_size_adaptor",
    "mass_matrix_adaptor",
    "mean_adaptor",
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
    "neural_imh_warmup",
    # MCMC kernels
    "hmc",
    "nuts",
    "mala",
    "mh",
    "gauss_rwmh",
    "imh",
    "gaussian_imh",
    "neural_imh",
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
    # SMC
    "smc",
    "adaptive_smc",
    "persistent_smc",
    "adaptive_persistent_smc",
    "run_pretrained_diffusion_smc",
    "run_rdsmc",
    "PretrainedTwistInfo",
    "PretrainedTwistState",
    "RDSMCInfo",
    "RDSMCState",
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

from probjax.inference.filtering.state_space import (
    LinearGaussianModel,
    StateSpaceModel,
    TemporalFilter,
    TemporalResult,
    TemporalTrace,
)
from probjax.inference.filtering.temporal import (
    run_temporal_filter,
    temporal_filter,
)
from probjax.inference.filtering.trajectory import (
    particle_gibbs,
    sample_paths,
    smooth_gaussian_path,
)
from probjax.inference.smc.temporal import (
    TemporalSMC,
    TemporalSMCInfo,
    TemporalSMCState,
    run_temporal_smc,
    temporal_smc,
)

__all__ += [
    "LinearGaussianModel",
    "StateSpaceModel",
    "TemporalFilter",
    "TemporalResult",
    "TemporalTrace",
    "run_temporal_filter",
    "temporal_filter",
    "particle_gibbs",
    "sample_paths",
    "smooth_gaussian_path",
    "TemporalSMC",
    "TemporalSMCInfo",
    "TemporalSMCState",
    "run_temporal_smc",
    "temporal_smc",
]

from probjax.inference.diagnostics import (
    DivergenceConcentrationReport,
    divergence_concentration,
    divergence_concentration_from_counts,
    effective_sample_size,
    ess_bulk,
    ess_tail,
    format_divergence_warning,
    imq_kernel,
    kernelized_stein_discrepancy,
    pareto_khat,
    potential_scale_reduction,
    rhat,
)
from probjax.inference.filtering.joint import (
    sample_joint_paths,
)
from probjax.inference.smc.ports import (
    pretuned_smc,
    tuned_smc,
    waste_free_smc,
    waste_free_strategy,
)
from probjax.inference.smc.temporal_utils import (
    population_diagnostics,
    summarize_replicates,
)

__all__ += [
    'waste_free_strategy',
    'waste_free_smc',
    'tuned_smc',
    'pretuned_smc',
    'population_diagnostics',
    'summarize_replicates',
    'sample_joint_paths',
    'effective_sample_size',
    'potential_scale_reduction',
    'rhat',
    'ess_bulk',
    'ess_tail',
    'pareto_khat',
    'divergence_concentration',
    'divergence_concentration_from_counts',
    'format_divergence_warning',
    'DivergenceConcentrationReport',
    'imq_kernel',
    'kernelized_stein_discrepancy',
]
