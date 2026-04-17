"""Inference algorithms for probabilistic models.

This module provides various inference algorithms including:

- MCMC (Markov Chain Monte Carlo) sampling
- SMC (Sequential Monte Carlo) methods
- Filtering and smoothing for state space models
- Rejection sampling

Example
-------
>>> import jax
>>> from probjax.inference import mcmc, MCMC
>>>
>>> # Create an HMC kernel
>>> kernel = mcmc.hmc(logdensity_fn=lambda x: -0.5 * x**2, step_size=0.1, num_integration_steps=10)
>>>
>>> # Run MCMC
>>> runner = MCMC(kernel, verbose=True)
>>> key = jax.random.PRNGKey(0)
>>> state = kernel.init_state(key, 0.0)
>>> final_state = runner.run(key, state, num_steps=1000)
"""

# MCMC imports
from probjax.inference.mcmc import (
    a2rms,
    adjusted_mclmc,
    adjusted_mclmc_dynamic,
    arms,
    dynamic_hmc,
    elliptical_slice,
    gaussian_imh,
    gauss_rwmh,
    hmc,
    imh,
    latent_slice,
    mala,
    mclmc,
    mh,
    nuts,
    pseudo_marginal,
    sghmc,
    sgld,
    sgnht,
    slice,
)
from probjax.inference.mcmc.base import MarkovKernel, Params, State
from probjax.inference.mcmc_runner import MCMC

# SMC imports
from probjax.inference.smc import (
    adaptive_persistent_smc,
    adaptive_persistent_smc_kernel,
    adaptive_smc,
    adaptive_smc_kernel,
    GeometricPath,
    PartialPosteriorsPath,
    path_smc,
    persistent_smc,
    persistent_smc_kernel,
    smc,
    tuning,
)
from probjax.inference.smc_runner import SMC

# Filtering imports
from probjax.inference.filtering import (
    extended_kalman_filter,
    FilterAPI,
    FilterInfo,
    FilterKernel,
    FilterState,
    kalman_filter,
    make_continuous_transition,
    make_filter_api,
    make_linearized_observation,
    make_linearized_transition,
    ParticleFilter,
    particle_smoother,
    rank_reduced_kalman_filter,
    rauch_tung_stribel_smoother,
    smooth,
    sq_kalman_filter,
    ukf,
)

# Rejection sampling
from probjax.inference.rejection import (
    ars,
    ARSState,
    init_ars_state,
    RejectionSampler,
    update_ars_state,
)

__all__ = [
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
