# Inference

Pure kernels, adaptation utilities and compiled runners. See
[Inference](../guides/inference.md) for how they fit together.

## Runners

::: probjax.inference.run
::: probjax.inference.warmup
::: probjax.inference.step

## MCMC kernels

::: probjax.inference.mcmc.hmc.hmc
    options:
      heading: probjax.inference.hmc
::: probjax.inference.mcmc.hmc.nuts
    options:
      heading: probjax.inference.nuts
::: probjax.inference.mcmc.dynamic_hmc.dynamic_hmc
    options:
      heading: probjax.inference.dynamic_hmc
::: probjax.inference.mcmc.mala.mala
    options:
      heading: probjax.inference.mala
::: probjax.inference.mcmc.mclmc.mclmc
    options:
      heading: probjax.inference.mclmc
::: probjax.inference.mcmc.mclmc.adjusted_mclmc
    options:
      heading: probjax.inference.adjusted_mclmc
::: probjax.inference.mcmc.mh.mh
    options:
      heading: probjax.inference.mh
::: probjax.inference.mcmc.mh.gauss_rwmh
    options:
      heading: probjax.inference.gauss_rwmh
::: probjax.inference.mcmc.imh.imh
    options:
      heading: probjax.inference.imh
::: probjax.inference.mcmc.slice.slice
    options:
      heading: probjax.inference.slice
::: probjax.inference.mcmc.latent_slice.latent_slice
    options:
      heading: probjax.inference.latent_slice
::: probjax.inference.mcmc.elliptical_slice.elliptical_slice
    options:
      heading: probjax.inference.elliptical_slice
::: probjax.inference.mcmc.arms.arms
    options:
      heading: probjax.inference.arms
::: probjax.inference.mcmc.arms.a2rms
    options:
      heading: probjax.inference.a2rms
::: probjax.inference.mcmc.imh.gaussian_imh
    options:
      heading: probjax.inference.gaussian_imh
::: probjax.inference.mcmc.mclmc.adjusted_mclmc_dynamic
    options:
      heading: probjax.inference.adjusted_mclmc_dynamic
::: probjax.inference.pseudo_marginal

## Exact samplers

Each step returns one independent, exact draw.

::: probjax.inference.rejection.adaptive.ars
    options:
      heading: probjax.inference.ars
::: probjax.inference.rejection.proposal.rejection
    options:
      heading: probjax.inference.rejection
::: probjax.inference.bound_adaptor

## Stochastic-gradient MCMC

::: probjax.inference.sgld
::: probjax.inference.sghmc
::: probjax.inference.sgnht

## Warmup and adaptation

::: probjax.inference.window_warmup
::: probjax.inference.pathfinder_warmup
::: probjax.inference.mclmc_warmup
::: probjax.inference.step_size_adaptor
::: probjax.inference.mass_matrix_adaptor
::: probjax.inference.covariance_adaptor
::: probjax.inference.mean_adaptor
::: probjax.inference.compose_adaptors
::: probjax.inference.acceptance_rate_adaptor
::: probjax.inference.slice_step_size_adaptor
::: probjax.inference.particle_adaptor
::: probjax.inference.pooled
::: probjax.inference.fit_adaptor
::: probjax.inference.maximum_likelihood

## Convergence diagnostics

::: probjax.inference.effective_sample_size
::: probjax.inference.ess_bulk
::: probjax.inference.ess_tail
::: probjax.inference.potential_scale_reduction
::: probjax.inference.rhat
::: probjax.inference.pareto_khat
::: probjax.inference.kernelized_stein_discrepancy
::: probjax.inference.imq_kernel
::: probjax.inference.divergence_concentration
::: probjax.inference.divergence_concentration_from_counts
::: probjax.inference.format_divergence_warning
::: probjax.inference.DivergenceConcentrationReport

## Sequential Monte Carlo

See the [SMC guide](../guides/smc.md).

::: probjax.inference.smc.tempering.smc
    options:
      heading: probjax.inference.smc
::: probjax.inference.SMCState
::: probjax.inference.SMCInfo
::: probjax.inference.particle_adaptor
::: probjax.inference.acceptance_rate_adaptor
::: probjax.inference.population_diagnostics
::: probjax.inference.summarize_replicates

## Twisted SMC for diffusion models

::: probjax.inference.diffusion_smc
::: probjax.inference.Guide
::: probjax.inference.score_guide
::: probjax.inference.neural.diffusion_guide
::: probjax.inference.DiffusionSMCState
::: probjax.inference.DiffusionSMCInfo

## Filtering and smoothing

Every filter is a kernel built from a model, with the model parameters
`theta` as its params. See the [filtering guide](../guides/temporal-smc.md).

::: probjax.inference.LinearGaussianModel
::: probjax.inference.GaussianModel
::: probjax.inference.StateSpaceModel
::: probjax.inference.filtering.kalman_filter.kalman_filter
    options:
      heading: probjax.inference.kalman_filter
::: probjax.inference.filtering.extended_kalman_filter.extended_kalman_filter
    options:
      heading: probjax.inference.extended_kalman_filter
::: probjax.inference.filtering.unscented_kalman_filter.unscented_kalman_filter
    options:
      heading: probjax.inference.unscented_kalman_filter
::: probjax.inference.filtering.square_root_kf.square_root_kalman_filter
    options:
      heading: probjax.inference.square_root_kalman_filter
::: probjax.inference.filtering.rank_reduced_kalman_filter.rank_reduced_kalman_filter
    options:
      heading: probjax.inference.rank_reduced_kalman_filter
::: probjax.inference.filtering.particle_filter.particle_filter
    options:
      heading: probjax.inference.particle_filter
::: probjax.inference.filtering.GaussianFilterState
::: probjax.inference.filtering.GaussianFilterInfo
::: probjax.inference.filtering.ParticleFilterState
::: probjax.inference.filtering.ParticleFilterInfo
::: probjax.inference.filtering.merwe_sigma_point
::: probjax.inference.filtering.julier_uhlmann_sigma_points
::: probjax.inference.filtering.spherical_simplex_sigma_points

## Smoothing and trajectories

::: probjax.inference.smooth
::: probjax.inference.sample_paths
::: probjax.inference.particle_gibbs

## Temporal SMC

::: probjax.inference.temporal_smc
::: probjax.inference.TemporalSMCState
::: probjax.inference.TemporalSMCInfo
::: probjax.inference.sample_joint_paths

## Learned components

::: probjax.inference.Density
::: probjax.inference.neural.density
::: probjax.inference.neural.bijection
::: probjax.inference.mcmc.imh.imh_data

## Variational inference

::: probjax.inference.flow_vi
::: probjax.inference.vi.neutra.neutra
    options:
      heading: probjax.inference.neutra
::: probjax.inference.FlowVIState
::: probjax.inference.FlowVIInfo
::: probjax.inference.NeuTraTransform

## States and results

The types returned by the runners and kernels. Kernel `State` and `Params` are
blackjax's own types, re-exported for convenience and documented there.

::: probjax.inference.RunResult
::: probjax.inference.Kernel
::: probjax.inference.Warmup
::: probjax.inference.Adaptor
