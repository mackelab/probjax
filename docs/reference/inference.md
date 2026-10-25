# Inference

Pure kernels, adaptation utilities and compiled runners. See
[Inference](../guides/inference.md) for how they fit together.

## Runners

::: probjax.inference.run
::: probjax.inference.warmup
::: probjax.inference.step
::: probjax.inference.SMC

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

::: probjax.inference.smc.smc
    options:
      heading: probjax.inference.smc
::: probjax.inference.smc.adaptive.adaptive_smc
    options:
      heading: probjax.inference.adaptive_smc
::: probjax.inference.smc.persistent_smc.persistent_smc
    options:
      heading: probjax.inference.persistent_smc
::: probjax.inference.smc.adaptive_persistent_smc.adaptive_persistent_smc
    options:
      heading: probjax.inference.adaptive_persistent_smc
::: probjax.inference.smc.path_smc.path_smc
    options:
      heading: probjax.inference.path_smc
::: probjax.inference.GeometricPath
::: probjax.inference.PartialPosteriorsPath

## SMC extensions

See the [SMC guide](../guides/smc.md) for BlackJAX presets and adaptive execution.

::: probjax.inference.waste_free_strategy
::: probjax.inference.waste_free_smc
::: probjax.inference.tuned_smc
::: probjax.inference.pretuned_smc
::: probjax.inference.population_diagnostics
::: probjax.inference.summarize_replicates

## Twisted SMC for diffusion models

::: probjax.inference.run_pretrained_diffusion_smc
::: probjax.inference.PretrainedTwistState
::: probjax.inference.PretrainedTwistInfo
::: probjax.inference.run_rdsmc
::: probjax.inference.RDSMCState
::: probjax.inference.RDSMCInfo

## Temporal SMC and trajectory inference

Temporal inference advances through physical observation times. See the
[temporal SMC guide](../guides/temporal-smc.md) and the executable
[example notebook](https://github.com/mackelab/probjax/blob/main/examples/inference/temporal_smc.ipynb).

::: probjax.inference.StateSpaceModel
::: probjax.inference.LinearGaussianModel
::: probjax.inference.run_temporal_filter
::: probjax.inference.temporal_filter
::: probjax.inference.temporal_smc
::: probjax.inference.run_temporal_smc
::: probjax.inference.sample_paths
::: probjax.inference.smooth_gaussian_path
::: probjax.inference.sample_joint_paths
::: probjax.inference.particle_gibbs
::: probjax.inference.TemporalFilter
::: probjax.inference.TemporalTrace
::: probjax.inference.TemporalResult
::: probjax.inference.TemporalSMC
::: probjax.inference.TemporalSMCState
::: probjax.inference.TemporalSMCInfo

## Filtering and smoothing

::: probjax.inference.filtering.kalman_filter.kalman_filter
    options:
      heading: probjax.inference.kalman_filter
::: probjax.inference.filtering.extended_kalman_filter.extended_kalman_filter
    options:
      heading: probjax.inference.extended_kalman_filter
::: probjax.inference.filtering.unscented_kalman_filter
::: probjax.inference.filtering.square_root_kf
::: probjax.inference.filtering.rank_reduced_kalman_filter.rank_reduced_kalman_filter
    options:
      heading: probjax.inference.rank_reduced_kalman_filter
::: probjax.inference.sq_kalman_filter
::: probjax.inference.ParticleFilter
::: probjax.inference.particle_smoother
::: probjax.inference.rauch_tung_stribel_smoother
::: probjax.inference.smooth

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
::: probjax.inference.SMCResult
::: probjax.inference.Kernel
::: probjax.inference.Warmup
::: probjax.inference.Adaptor
::: probjax.inference.FilterState
::: probjax.inference.FilterInfo
::: probjax.inference.FilterKernel
