# Inference

Pure kernels, adaptation utilities and compiled runners. See
[Inference](../guides/inference.md) for how they fit together.

## Runners

::: probjax.inference.MCMC
::: probjax.inference.SMC

## MCMC kernels

::: probjax.inference.hmc
::: probjax.inference.nuts
::: probjax.inference.dynamic_hmc
::: probjax.inference.mala
::: probjax.inference.mclmc
::: probjax.inference.adjusted_mclmc
::: probjax.inference.mh
::: probjax.inference.gauss_rwmh
::: probjax.inference.imh
::: probjax.inference.slice
::: probjax.inference.latent_slice
::: probjax.inference.elliptical_slice
::: probjax.inference.arms
::: probjax.inference.a2rms
::: probjax.inference.ars
::: probjax.inference.gaussian_imh
::: probjax.inference.adjusted_mclmc_dynamic
::: probjax.inference.RejectionSampler
::: probjax.inference.pseudo_marginal

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
::: probjax.inference.compose_adaptors
::: probjax.inference.acceptance_rate_adaptor
::: probjax.inference.slice_step_size_adaptor
::: probjax.inference.particle_adaptor
::: probjax.inference.adapt
::: probjax.inference.adapt_step
::: probjax.inference.as_warmup

## Sequential Monte Carlo

::: probjax.inference.smc
::: probjax.inference.adaptive_smc
::: probjax.inference.persistent_smc
::: probjax.inference.adaptive_persistent_smc
::: probjax.inference.path_smc
::: probjax.inference.GeometricPath
::: probjax.inference.PartialPosteriorsPath

## Filtering and smoothing

::: probjax.inference.kalman_filter
::: probjax.inference.extended_kalman_filter
::: probjax.inference.filtering.unscented_kalman_filter
::: probjax.inference.filtering.square_root_kf
::: probjax.inference.rank_reduced_kalman_filter
::: probjax.inference.sq_kalman_filter
::: probjax.inference.ParticleFilter
::: probjax.inference.particle_smoother
::: probjax.inference.rauch_tung_stribel_smoother
::: probjax.inference.smooth

## Variational inference

::: probjax.inference.flow_vi
::: probjax.inference.neutra
::: probjax.inference.FlowVIState
::: probjax.inference.FlowVIInfo
::: probjax.inference.NeuTraTransform

## States and results

The types returned by the runners and kernels. Kernel `State` and `Params` are
blackjax's own types, re-exported for convenience and documented there.

::: probjax.inference.MCMCResult
::: probjax.inference.SMCResult
::: probjax.inference.FilteringResult
::: probjax.inference.AdaptationResult
::: probjax.inference.WarmupResult
::: probjax.inference.MarkovKernel
::: probjax.inference.Kernel
::: probjax.inference.Warmup
::: probjax.inference.Adaptor
::: probjax.inference.FilterState
::: probjax.inference.FilterInfo
::: probjax.inference.FilterKernel
