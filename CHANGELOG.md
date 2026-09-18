# Changelog

All notable changes to ProbJax will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — preparing 0.2.0

### Added
- Zensical documentation with executable examples, rendered PPL/Kalman/SMC
  tutorials, migration notes and a release checklist
- Stateful training with `FitState`/`FitInfo`, resumable optimizer/RNG state,
  optional EMA, callbacks, auxiliary metrics and explicit JIT IO callbacks
- Required, overridable event specifications for diffusion and flow matching;
  `DiffusionTransformer` with time and optional global-context conditioning
- Adaptive diagonal-noise SDE step doubling with weak and strong Brownian
  controllers; ODE inverse log-determinants with exact or Hutchinson traces
- Complementary inverse beta/gamma functions (`betainccinv`, `gammainccinv`)
  and stable log-space helpers (`log1mexp`, `logdiffexp`)
- Temporal SMC, BlackJAX-backed adaptation controls, and shared inference runners
- ISAB key masks and per-example attention biases folded into query/key features
- `Autoregressive.sample` supports prefix conditioning (`prefix=`, `prefix_len=`)
  and KV-cached decoding (`use_cache=`) for transformer conditioners, with the
  cached loop compiled to a single program via `nnx.scan`
- `TransformerARConditionerConfig` accepts an `attention_fn` kernel and works
  with discrete families
- `probjax.enable_rv_tracing()` context manager (plus
  `probjax.rv_tracing_enabled()` predicate) to opt sampling code into
  traceable `random_variable` sites outside the PPL transformations
- `probjax.disable_custom_inverse()` context manager (plus
  `probjax.custom_inverse_enabled()` predicate) to opt out of
  `custom_inverse` handling, e.g. for a buggy registered inverse: forward
  calls run the plain function with no `custom_inverse_call_p` primitive,
  and `inverse`/`inverse_and_logabsdet` fall back to structural inversion.
  Set `PROBJAX_DISABLE_CUSTOM_INVERSE=1` to disable globally

### Changed
- Target Python 3.11–3.13 and JAX 0.9.x; require Flax >=0.12.6 and BlackJAX 1.6.2
- Migrate legacy Pallas features to `probjax.nn.pallas_kernels` and remove the
  old package; use an associative-scan fallback for multi-tile GPU Mamba
- Require `event_spec` on diffusion and flow-matching constructors; allow
  per-distribution overrides without changing existing views
- Expose architecture builder overrides and shared training hooks on NN modules
- Keep runtime microbenchmarks opt-in; retain structural/correctness checks in CI
- Mean-flow training objective now defaults to the Improved MeanFlow (iMF)
  v-loss formulation (arXiv:2512.02012): the JVP tangent is the network's
  own boundary-condition velocity `u(z_t, t, t)` (marginal-velocity
  estimate, lower variance) instead of the conditional velocity, and the
  regression target no longer depends on the network. Same optimum, more
  stable optimization. The original objective remains available via
  `loss_kwargs={"imf": False}` or `model.loss(..., imf=False)`. Results
  will differ from previous versions
- Mean-flow `(t, r)` pair sampling (`SigmoidPairFlowTrainingConfig`):
  distinct pairs are now min/max of two independent logit-normal draws
  (matching the MeanFlow paper) instead of `r = clip(t + r_raw)`, which
  pinned ~70% of distinct pairs at exactly `r = 1` and starved interior
  small-gap pairs needed for few-step sampling. This changes the training
  distribution, so mean-flow results will differ from previous versions
- Renamed `AutoregressiveModel` to `Autoregressive` (hard rename, no alias)
- Sampling is now primitive-free by default: `Distribution.rvs`/`sample`
  no longer emit the `random_variable` primitive unless PPL tracing is
  enabled. The PPL transformations (`trace`, `joint_sample`,
  `log_joint_fn`/`log_potential_fn`/`log_prob_fn`, `intervene`/`do`,
  `condition`/`observe`, `substitute`) enable it automatically around their
  internal tracing, so probabilistic programs are unaffected, while plain
  `jit`/`vmap`/`scan` sampling skips the forward-jaxpr trace and the
  site-name counter. Raw `jax.make_jaxpr(model)` on sampling code no longer
  contains `random_variable` equations -- wrap it in the new
  `probjax.enable_rv_tracing()` context if you inspect jaxprs for sites or
  call `interpret` manually. Set `PROBJAX_RV_TRACING=1` to restore the legacy
  always-emit behaviour globally

### Fixed
- Correct the classical RK4 third-stage tableau and stop the stage loop at its
  actual bound; add accuracy and parameter-gradient regressions
- VP/cosine preconditioning, time-rescaled SDE coefficients, stochastic reverse
  V-solver sampling and nonuniform exponential AB2 integration
- `LinearOperator` rectangular shapes, dense composition order, transpose dtype
  and JIT-tracer support; compiled Kalman operator/dense parity
- Upper-tail quantile cancellation, numerical edge cases across distributions,
  inverse propagation, filtering and SMC evidence/resume handling
- Background data-loader error propagation and deterministic prefetch regression
  checks; preserve coverage artifacts when the external Codecov upload fails
- `MeanFlowMatcher.__call__` no longer leaks `t`'s tangent into `r` through
  `jnp.clip(r, min=t)` during the training-time JVP (`stop_gradient` on the
  bound). Loss-neutral for `r == t` pairs (the corrupted term is multiplied
  by `(t - r) = 0`), but removes a latent trap for custom pair configs
- Transformer conditioner full-forward path is now causal: the explicit
  `mask=None` used to override the attention kernel's baked-in causal mask,
  so every position read the future and the density was invalid. Cached
  decoding, naive sampling, and training now agree
- `flex_attention` no longer crashes on stateful masks (their data children
  are not spatial dims and must not be padded)

### Compatibility and validation

See [migration notes](docs/guides/migration.md) before upgrading. Adaptive ODEs
still use a custom reverse-mode adjoint; a gradient-mode selector and Jacobian
stride controls are not public APIs in this release. GPU performance of the
multi-tile Mamba fallback remains unverified. Beta shape gradients use finite
differences and beta inverse forward AD is unsupported. Codecov upload currently
requires repository setup independently of the passing coverage test suite.

## [0.1.1] - 2026-08-01

Published release preceding the changes above. See the tagged source for its
API and dependency versions.

## 0.1.0 development history - 2024

### Added
- Core functionality:
  - Automatic function inversion (`inverse`, `inverse_and_logabsdet`)
  - Function tracing (`trace`)
  - Log probability computation (`log_prob_fn`, `log_joint_fn`)
  - Interventions and conditioning (`intervene`, `observe`, `condition`)
  - Custom JAX primitives for probabilistic programming

- Distributions module:
  - Base classes: `rv_generic`, `rv_continuous`, `rv_discrete`, `rv_multivariate`, `rv_exponential_family`, `rv_spherical`
  - Continuous distributions: `norm`, `gamma`, `beta`, `uniform`, `expon`, `laplace`, `logistic`, `cauchy`, `chi2`, `t`, `pareto`, `skewnorm`, `truncnorm`, `gennorm`, `genpareto`, `vonmises`, `watson`, `bingham`, `wrapcauchy`, `dirichlet`, `multivariate_normal`
  - Discrete distributions: `bernoulli`, `binomial`, `categorical`, `poisson`, `geometric`, `dirac`, `empirical`
  - Higher-order distributions: `transformed`, `mixture`, `indep`
  - Bijective transforms for normalizing flows

- Neural networks module:
  - Architectures: `MLP`, `ResNet`, `Transformer`, `UNet`, `DeepSet`, `SSMModel`, `Sequential`
  - Normalizing flows: `NormalizingFlow`, `AffineCouplingFlow`, `AdditiveCouplingFlow`, `NeuralSplineFlow`, `NeuralAutoregressiveFlow`, `GaussianizationFlow`, `LinearFlow`, `maf`, `realnvp`, `nsf`, `naf`
  - Diffusion models: `DiffusionDenoiser`, `EDM`, `VP`, `VE`, `MultinomialDiffusion`, `FlowMatcher`, `MeanFlowMatcher`
  - Layers: `MultiHeadAttention`, `MaskedLinear`, `ResnetBlock`, `ConvBlock`, `LRUCell`, `MambaCell`, and more
  - Loss functions for flow matching, score matching, and denoising

- Inference module:
  - MCMC: `hmc`, `nuts`, `mala`, `mh`, `slice`, `elliptical_slice`, `latent_slice`, `mclmc`, `dynamic_hmc`, `arms`, `a2rms`, `pseudo_marginal`, `sgld`, `sghmc`, `sgnht`
  - MCMC runner with progress bar support
  - SMC: `smc`, `adaptive_smc`, `persistent_smc`, with geometric and partial posterior tempering
  - Filtering: `kalman_filter`, `extended_kalman_filter`, `ukf`, `particle_filter`, and smoothing algorithms
  - Rejection sampling: `ars` (Adaptive Rejection Sampling), `RejectionSampler`

- Utilities module:
  - ODE/SDE integration: `odeint`, `sdeint`
  - Linear algebra: Cholesky updates, matrix-vector operations
  - Interpolation: linear and polynomial interpolation
  - Graph utilities: JAX-compatible graph algorithms
  - Special functions: `betaincinv`, `digammainv`, `gammaincinv`
  - Solvers: Newton-Raphson root finding

### Features
- JAX-native implementation throughout
- JAX-native inference kernels and compiled runners
- Support for both CPU and GPU (CUDA and Metal)
- Type hints throughout
- Comprehensive example notebooks

### Dependencies
- JAX >= 0.4.34
- NumPy >= 2.0.0
- Flax >= 0.12.0
- Optax, Chex, Einops
- NetworkX, SymPy

[Unreleased]: https://github.com/mackelab/probjax/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/mackelab/probjax/releases/tag/v0.1.1
