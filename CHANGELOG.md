# Changelog

All notable changes to ProbJax will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial documentation structure with Sphinx
- API documentation generation from docstrings
- FAQ and troubleshooting guides
- Comprehensive tutorials section

## [0.1.0] - 2024

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
  - Higher-order distributions: `transformed`, `mixture`, `independent`
  - Bijective transforms for normalizing flows

- Neural networks module:
  - Architectures: `MLP`, `ResNet`, `Transformer`, `UNet`, `DeepSet`, `LRUModel`, `Sequential`
  - Normalizing flows: `NormalizingFlow`, `AffineCouplingFlow`, `AdditiveCouplingFlow`, `NeuralSplineFlow`, `NeuralAutoregressiveFlow`, `GaussianizationFlow`, `LinearFlow`, `MAF`, `RealNVP`, `NSF`, `NAF`
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
- JIT-compatibility for all inference algorithms
- Support for both CPU and GPU (CUDA and Metal)
- Type hints throughout
- Comprehensive example notebooks

### Dependencies
- JAX >= 0.4.34
- NumPy >= 2.0.0
- Flax >= 0.12.0
- Optax, Chex, Einops
- NetworkX, SymPy

[Unreleased]: https://github.com/probjax/probjax/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/probjax/probjax/releases/tag/v0.1.0
