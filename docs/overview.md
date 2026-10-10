# Overview

ProbJax is a JAX-native library for probabilistic computation. It is organised
into a small number of top-level modules, each doing one job: probabilistic
programs and their transformations, SciPy-shaped distributions, Flax NNX
generative models, an inference suite, and the numerical helpers the rest are
built on.

This page is a map. Each section names the module, what lives in it, and where
to read more. If a term is unfamiliar, the [Glossary](glossary.md) defines the
vocabulary used throughout.

## `probjax.core` — probabilistic programs

A model is an ordinary JAX function. When it draws a named value under a ProbJax
transformation, the draw records a **site**; the transformations then rewire the
program: condition on data (`condition`/`observe`), intervene causally
(`intervene`/`do`), fix values (`substitute`), or group auto-named draws
(`scope`). Sampling and scoring use `joint_sample`, `log_joint_fn` and
`log_potential_fn`, and `trace(..., sites=True)` reports every site it saw.

The module also holds **automatic program inversion** — `inverse`,
`inverse_and_logabsdet` and `custom_inverse` walk a jaxpr backwards, with
`JaxprGraph` exposing the traced graph.

- [Probabilistic programs](guides/ppl.md) — the transformation guide
- [Program inversion](guides/program-inversion.md) — inverting a function
- [Core reference](reference/core.md) — the full symbol list

## `probjax.stats` — distributions

SciPy-shaped distributions: lowercase generators such as `norm`, `gamma`,
`dirichlet` and `categorical`, used directly (`norm.rvs(key, loc, scale)`) or
frozen with fixed parameters (`norm(loc=..., scale=...)`). Higher-order
constructors — `indep`, `mixture` and `transformed` — build joint, mixture and
pushed-forward distributions. The module also provides the training loop, `fit`,
shared by the generative models.

- [Distributions](guides/stats-distributions.md) — using, freezing and fitting
- [Stats reference](reference/stats.md) — every family and method

## `probjax.nn` — generative models

Flax NNX modules for density estimation: normalizing flows, autoregressive
models, diffusion models and flow matching. They share one interface —
construct with `nnx.Rngs`, call `fit` to train, and `as_dist()` for a frozen
distribution with `logpdf` and `sample`.

- [Density estimation](guides/density-estimation.md) — the shared interface
- [Reference](reference/nn.md) — per-family documentation

## `probjax.inference` — inference

MCMC kernels, sequential Monte Carlo, Kalman-family filters, particle filtering
and flow-based variational inference, with NeuTra preconditioning. The inference
algorithms take a `log_potential_fn` (or a model) and a key.

- [Inference](guides/inference.md) — MCMC and variational inference
- [Sequential Monte Carlo](guides/smc.md) and
  [Temporal SMC](guides/temporal-smc.md) — particle methods

## `probjax.utils` — numerical building blocks

ODE and SDE integrators, root finders, special functions (inverse incomplete
beta and gamma, `log1mexp`, `logdiffexp`), interpolation and linear-algebra
helpers, and the shared type aliases used across the package.

- [ODE and SDE integration](guides/numerical-solvers.md) — solvers and gradients
- [Utilities reference](reference/utils.md) — the full symbol list

## Where to go next

- [Getting started](getting-started.md) — install and first steps
- [Glossary](glossary.md) — the vocabulary these pages use
- [Troubleshooting](troubleshooting.md) — common errors and their fixes
- [Contributing](contributing.md) — development workflow and documentation build
