# ProbJax

A JAX-native research toolbox for probabilistic computation: probabilistic-program
transformations, SciPy-style distributions, automatic program inversion, a large
inference suite, and Flax NNX generative models. Core array computations compose
with JAX transformations; host callbacks, data loaders and some solver gradient
paths have additional restrictions described in their guides.

!!! note

    ProbJax is research software. APIs evolve, and the accelerator-specific
    kernels and sharding features are experimental.

```python
import jax
import jax.numpy as jnp
from probjax.core import condition, joint_sample, log_joint_fn
from probjax.stats import norm

def model(key):
    key_z, key_y = jax.random.split(key)
    z = norm.rvs(key_z, 0.0, 1.0, name="z")
    return norm.rvs(key_y, z, 0.5, name="y")

# Sample every named site at once
sites = joint_sample(model)(jax.random.key(0))

# Condition on an observation and score the latent
posterior = condition(model, {"y": jnp.asarray(0.25)})
latent = joint_sample(posterior)(jax.random.key(1))["z"]
log_joint = log_joint_fn(posterior)(z=latent)
```

## What is here

<div class="grid cards" markdown>

-   **Probabilistic programs**

    Write an ordinary JAX function, then `trace`, `condition`, `observe`,
    `intervene` and `joint_sample` it. Sampling a distribution inside a
    transformation records a named site; outside one it is a plain JAX sample.

-   **Program inversion**

    `inverse` and `inverse_and_logabsdet` walk a jaxpr backwards to build the
    inverse of a function, with `custom_inverse` for the parts that need an
    analytic rule. See [Program inversion](guides/program-inversion.md).

-   **Density estimation**

    Normalizing flows, autoregressive models, diffusion and flow matching, with
    one `fit` and one `as_dist` interface across all of them. See
    [Density estimation](guides/density-estimation.md).

-   **Inference**

    MCMC kernels, SMC, Kalman-family filters and particle filtering, plus
    flow-based variational inference and NeuTra preconditioning. See
    [Inference](guides/inference.md).

</div>

## New here?

Start with the [Overview](overview.md) for how the modules fit together and the
[Glossary](glossary.md) for the vocabulary. Then follow
[Getting started](getting-started.md); if something breaks, check
[Troubleshooting](troubleshooting.md), and to work on ProbJax itself see
[Contributing](contributing.md).

## Install

```bash
git clone https://github.com/mackelab/probjax.git
cd probjax
python -m pip install -e .
```

These pages track `main`; the published package can lag behind them. For the
next release, see [migration notes](guides/migration.md).

Python 3.11–3.13 and JAX `>=0.9.0,<0.12`. Accelerator options are in
[Getting started](getting-started.md).

## Tutorials

A curated set of example notebooks is rendered as part of these pages
(generated from `examples/` without re-execution, so outputs are those
committed in the notebooks):

- **Core**: [Jaxpr graphs](tutorials/graph.md),
  [program inversion](tutorials/inversion.md),
  [PPL basics](tutorials/ppl.md)
- **Statistics**: [distribution basics](tutorials/distributions_basic.md),
  [higher-level distributions](tutorials/distributions_highlevel.md)
- **Inference**: [Kalman filtering](tutorials/kalman_filter.md),
  [particle filters](tutorials/filters.md),
  [sequential Monte Carlo](tutorials/smc.md), [MCMC](tutorials/mcmc.md),
  [neural MCMC](tutorials/neural_mcmc.md),
  [rejection sampling](tutorials/rejection.md),
  [Bayesian neural networks](tutorials/bnn.md)
- **Neural models**: [training loops](tutorials/training.md),
  [mixture of experts](tutorials/moe.md), [attention](tutorials/attention.md),
  [Transformer](tutorials/transformer.md),
  [architectures](tutorials/architectures.md),
  [sharding](tutorials/sharding.md),
  [normalizing flows](tutorials/normalizing_flows.md),
  [flow matching](tutorials/flow_matching.md),
  [diffusion](tutorials/diffusion.md), and a
  [generative-model comparison](tutorials/generative_model_tour.md)
- **Numerical utilities**: [ODE integration](tutorials/odeint.md),
  [SDE integration](tutorials/sdeint.md),
  [structured linear operators](tutorials/linear_operator.md)

All runnable notebooks live in
[`examples/`](https://github.com/mackelab/probjax/tree/main/examples/)
(`core`, `stats`, `inference`, `nn`, `utils`).

CPU-runnable tutorial pages have executable examples checked by CI. The heavy
generative notebooks (diffusion, flow matching, normalizing flows, and the
generative-model study) carry committed GPU outputs and are not re-executed on
CPU. The remaining notebooks may predate recent API refactors; prefer the
guides and reference when they disagree with an older notebook.
