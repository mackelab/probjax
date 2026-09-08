# ProbJax

A JAX-native research toolbox for probabilistic computation: probabilistic-program
transformations, SciPy-style distributions, automatic program inversion, a large
inference suite, and Flax NNX generative models — all `jit`-able and `vmap`-able
together.

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

    Ten MCMC kernels, SMC, Kalman-family filters and particle filtering, plus
    flow-based variational inference and NeuTra preconditioning. See
    [Inference](guides/inference.md).

</div>

## Install

```bash
git clone https://github.com/mackelab/probjax.git
cd probjax
python -m pip install -e .
```

Python 3.11 or newer. Accelerator options are in
[Getting started](getting-started.md).

## Tutorials

A curated set of example notebooks is rendered as part of these pages
(generated from `examples/` without re-execution, so outputs are those
committed in the notebooks — each one is re-executed against current `main`
before inclusion):

- [PPL basics](tutorials/ppl.md)
- [Kalman filtering](tutorials/kalman_filter.md)
- [Sequential Monte Carlo](tutorials/smc.md)

All runnable notebooks live in
[`examples/`](https://github.com/mackelab/probjax/tree/main/examples/)
(`core`, `stats`, `inference`, `nn`, `utils`). More tutorials are added
gradually as notebook outputs are refreshed.

The remaining notebooks under `examples/` may predate recent API refactors —
the tutorial pages above are re-executed against current `main` and kept fresh.
Where anything disagrees with these pages, prefer these pages and the
[Reference](reference/core.md).
