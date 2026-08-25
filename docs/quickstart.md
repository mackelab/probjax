---
title: Quick Start
---

# Quick Start

Install ProbJax from the repository root before running these examples:

```bash
python -m pip install -e .
```

## Distributions

The recommended distribution API follows SciPy naming conventions. Lowercase
objects such as `norm`, `gamma`, and `categorical` can be used directly or
frozen with fixed parameters.

```python
import jax
from probjax.stats import norm

key = jax.random.key(0)
normal = norm(loc=0.0, scale=1.0)

samples = normal.sample(key, shape=(1_000,))
log_density = normal.logpdf(samples)
mean = normal.mean()
variance = normal.var()
```

The equivalent unfrozen calls are:

```python
samples = norm.rvs(key, 0.0, 1.0, shape=(1_000,))
log_density = norm.logpdf(samples, 0.0, 1.0)
```

Distribution methods vary by family. Consult the [stats API](api/stats.rst)
for the available continuous, discrete, multivariate, and higher-order
distributions.

## Probabilistic Programs

Distribution sampling creates a named site under ProbJax transformations.
Outside a transformation, the same call remains an ordinary JAX sample.

```python
import jax
import jax.numpy as jnp
from probjax.core import condition, joint_sample, log_joint_fn, trace
from probjax.stats import norm

def model(key):
    key_z, key_y = jax.random.split(key)
    z = norm.rvs(key_z, 0.0, 1.0, name="z")
    return norm.rvs(key_y, z, 0.5, name="y")

key = jax.random.key(1)
sites = joint_sample(model)(key)

observed_model = condition(model, {"y": jnp.asarray(0.25)})
latent = joint_sample(observed_model)(key)["z"]
log_joint = log_joint_fn(observed_model)(z=latent)
site_metadata = trace(observed_model, sites=True)(key)
```

Related transformations include `observe`, `intervene`/`do`, `substitute`,
and `scope`.

## Automatic Inversion

ProbJax propagates known values backward through supported JAX primitives.

```python
import jax
import jax.numpy as jnp
from probjax.core import inverse, inverse_and_logabsdet

def transform(x):
    return jnp.exp(2.0 * x + 1.0)

x = jnp.asarray(0.4)
y = transform(x)

recovered = inverse(transform)(y)
recovered, inverse_logdet = inverse_and_logabsdet(transform)(y)

assert jnp.allclose(recovered, x)
```

Inverse transformations compose with `jax.jit` and support custom rules through
`custom_inverse`. Inversion is rule-based; unsupported or non-injective
operations may produce unresolved/NaN results rather than a symbolic inverse.

## Flax NNX Models

ProbJax neural modules use Flax NNX rather than the older Linen `init/apply`
pattern.

```python
import jax.numpy as jnp
from flax import nnx
from probjax.nn import MLP

model = MLP([4, 16, 2], rngs=nnx.Rngs(0))
output = model(jnp.ones((3, 4)))
```

`probjax.nn.generative` provides normalizing flows, diffusion models,
continuous and mean flow matching, and categorical diffusion. Generative-model
families share training and distribution-view interfaces such as `loss(...)`
and `as_dist(event_spec, ...)`; normalizing flows can infer their configured
event shape.

## Inference

The inference package exposes pure kernels, adaptation utilities, and compiled
runners:

```python
from probjax.inference import MCMC, SMC, hmc, nuts, adaptive_smc
```

It also includes Kalman-family filters, particle filtering and smoothing,
rejection sampling, stochastic-gradient MCMC, and multiple warmup strategies.
See the [inference API](api/inference.rst) and curated examples for concrete
kernel configuration.

## Next Steps

- Browse the [API overview](api/index.rst).
- Review the [installation guide](installation.md) for accelerator options.
- Explore the repository's `examples/core`, `examples/stats`,
  `examples/inference`, `examples/nn`, and `examples/utils` directories.

Some notebooks are still being migrated after API refactors. When a notebook
disagrees with this guide, prefer this guide and the current API reference.
