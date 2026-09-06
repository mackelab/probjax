# Getting started

## Install

ProbJax requires Python 3.11 or newer and is installed from a checkout:

```bash
git clone https://github.com/mackelab/probjax.git
cd probjax
python -m pip install -e .
```

With [uv](https://docs.astral.sh/uv/), `uv pip install -e .`.

### Accelerators

Backend support depends on your OS, drivers and JAX's own compatibility matrix.

=== "NVIDIA CUDA 13"

    ```bash
    python -m pip install -e ".[cuda13]"
    ```

=== "Apple Silicon Metal"

    ```bash
    python -m pip install -e ".[metal]"
    export JAX_PLATFORMS=metal,cpu
    ```

    Metal support in JAX is experimental and covers fewer primitives than the
    CPU and CUDA backends. If something fails only under Metal, re-run with
    `JAX_PLATFORMS=cpu` to confirm before reporting it.

=== "CPU only"

    ```bash
    export JAX_PLATFORMS=cpu
    ```

Check what JAX actually picked:

```python
import jax
print(jax.devices())
```

## Distributions

The distribution API follows SciPy's naming. Lowercase objects such as `norm`,
`gamma` and `categorical` are used directly, or frozen with fixed parameters:

```python
import jax
from probjax.stats import norm

key = jax.random.key(0)
frozen = norm(loc=0.0, scale=1.0)

samples = frozen.sample(key, shape=(1000,))
log_density = frozen.logpdf(samples)
mean, variance = frozen.mean(), frozen.var()
```

The unfrozen equivalents take the parameters positionally:

```python
samples = norm.rvs(key, 0.0, 1.0, shape=(1000,))
log_density = norm.logpdf(samples, 0.0, 1.0)
```

Available methods vary by family — see the [stats reference](reference/stats.md)
for the continuous, discrete, multivariate and higher-order distributions.

## Probabilistic programs

Sampling a distribution with a `name` records a site when the function is under
a ProbJax transformation. The same call outside one is an ordinary JAX sample,
so a model is just a function:

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

sites = joint_sample(model)(key)          # {"z": ..., "y": ...}
observed = condition(model, {"y": jnp.asarray(0.25)})
latent = joint_sample(observed)(key)["z"]
log_joint = log_joint_fn(observed)(z=latent)
metadata = trace(observed, sites=True)(key)
```

`observe`, `intervene`/`do`, `substitute` and `scope` transform programs the
same way.

Sampling records a site only while a transformation is tracing the model --
everywhere else the same call is an ordinary JAX sample with no `rv`
primitive in the jaxpr, so plain `jit`/`vmap` sampling stays cheap. If you
inspect jaxprs for sites yourself (e.g. raw `jax.make_jaxpr`), opt in
explicitly:

```python
from probjax import enable_rv_tracing

with enable_rv_tracing():
    jaxpr = jax.make_jaxpr(model)(key)
```

The flag is read at trace time (`make_jaxpr` caches per function, and `jit`
compiles on first call), so enable it around the tracing call itself -- and
avoid first-calling a jitted sampler inside the context unless you want the
compiled artifact to keep the sites.

## Neural modules

Neural components use Flax NNX, not the older Linen `init`/`apply` pattern:

```python
import jax.numpy as jnp
from flax import nnx
from probjax.nn import MLP

model = MLP([4, 16, 2], rngs=nnx.Rngs(0))
output = model(jnp.ones((3, 4)))
```

## Where next

- [Program inversion](guides/program-inversion.md) — the part of ProbJax with no
  direct equivalent elsewhere
- [Density estimation](guides/density-estimation.md) — flows, autoregressive
  models, diffusion
- [Inference](guides/inference.md) — MCMC, SMC, filtering, variational inference
