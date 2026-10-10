# Distributions

ProbJax implements SciPy-shaped distributions on top of JAX. Each family has a
lowercase **generator** (`norm`, `gamma`, `dirichlet`, `categorical`, ...) that
is used directly, and a **frozen** form that binds the parameters. Both expose
the same methods: `logpdf`, `sample` and, where the family supports them,
`cdf`, `ppf`, `isf`, `mean` and `var`.

The reference lists every family and method; this page covers how to use them,
compose them, and fit them. The [Glossary](../glossary.md) defines frozen/view and
`name=`.

## Generator vs frozen

The generator takes the parameters positionally, one draw at a time:

```python
import jax

from probjax.stats import norm

key = jax.random.key(0)
draw = norm.rvs(key, 0.0, 1.0)
log_density = norm.logpdf(draw, 0.0, 1.0)
mean = norm.mean(0.0, 1.0)
```

Freezing binds the parameters and gives the same methods without repeating
them. `norm(loc=..., scale=...)` returns an `rv_frozen`:

```python
frozen = norm(loc=0.0, scale=1.0)
draws = frozen.sample(key, (1000,))
log_densities = frozen.logpdf(draws)
mean, variance = frozen.mean(), frozen.var()
```

`cdf`, `ppf` and `isf` are available on families that implement them — for
example the normal:

```python
p = norm.cdf(0.0, 0.0, 1.0)
quantile = norm.ppf(0.5, 0.0, 1.0)
upper = norm.isf(0.5, 0.0, 1.0)
```

## Sites and `name=`

Inside a ProbJax transformation, a draw with `name=` records a site that the
transformations can condition on or intervene at:

```python
import jax
import jax.numpy as jnp

from probjax.core import log_joint_fn, trace
from probjax.stats import norm


def model(key):
    k1, k2 = jax.random.split(key)
    z = norm.rvs(k1, 0.0, 1.0, name="z")
    y = norm.rvs(k2, z, 0.5, name="y")
    return y


sites = trace(model, sites=True)(key)
assert set(sites) == {"z", "y"}
```

Outside a transformation the same call is an ordinary JAX sample. See
[Probabilistic programs](ppl.md) for the full set of transformations.

## Higher-order distributions

Three constructors build new distributions from existing ones.

`indep` treats the batch dimensions of a frozen distribution as a single event,
turning a vector of independent draws into one multivariate draw:

```python
import jax.numpy as jnp

from probjax.stats import indep, norm

base = norm(loc=jnp.zeros(3), scale=jnp.ones(3))
joint = indep(base, reinterpreted_batch_ndims=1)
assert joint.event_shape == (3,)
```

`mixture` combines components with mixing probabilities:

```python
from probjax.stats import mixture, norm

mix = mixture(jnp.array([0.5, 0.5]), [norm(0.0, 1.0), norm(5.0, 1.0)])
draws = mix.sample(key, (4,))
```

`transformed` pushes a base distribution through a bijective transform, adding
the log-determinant of the Jacobian to the density:

```python
from probjax.stats import norm, transformed

shifted = transformed(norm(loc=0.0, scale=1.0), lambda x: 2.0 * x + 1.0)
draws = shifted.sample(key, (5,))
log_densities = shifted.logpdf(draws)
```

The transform protocol used here is the same one normalizing flows satisfy;
`TransformedDistribution`, `forward_and_logdet` and `ensure_invertible` are
documented in the [stats reference](../reference/stats.md).

## Fitting

There are two fitting entry points.

The functional `probjax.stats.fit` minimises any
`loss_fn(params, rng, batch)` over a params pytree with Optax. The batch is
either one batch (a bare array, or a dict whose leaves share a leading example
axis) or an iterable of batches:

```python
import jax
import jax.numpy as jnp

from probjax.stats import fit


def loss_fn(params, rng, batch):
    del rng
    return jnp.mean((batch["data"] - params["w"]) ** 2)


data = jax.random.normal(jax.random.key(0), (64, 2))
params, losses = fit(
    loss_fn, {"w": jnp.zeros(2)}, jax.random.key(1), {"data": data},
    num_steps=20, learning_rate=1e-2,
)
assert losses.shape == (20,)
```

Pass `return_result=True` to get a `FitResult` carrying `params`, `losses` and a
resumable `state` (`FitState`), along with `FitInfo` and early-stopping fields:

```python
result = fit(
    loss_fn, {"w": jnp.zeros(2)}, jax.random.key(1), {"data": data},
    num_steps=20, return_result=True,
)
assert result.params["w"].shape == (2,)
```

The object-layer counterpart is `FitMixin.fit(rng, data, *, context=None,
weights=None, **fit_kwargs)`, which the `nn` generative models use so a fitted
module can train in place (`flow.fit(key, samples)`). Generative models are
covered in [Density estimation](density-estimation.md).

Parametric families additionally offer a generator-level MLE:
`norm.fit(data)` returns fitted parameters in the family's canonical order
(`loc, scale`).

## Bridging to generative models

`as_dist()` on a fitted `nn` generative model returns a lazy frozen
distribution with `logpdf` and `sample`, so a trained flow, autoregressive or
diffusion model is scored and sampled exactly like a `probjax.stats`
distribution:

```python
import jax
from flax import nnx

from probjax.nn import maf

flow = maf(2, 2, rngs=nnx.Rngs(0))
distribution = flow.as_dist()
log_densities = distribution.logpdf(jax.random.normal(jax.random.key(3), (4, 2)))
```

- [Density estimation](density-estimation.md) — the shared `fit` / `as_dist` interface
- [Stats reference](../reference/stats.md) — every family, method and protocol
