# Glossary

Terms that recur across the guides and tutorials. Each entry links to the page
that uses it in full.

## site

A **site** is a named random variable recorded when a distribution draw happens
inside a ProbJax transformation. Sampling outside a transformation is an
ordinary JAX sample and records nothing — the same call becomes a site only
while a transformation is tracing the model.

A site is referenced by its name in `condition`, `observe`, `intervene`, `do`,
`substitute` and the auto-naming `scope`; it is returned by `joint_sample` and
described by `trace(model, sites=True)`. See
[Probabilistic programs](guides/ppl.md).

## `name=`

The keyword that assigns a site's name to a draw:

```python
import jax

from probjax.stats import norm

key = jax.random.key(0)
z = norm.rvs(key, 0.0, 1.0, name="z")
```

Without `name=`, the draw is anonymous: it samples normally but cannot be
conditioned on or referenced by the transformations. Under `scope("outer")` an
unnamed draw gets an auto-generated name such as `outer__norm_0`.

## frozen / view

A generator such as `norm` is an **unfrozen** object: the parameters are passed
positionally at call time (`norm.rvs(key, loc, scale)`). Calling it with fixed
parameters returns an **`rv_frozen`** instance that carries those parameters and
exposes `logpdf`, `sample` and the other distribution methods:

```python
import jax

from probjax.stats import norm

key = jax.random.key(0)
frozen = norm(loc=0.0, scale=1.0)  # an rv_frozen
log_density = frozen.logpdf(0.0)
draws = frozen.sample(key, (1000,))
```

The frozen object is a *view*: it binds parameters, it does not copy data.
`as_dist()` on an `nn` model returns a lazy frozen distribution in the same
sense — a distribution view of a trained module with its own `logpdf` and
`sample`. See [Distributions](guides/stats-distributions.md) and the
[stats reference](reference/stats.md).

## `event_spec`

The shape/spec describing **one event**, excluding the sample and batch axes.
An integer is shorthand for a feature vector; a tuple describes an image or
sequence event. Diffusion and flow-matching constructors require it:

```python
from flax import nnx

from probjax.nn import EDM, LinearFlow


class Denoiser(nnx.Module):
    def __call__(self, t, x):
        return 0.1 * x


class Velocity(nnx.Module):
    def __call__(self, t, x, **kwargs):
        return 0.1 * x


diffusion = EDM(Denoiser(), event_spec=(8, 3), num_steps=2)
flow = LinearFlow(Velocity(), event_spec=3)
```

`model.as_dist(event_spec=...)` overrides one distribution view; `event_spec`
does not resize network weights. See
[Diffusion and flow matching](reference/nn.md#diffusion-and-flow-matching).

## trace / jaxpr

A **trace** is a recording of a program's execution. ProbJax transformations
run the model under a custom interpreter that sees the random-variable
primitive as it is staged out into a **jaxpr** (JAX's intermediate
representation). `probjax.core.trace(fun, sites=True)` returns a callable that
runs the traced program and reports its sites. See
[Program inversion](guides/program-inversion.md) for how a jaxpr is walked
backwards to build an inverse.

## log-density vs log-potential

The **log-density** of a site is the log of its distribution evaluated at the
sampled value; `trace(..., sites=True)` exposes it per site. The **log
potential** is the sum of the density terms over a program — the unnormalised
target that inference algorithms score. `log_joint_fn` and `log_potential_fn`
return functions of the program's sites as keyword arguments; the potential
drops the density term of any causally intervened site. See
[Probabilistic programs](guides/ppl.md).

## `as_dist`

`as_dist()` turns a fitted `nn` generative model into a frozen distribution with
`logpdf` and `sample`. It bridges density estimation and the distribution API:
the returned view can be scored and sampled like any `probjax.stats`
distribution. See [Density estimation](guides/density-estimation.md) and
[Distributions](guides/stats-distributions.md).

## See also

- [Overview](overview.md) — how the modules fit together
- [Probabilistic programs](guides/ppl.md) — sites and transformations in practice
