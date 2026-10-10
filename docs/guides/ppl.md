# Probabilistic programs

A ProbJax model is an ordinary JAX function. What makes it *probabilistic* is
that drawing a named value under a ProbJax transformation records a **site**, and
the transformations then rewire the program: condition on data, intervene
causally, fix values, or rename auto-generated draws. The same model samples
jointly, scores, and drives inference without being rewritten.

```python
import jax

from probjax.core import joint_sample
from probjax.stats import norm


def model(key):
    k1, k2 = jax.random.split(key)
    z = norm.rvs(k1, 0.0, 1.0, name="z")
    y = norm.rvs(k2, z, 0.5, name="y")
    return y


samples = joint_sample(model)(jax.random.key(0))
assert set(samples) == {"z", "y"}
```

`norm.rvs(key, *params, name=...)` is the public wrapper for a random-variable
primitive. It is exactly equivalent to the lower-level
`rv_p.bind(k1, 0.0, 1.0, dist=norm, name="z")` from
`probjax.core.custom_primitives.random_variable`; both record a site only while
a ProbJax transformation is active, and both are ordinary JAX samples
otherwise. This guide uses the `norm.rvs` form; the
[PPL tutorial](../tutorials/ppl.md) uses the `rv_p.bind` form and shows the same
results. The [Glossary](../glossary.md) defines site and `name=`.

## Tracing sites

`trace(fun, sites=True)` returns a callable that runs the program and reports
every site it saw. Each site is described by its name, kind, value, log-density,
distribution and shape:

```python
from probjax.core import trace

sites = trace(model, sites=True)(jax.random.key(1))
assert set(sites) == {"z", "y"}
assert sites["z"]["name"] == "z"
assert sites["z"]["kind"] == "sample"
assert sites["z"]["value"].shape == ()
```

## Joint sampling and densities

`joint_sample(model)` returns a callable that draws every site. The density
transformations return functions of the **model's sites as keyword arguments** —
there is no positional `log_joint_fn(model)(key)` form.

```python
import jax.numpy as jnp

from probjax.core import log_joint_fn, log_potential_fn

samples = joint_sample(model)(jax.random.key(2))
log_joint = log_joint_fn(model)(**samples)  # z=..., y=...

# `log_potential_fn` is the same engine for a custom sampling function; sites
# are still keyword arguments of the returned callable.
partial = log_potential_fn(model, allow_partial=True)(z=jnp.asarray(0.1))
assert partial.shape == ()
```

By default `log_joint_fn` is strict: every site must be supplied, or it raises
`KeyError`. Pass `allow_partial=True` to score only the sites you provide.

## Observe and condition

`observe(model, observations)` (aliased `condition`) fixes a site to an
observed value and **keeps** its log-density term. The returned object is a new
model that can be sampled and scored like any other:

```python
from probjax.core import observe

obs_y = jnp.asarray(0.25)
observed = observe(model, {"y": obs_y})
latent = joint_sample(observed)(jax.random.key(3))
log_joint = log_joint_fn(observed)(z=latent["z"])
```

This is the standard Bayesian setup: observe the data, sample the latent, score
the joint.

## Intervene, do and substitute

`intervene(model, rvs)` (aliased `do`) sets a site *causally* and **drops** its
own log-density term. `substitute` is the general operation, with `mode=`
selecting between the two behaviours:

- `mode="condition"` (alias `"replay"`) keeps the substituted site's density —
  equivalent to `observe`.
- `mode="do"` (alias `"intervene"`) drops it — equivalent to `intervene`/`do`.

```python
from probjax.core import do, intervene, substitute

fixed_z = jnp.asarray(-0.4)

# Causal intervention: z's own density is removed.
do_model = do(model, {"z": fixed_z})
do_samples = joint_sample(do_model)(jax.random.key(4))
do_log_joint = log_joint_fn(do_model)(y=do_samples["y"])

# Substitution in `condition` mode keeps z's density term.
conditioned = substitute(model, {"z": fixed_z}, mode="condition")
conditioned_samples = joint_sample(conditioned)(jax.random.key(5))
conditioned_log_joint = log_joint_fn(conditioned)(y=conditioned_samples["y"])
```

The distinction matters: conditioning treats the fixed value as *evidence* and
scores it; intervening treats it as an *input* set from outside the model. See
the [PPL tutorial](../tutorials/ppl.md) for a side-by-side comparison.

## Scoped names

Draws without an explicit `name=` still record a site while under a
transformation. `scope(name)` is a context manager that prefixes those
auto-generated names, so nested programs produce stable, namespaced sites:

```python
from probjax.core import scope


def scoped_model(key):
    with scope("outer"):
        z = norm.rvs(key, 0.0, 1.0)
    return z


drawn = joint_sample(scoped_model)(jax.random.key(6))
assert set(drawn) == {"outer__norm_0"}
```

## Composing with inference

The same `log_potential_fn` a model produces is what the inference algorithms
consume, so a model written here runs under MCMC, SMC and variational inference
without changes. An intervened model gives a potential with the causal site's
term removed, which is how counterfactual and abduction queries are written.

- [Program inversion](program-inversion.md) — inverting a traced program
- [Inference](inference.md) — MCMC and variational inference
- [Core reference](../reference/core.md) — every transformation and alias
