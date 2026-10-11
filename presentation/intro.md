# ProbJax

### Probabilistic computation in JAX

A JAX-native research toolbox for probabilistic programming, program inversion,
SciPy-style distributions, inference, and Flax NNX generative models.

---

## The problem

Probabilistic building blocks live in separate ecosystems:

- distributions and fitting in SciPy-style libraries,
- probabilistic-program transformations in PPLs,
- invertible programs and Jacobian accounting elsewhere,
- generative models in deep-learning frameworks,
- inference algorithms in yet more packages.

They rarely compose: a JAX transformation that works in one does not flow
through the others. ProbJax composes them in **one JAX-native codebase**, so the
same traced function can be sampled, conditioned, inverted, scored, and fit.

---

## Core idea 1 — PPL + invertible programs

**Probabilistic programs.** Write an ordinary JAX function; `trace`, `condition`,
`observe`, `intervene`, and `joint_sample` it. Sampling a distribution inside a
transformation records a named random-variable site; outside one it is a plain
JAX sample.

**Program inversion.** `inverse` and `inverse_and_logabsdet` walk a jaxpr
backwards to build the inverse of a function, with `custom_inverse` for the
parts that need an analytic rule. Inversion carries pytrees, batching, and
selected control-flow primitives.

```python
from probjax.core import inverse_and_logabsdet
x, inverse_logdet = inverse_and_logabsdet(transform)(y)
```

---

## Core idea 2 — `probjax.nn` + `probjax.inference`

**`probjax.nn`** — Flax NNX layers, architectures, and generative models:
normalizing flows, diffusion and score models, continuous and mean flow
matching, categorical diffusion, attention/SSD/Mamba kernels, and sharding.

```python
from probjax.nn import MLP
```

**`probjax.inference`** — MCMC kernels and adaptation, compiled MCMC runners,
SMC and tempering, Kalman and particle filters, smoothers, and rejection
samplers.

---

## Demo notebooks

A curated set rendered as part of the docs:

- `examples/core/ppl.ipynb` — named sites, tracing, conditioning, joint sampling.
- `examples/nn/generative/diffusion.ipynb` — Flax NNX diffusion model.
- `examples/stats/distributions_basic.ipynb` — SciPy-style distributions.

All runnable notebooks live under `examples/`. See the repository README for the
current state of the remaining examples.

---

## Install & resources

```bash
pip install probjax
```

- **Docs:** https://www.mackelab.org/probjax/
- **Source:** https://github.com/mackelab/probjax

ProbJax targets Python 3.11–3.13 and JAX `>=0.9.0,<0.10`.
