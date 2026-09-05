# ProbJax

[![CI](https://github.com/mackelab/probjax/actions/workflows/ci.yml/badge.svg)](https://github.com/mackelab/probjax/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/probjax/badge/?version=latest)](https://probjax.readthedocs.io/en/latest/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt)

📖 **Documentation:** https://probjax.readthedocs.io/en/latest/

> [!WARNING]
> ProbJax is a highly experimental research toolbox. Breaking API changes,
> deprecations, and untested code are common. Do not rely on API stability or
> production readiness.

ProbJax is a JAX-native research toolbox for probabilistic computation. It
combines SciPy-style probability distributions, transformations for
probabilistic programs, automatic inversion and Jacobian accounting, inference
algorithms, Flax NNX models, and numerical solvers in one composable codebase.

## Highlights

- **Probabilistic programs:** named random-variable sites, tracing, sampling,
  conditioning, observation, intervention, replay, scopes, and joint log
  densities.
- **Program inversion:** JAXPR-based inversion and inverse log-absolute
  determinants, including custom inverse rules, pytrees, batching, and selected
  control-flow primitives.
- **Statistics:** continuous, discrete, multivariate, mixture, independent, and
  transformed distributions with a SciPy-style API, parameter constraints,
  fitting utilities, bijections, and divergences.
- **Inference:** MCMC kernels and adaptation, compiled MCMC runners, SMC and
  tempering methods, Kalman and particle filters, smoothers, and rejection
  samplers.
- **Generative models:** normalizing flows, diffusion and score models,
  continuous and mean flow matching, and categorical diffusion built with
  Flax NNX.
- **Numerics:** ODE and SDE integration, interpolation, root finding, special
  functions, graph utilities, and linear algebra.
- **Accelerated neural components:** attention, recurrent/SSD/Mamba components,
  Pallas kernels, model sharding, and export-oriented sampling utilities.

## Installation

ProbJax currently targets Python 3.11 or newer. From a source checkout:

```bash
pip install probjax
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv pip install probjax
```

Optional environments:

```bash
# NVIDIA CUDA 13
python -m pip install "probjax[cuda13]"

# Apple Silicon via jax-metal
python -m pip install "probjax[metal]"

# Source checkout with tests and development tools
git clone https://github.com/mackelab/probjax.git
cd probjax
python -m pip install -e ".[dev]"
```

JAX accelerator compatibility depends on the installed JAX backend, driver,
and hardware. Metal support is exposed as an optional dependency but is not
covered by the repository's Linux test environment. See
[`docs/getting-started.md`](docs/getting-started.md) for details.

## Quick Start

### Distributions

ProbJax distributions use lowercase SciPy-style names and methods such as
`sample`, `logpdf`, `cdf`, and `ppf` where implemented.

```python
import jax
from probjax.stats import norm

normal = norm(loc=0.0, scale=1.0)
key = jax.random.key(0)

samples = normal.sample(key, shape=(1_000,))
log_density = normal.logpdf(samples)
```

The unfrozen form is also available:

```python
samples = norm.rvs(key, 0.0, 1.0, shape=(1_000,))
log_density = norm.logpdf(samples, 0.0, 1.0)
```

### Probabilistic Programs

Sampling through a distribution creates a named random-variable site under
ProbJax transformations while remaining an ordinary JAX sample in eager code.

```python
import jax
import jax.numpy as jnp
from probjax.core import condition, joint_sample, log_joint_fn
from probjax.stats import norm

def model(key):
    key_z, key_y = jax.random.split(key)
    z = norm.rvs(key_z, 0.0, 1.0, name="z")
    return norm.rvs(key_y, z, 0.5, name="y")

observed_model = condition(model, {"y": jnp.asarray(0.25)})
latent = joint_sample(observed_model)(jax.random.key(1))["z"]
log_joint = log_joint_fn(observed_model)(z=latent)
```

### Automatic Inversion

```python
import jax.numpy as jnp
from probjax.core import inverse, inverse_and_logabsdet

def transform(x):
    return jnp.exp(2.0 * x + 1.0)

y = transform(jnp.asarray(0.4))
x = inverse(transform)(y)
x, inverse_logdet = inverse_and_logabsdet(transform)(y)
```

## Package Map

| Package | Purpose |
| --- | --- |
| `probjax.core` | Probabilistic-program transformations, JAXPR propagation, and inversion |
| `probjax.stats` | Distributions, fitting, bijections, constraints, and divergences |
| `probjax.inference` | MCMC, SMC, adaptation, filtering, smoothing, and rejection sampling |
| `probjax.nn` | Flax NNX layers, architectures, generative models, kernels, and sharding |
| `probjax.utils` | ODE/SDE solvers, special functions, interpolation, graphs, and linear algebra |

## Examples And Documentation

Full documentation (guides, API reference, troubleshooting) is hosted at
https://probjax.readthedocs.io/en/latest/.

- [`examples/core`](examples/core): tracing and probabilistic-program examples
- [`examples/stats`](examples/stats): distribution examples
- [`examples/inference`](examples/inference): MCMC, SMC, and filtering examples
- [`examples/nn`](examples/nn): flows, diffusion, flow matching, and transformers
- [`examples/utils`](examples/utils): ODE, SDE, and special-function examples
- [`docs`](docs): documentation source (getting started, guides, reference)

Some notebooks predate recent API refactors and are being migrated. Prefer the
README, getting-started documentation, source API reference, and tests when an
example disagrees with the current package.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

Contributions are welcome. See [`docs/contributing.md`](docs/contributing.md)
for the development workflow.

## License

ProbJax is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt).

## Citation

```bibtex
@software{probjax2024,
  author = {Manuel Gloeckler},
  title = {ProbJax: Probabilistic computation in JAX},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/mackelab/probjax}
}
```
