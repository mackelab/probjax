---
title: Quick Start
---

# Quick Start

This guide will help you get started with ProbJax quickly.

## Basic Setup

First, install ProbJax:

```bash
pip install -e probjax
```

## Working with Distributions

ProbJax provides a comprehensive set of probability distributions:

```python
import jax
import jax.numpy as jnp
from probjax import distributions as dist

# Create a normal distribution
normal = dist.Normal(loc=0.0, scale=1.0)

# Sample from the distribution
key = jax.random.PRNGKey(0)
samples = normal.sample(key, sample_shape=(1000,))

# Compute log probability
log_prob = normal.log_prob(samples)

# Compute mean and variance
mean = normal.mean()
variance = normal.var()
```

## Multiple Distributions

ProbJax supports various distributions:

```python
# Continuous distributions
beta = dist.Beta(a=2.0, b=5.0)
gamma = dist.Gamma(a=2.0, scale=1.0)
uniform = dist.Uniform(low=0.0, high=1.0)

# Discrete distributions
bernoulli = dist.Bernoulli(probs=0.7)
poisson = dist.Poisson(rate=5.0)

# Multivariate distributions
mvn = dist.MultivariateNormal(
    loc=jnp.zeros(3),
    covariance_matrix=jnp.eye(3)
)
```

## Transforming Distributions

Apply transformations to create new distributions:

```python
from probjax.stats import transformed

# Create a log-normal distribution
log_normal = transformed(
    dist.Normal(0.0, 1.0),
    transform=jnp.exp
)

# Sample from log-normal
samples = log_normal.sample(key, sample_shape=(100,))
```

## Neural Networks

ProbJax includes neural network modules built on Flax:

```python
from probjax.nn import MLP, layers

# Create a simple MLP
mlp = MLP(features=[64, 32, 10])

# Initialize with random parameters
key, subkey = jax.random.split(key)
params = mlp.init(subkey, jnp.ones((1, 784)))

# Forward pass
output = mlp.apply(params, jnp.ones((1, 784)))
```

## Inference Algorithms

ProbJax provides various inference methods:

```python
from probjax.inference import mcmc

# Example: MCMC sampling
# See the examples directory for detailed tutorials
```

## Core Transformations

ProbJax provides powerful transformation capabilities:

```python
from probjax.core import inverse, trace, log_prob_fn

# Automatic function inversion
def f(x):
    return jnp.exp(x) + 1

# Compute the inverse
f_inv = inverse(f)
result = f_inv(2.0)  # Should be log(1) = 0

# Trace function execution
traced_fn = trace(f)
result, trace_info = traced_fn(1.0)

# Compute log probability of transformed variables
log_prob_transformed = log_prob_fn(f)(1.0, dist.Normal(0.0, 1.0))
```

## Next Steps

- Explore the [API Reference](api/index.md) for detailed documentation
- Check out the [Examples](examples.md) for more tutorials
- Visit the `examples/` directory in the repository for Jupyter notebooks
