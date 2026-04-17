---
title: Frequently Asked Questions
---

# Frequently Asked Questions

## General Questions

### What is ProbJax?

ProbJax is a powerful library for probabilistic computation in JAX. It provides tools for:
- Building probabilistic models with automatic inference
- Working with probability distributions
- Implementing neural networks and normalizing flows
- Running MCMC, SMC, and filtering algorithms

### Why use ProbJax instead of NumPyro or other PPLs?

ProbJax offers:
- Tight integration with JAX's functional programming paradigm
- Advanced function tracing and automatic inversion capabilities
- Custom primitives for probabilistic programming
- Comprehensive set of inference algorithms
- Focus on performance and flexibility

## Installation

### Which Python versions are supported?

ProbJax requires Python 3.11 or later.

### How do I install with GPU support?

For NVIDIA GPUs with CUDA 12:
```bash
pip install -e "probjax[cuda12]"
```

For Apple Silicon (Metal):
```bash
pip install -e "probjax[metal]"
```

### I get an error about JAX version

Make sure you have JAX >= 0.4.34 installed:
```bash
pip install --upgrade jax jaxlib
```

## Distributions

### How do I create a simple distribution?

```python
from probjax import distributions as dist
import jax.numpy as jnp

# Create a normal distribution
normal = dist.Normal(loc=0.0, scale=1.0)

# Or use the functional form
samples = dist.norm.sample(key, sample_shape=(100,))
```

### How do I compute log probabilities?

```python
log_prob = normal.log_prob(samples)
```

### Can I create transformed distributions?

Yes, use the `transformed` function:
```python
from probjax import transformed

log_normal = transformed(
    dist.Normal(0.0, 1.0),
    transform=jnp.exp
)
```

### What distributions are available?

Continuous: `norm`, `gamma`, `beta`, `uniform`, `expon`, `laplace`, `logistic`, `cauchy`, `chi2`, `t`, `pareto`, `skewnorm`, `truncnorm`, `gennorm`, `genpareto`, `vonmises`, `watson`, `bingham`, `wrapcauchy`, `dirichlet`, `multivariate_normal`

Discrete: `bernoulli`, `binomial`, `categorical`, `poisson`, `geometric`, `dirac`, `empirical`

## Inference

### What MCMC algorithms are available?

- HMC (Hamiltonian Monte Carlo)
- NUTS (No-U-Turn Sampler)
- MALA (Metropolis-Adjusted Langevin Algorithm)
- Slice sampling
- Elliptical slice sampling
- MCLMC (Microcanonical Langevin Monte Carlo)
- Gibbs sampling
- And more...

### How do I run MCMC?

```python
from probjax.inference import mcmc, MCMC

# Create a kernel
kernel = mcmc.hmc(logdensity_fn=lambda x: -0.5 * x**2, step_size=0.1, num_integration_steps=10)

# Create runner with progress bar
runner = MCMC(kernel, verbose=True)

# Run
key = jax.random.PRNGKey(0)
state = kernel.init_state(key, 0.0)
final_state = runner.run(key, state, num_steps=1000)
```

### What is SMC?

Sequential Monte Carlo (SMC) is a family of algorithms that use sequential importance sampling. ProbJax provides:
- Standard SMC with geometric tempering
- Adaptive SMC
- Persistent SMC
- Path SMC

### How do I use filtering algorithms?

```python
from probjax.inference import kalman_filter

# Create filter
kf = kalman_filter(...)
```

## Neural Networks

### What architectures are available?

- MLP (Multi-Layer Perceptron)
- ResNet
- Transformer
- U-Net
- DeepSet
- LRU (Linear Recurrent Unit)
- And more...

### What normalizing flows are available?

- Affine Coupling Flow
- Additive Coupling Flow
- Neural Spline Flow (NSF)
- Neural Autoregressive Flow (NAF)
- Masked Autoregressive Flow (MAF)
- RealNVP
- Gaussianization Flow
- And more...

### How do I create a normalizing flow?

```python
from probjax import nn

flow = nn.AffineCouplingFlow(
    event_shape=(28, 28),
    num_layers=4,
    hidden_features=[128, 128]
)
```

## Core Features

### What is automatic function inversion?

ProbJax can automatically invert functions and compute their log-determinants:
```python
from probjax import inverse, inverse_and_logabsdet

f_inv = inverse(f)
x_recovered = f_inv(y)

# With log-determinant
f_inv_logdet = inverse_and_logabsdet(f)
x, logdet = f_inv_logdet(y)
```

### What is function tracing?

Tracing allows you to inspect the execution of probabilistic programs:
```python
from probjax import trace

traced_fn = trace(f)
result, trace_info = traced_fn(x)
```

### What interventions are supported?

- `intervene` / `do`: Fix stochastic sites to specific values
- `condition` / `observe`: Condition on observed values
- `substitute`: General substitution mechanism

## Performance

### How can I speed up my code?

1. Use `jax.jit` on your functions
2. Use the `vmap` for batching
3. Enable XLA optimizations
4. Use appropriate hardware (GPU/TPU)

### How do I enable GPU acceleration?

For NVIDIA GPUs:
```bash
pip install jax[cuda12]
```

For Apple Silicon:
```bash
pip install jax-metal
export JAX_PLATFORMS=metal,cpu
```

### My code is slow - what should I check?

- Ensure you're using `jax.jit`
- Check for unnecessary Python overhead
- Use efficient data structures
- Profile your code with JAX profiler

## Troubleshooting

### I get a shape error

Make sure your input shapes match the expected shapes. Use `jax.eval_shape` to debug:
```python
from jax import eval_shape
print(eval_shape(fn, *args))
```

### I get a tracer error

This usually means you're trying to use JAX tracers in non-JAX contexts. Make sure all operations are JAX-compatible.

### Documentation for a function is missing

We're actively improving documentation. If you find missing documentation:
1. Check the source code directly
2. Look at the examples
3. Open an issue on GitHub

## Contributing

### How can I contribute?

See our [Contributing Guide](contributing.md) for details on:
- Setting up your development environment
- Code style requirements
- Testing procedures
- Pull request process

### How do I report a bug?

Open an issue on GitHub with:
1. A minimal reproducible example
2. Your environment (Python version, JAX version, OS)
3. The expected vs actual behavior
4. Any error messages

### Can I add a new distribution?

Yes! See existing distributions in `probjax/stats/` for examples. Make sure to:
1. Inherit from appropriate base class (`rv_continuous`, `rv_discrete`, etc.)
2. Implement required methods (`sample`, `log_prob`)
3. Add tests
4. Update documentation
