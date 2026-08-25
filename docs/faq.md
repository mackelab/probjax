---
title: Frequently Asked Questions
---

# Frequently Asked Questions

## General Questions

### What is ProbJax?

ProbJax is a library for probabilistic computation in JAX. It provides:
- SciPy-like probability distributions
- Transformations for tracing and manipulating probabilistic programs
- Flax NNX neural networks and generative models
- MCMC, SMC, filtering, smoothing, and rejection-sampling tools

### Why use ProbJax instead of NumPyro or other PPLs?

ProbJax offers:
- Tight integration with JAX's functional programming paradigm
- Advanced function tracing and automatic inversion capabilities
- Custom primitives for probabilistic programming
- Focus on performance and flexibility

## Installation

### Which Python versions are supported?

ProbJax requires Python 3.11 or later.

### How do I install with GPU support?

For NVIDIA GPUs with CUDA 12:
```bash
python -m pip install -e ".[cuda12]"
```

For Apple Silicon (Metal):
```bash
python -m pip install -e ".[metal]"
```

### I get an error about JAX version

Make sure you have JAX >= 0.4.34 installed:
```bash
pip install --upgrade jax jaxlib
```

## Distributions

### How do I create a simple distribution?

```python
import jax
from probjax.stats import norm

normal = norm(loc=0.0, scale=1.0)
key = jax.random.key(0)
samples = normal.sample(key, shape=(100,))
```

### How do I compute log probabilities?

```python
log_prob = normal.logpdf(samples)
```

### Can I create transformed distributions?

Yes. Pass a frozen base distribution and a bijective JAX callable to
`transformed`:
```python
import jax.numpy as jnp
from probjax.stats import norm, transformed

log_normal = transformed(
    base_dist=norm(loc=0.0, scale=1.0),
    bijector=jnp.exp,
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
- Adaptive and stochastic-gradient MCMC variants

### How do I run MCMC?

```python
import jax
from probjax.inference import MCMC, hmc

kernel = hmc(lambda x: -0.5 * x**2, num_integration_steps=10)
init_key, sample_key = jax.random.split(jax.random.key(0))
state = kernel.init(init_key, 0.0)
params = kernel.init_params(state, step_size=0.1)

runner = MCMC(kernel, verbose=True)
result = runner.sample(sample_key, state, num_samples=1000, params=params)
samples = result.samples
```

### What is SMC?

Sequential Monte Carlo (SMC) uses weighted particle populations. ProbJax
provides fixed-schedule, adaptive geometric, persistent, adaptive persistent,
and path SMC kernels, plus a compiled `SMC` runner and parameter adaptors.

### How do I use filtering algorithms?

Construct one of the filter APIs, such as `kalman_filter`,
`extended_kalman_filter`, `ukf`, or `ParticleFilter`, with the model functions
required by that filter. The resulting `FilterKernel` exposes `init` and
`step`; `smooth` and the dedicated particle and Rauch-Tung-Striebel smoothers
operate on filtering results.

## Neural Networks

### What architectures are available?

- MLP (Multi-Layer Perceptron)
- ResNet
- Transformer
- U-Net
- DeepSet
- LRU (Linear Recurrent Unit)
- Recurrent and attention-based components

### What normalizing flows are available?

- Affine Coupling Flow
- Additive Coupling Flow
- Neural Spline Flow (NSF)
- Neural Autoregressive Flow (NAF)
- Masked Autoregressive Flow (MAF)
- RealNVP presets
- Gaussianization Flow
- Configurable coupling, autoregressive, and elementwise flows

### How do I create a normalizing flow?

```python
from flax import nnx
from probjax import nn

flow = nn.AffineCouplingFlow(
    input_dim=8,
    num_transforms=4,
    rngs=nnx.Rngs(0),
)
```

ProbJax neural-network modules use Flax NNX. Generative families include
normalizing flows, continuous and mean flow matching, continuous diffusion,
and multinomial diffusion. They provide model-specific `loss(...)` methods and
distribution views through `as_dist(event_spec, ...)`.

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
trace_info = traced_fn(x)
```

### What interventions are supported?

- `intervene` / `do`: Fix stochastic sites to specific values
- `condition` / `observe`: Condition on observed values
- `substitute`: General substitution mechanism

## Performance

### How can I speed up my code?

1. Use `jax.jit` on your functions
2. Use `jax.vmap` for batching
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
2. Implement the required density or mass methods and sampling method
3. Add tests
4. Update documentation
