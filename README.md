# Probjax

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt)

Probjax is a powerful library for probabilistic computation in JAX, designed to simplify the development of probabilistic models and inference algorithms. It provides a comprehensive set of tools for building, training, and deploying probabilistic models with high performance and flexibility.

## Features

### Core Functionality
- **Tracing**: Advanced function tracing and manipulation capabilities
- **Automatic Inversion**: Robust automatic function inversion with support for complex transformations
- **Automatic Log-Probability**: Efficient computation of log-probabilities and transformed distribution handling

### Distributions
- Comprehensive set of probability distributions
- Support for sampling, log-probability computation, and distribution transformations
- Integration with JAX's functional programming paradigm

### Neural Networks
Built on top of [Haiku](https://github.com/deepmind/dm-haiku), featuring:
- Standard architectures (Transformers, ResNets, U-Nets)
- Specialized layers for normalizing flows
- Coupling and autoregressive layers
- Custom layer implementations

### Inference
- Various inference algorithms
- Support for variational inference
- MCMC sampling capabilities

### Utilities
- Numerical computation tools (ODE/SDE integration)
- Optimization utilities
- Performance monitoring and benchmarking

## Installation

### Basic Installation
```bash
pip install -e probjax
```

For CUDA 12 support with GPU acceleration:
```bash
pip install -e "probjax[cuda12]"
```

### Development Installation
For development and testing:
```bash
pip install -e "probjax[dev]"
```


## Quick Start

```python
import jax
import jax.numpy as jnp
from probjax import distributions as dist
from probjax.nn import layers

# Create a simple normal distribution
normal = dist.Normal(loc=0.0, scale=1.0)

# Sample from the distribution
key = jax.random.PRNGKey(0)
samples = normal.sample(key, sample_shape=(1000,))

# Compute log probability
log_prob = normal.log_prob(samples)
```

## Examples

Check out the `examples/` directory for detailed tutorials and use cases:
- `basics/`: Basic usage examples
- `core/`: Core functionality demonstrations
- `distributions/`: Distribution examples
- `nn/`: Neural network implementations
- `utils/`: Utility function examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Citation

If you use Probjax in your research, please cite:

```bibtex
@software{probjax2024,
  author = {Manuel Gloeckler},
  title = {Probjax: Probabilistic computation in JAX},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/probjax}
}
```
