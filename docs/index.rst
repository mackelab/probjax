Welcome to ProbJax's documentation!
==================================

ProbJax is a powerful library for probabilistic computation in JAX, designed to simplify the development of probabilistic models and inference algorithms. It provides a comprehensive set of tools for building, training, and deploying probabilistic models with high performance and flexibility.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 1
   :caption: More

   contributing

Key Features
------------

**Core Functionality**
  - Advanced function tracing and manipulation capabilities
  - Robust automatic function inversion with support for complex transformations
  - Efficient computation of log-probabilities and transformed distribution handling

**Distributions**
  - Comprehensive set of probability distributions
  - Support for sampling, log-probability computation, and distribution transformations
  - Integration with JAX's functional programming paradigm

**Neural Networks**
  - Standard architectures (Transformers, ResNets, U-Nets)
  - Specialized layers for normalizing flows
  - Coupling and autoregressive layers
  - Custom layer implementations

**Inference**
  - Various inference algorithms
  - Support for variational inference
  - MCMC sampling capabilities

**Utilities**
  - Numerical computation tools (ODE/SDE integration)
  - Optimization utilities
  - Performance monitoring and benchmarking

Quick Example
-------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from probjax import distributions as dist

   # Create a simple normal distribution
   normal = dist.Normal(loc=0.0, scale=1.0)

   # Sample from the distribution
   key = jax.random.PRNGKey(0)
   samples = normal.sample(key, sample_shape=(1000,))

   # Compute log probability
   log_prob = normal.log_prob(samples)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
