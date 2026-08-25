ProbJax Documentation
=====================

ProbJax is a JAX-native research toolbox for probabilistic computation. It
brings together probabilistic-program transformations, SciPy-style
distributions, automatic inversion, inference algorithms, Flax NNX generative
models, and numerical solvers.

.. note::

   ProbJax is research software. APIs may evolve, and accelerator-specific
   kernels and sharding features are experimental.

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
   :caption: Guides

   faq
   troubleshooting
   contributing

.. toctree::
   :maxdepth: 1
   :caption: Project Information

   changelog

Capabilities
------------

**Probabilistic programs**
   Named random-variable sites, tracing, sampling, conditioning, observation,
   intervention, replay, scopes, and joint log-density construction.

**Statistics**
   Continuous, discrete, multivariate, mixture, independent, and transformed
   distributions with SciPy-style methods, parameter fitting, bijections, and
   divergences.

**Inference**
   MCMC and adaptation, SMC and tempering, Kalman and particle filtering,
   smoothing, and rejection sampling.

**Neural and generative models**
   Flax NNX architectures, normalizing flows, diffusion and score models, flow
   matching, categorical diffusion, sharding, and accelerator kernels.

**Numerical utilities**
   ODE/SDE integration, interpolation, root finding, special functions, graph
   utilities, and linear algebra.

Quick Example
-------------

.. code-block:: python

   import jax
   from probjax.stats import norm

   normal = norm(loc=0.0, scale=1.0)
   samples = normal.sample(jax.random.key(0), shape=(1_000,))
   log_density = normal.logpdf(samples)

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
