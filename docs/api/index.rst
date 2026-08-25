API Reference
=============

This section documents the public ProbJax modules and their exported APIs.

.. toctree::
   :maxdepth: 2

   core
   stats
   nn
   inference
   utils

Module Overview
---------------

ProbJax consists of several main modules:

**probjax.core**
   Core functionality for tracing, inversion, and JAXPR manipulation.

**probjax.stats**
   Probability distributions, fitting, constraints, and bijective transforms
   with a SciPy-like frozen-distribution API.

**probjax.nn**
   Flax NNX layers and architectures, generative models, and training losses.

**probjax.inference**
   MCMC and SMC kernels and runners, filtering and smoothing, adaptation, and
   rejection sampling.

**probjax.utils**
   ODE/SDE integration, linear algebra, interpolation, graph, solver, special,
   JAX, and typing utilities.
