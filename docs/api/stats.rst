Stats Module
============

The stats module provides probability distributions with a SciPy-like API.
Calling a distribution freezes its parameters; frozen distributions expose
``sample(key, shape=...)`` and density methods such as ``logpdf``.

.. code-block:: python

   import jax
   from probjax.stats import norm

   normal = norm(loc=0.0, scale=1.0)
   samples = normal.sample(jax.random.key(0), shape=(100,))
   log_density = normal.logpdf(samples)

.. module:: probjax.stats

This module includes:

- Continuous and discrete distributions
- Multivariate distributions
- Transformed distributions
- Mixture distributions
- Indep distributions
- Gradient-based distribution fitting

Base Classes
------------

.. autosummary::
   :toctree: generated

   rv_generic
   rv_continuous
   rv_discrete
   rv_multivariate
   rv_exponential_family
   rv_spherical

Continuous Distributions
------------------------

.. autosummary::
   :toctree: generated

   norm
   beta
   gamma
   cauchy
   chi2
   expon
   laplace
   logistic
   uniform
   t
   pareto
   skewnorm
   truncnorm
   gennorm
   genpareto
   vonmises
   watson
   bingham
   wrapcauchy
   dirichlet
   multivariate_normal

Discrete Distributions
----------------------

.. autosummary::
   :toctree: generated

   bernoulli
   binomial
   categorical
   poisson
   geometric
   dirac
   empirical

Flexible Univariate Families
----------------------------

Highly parameterised univariate densities, shaped so a neural network can emit
their parameters directly: each carries its vector parameters in a trailing
axis and reports the lengths through ``param_sizes``. Useful on their own, and
the conditional heads for
:class:`~probjax.nn.generative.autoregressive.AutoregressiveModel`.

.. autosummary::
   :toctree: generated

   mixture_kernel
   logistic_mixture_kernel
   histogram
   tailed_histogram
   spline_normal

Higher-Order Distributions
--------------------------

.. autosummary::
   :toctree: generated

   transformed
   mixture
   indep

Bijective Transforms
--------------------

Pure functions of ``x`` and *natural* parameters (knot positions, slopes,
mixture log-weights), each registering its analytic inverse and
log-determinant. Mapping an unconstrained parameter vector onto them is the job
of the bijector configs in :mod:`probjax.nn.generative.nflows.config`.

.. autosummary::
   :toctree: generated

   bijective.affine
   bijective.monotone
   bijective.monotone_hermite_cubic
   bijective.piecewise_affine
   bijective.rational_linear
   bijective.rational_quadratic

Detailed Documentation
----------------------

.. automodule:: probjax.stats.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.stats.transformed
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.stats.mixture
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.stats.indep
   :members:
   :undoc-members:
   :show-inheritance:
