Stats Module
============

The stats module provides a comprehensive set of probability distributions with a SciPy-like API.

.. module:: probjax.stats

This module includes:

- Continuous and discrete distributions
- Multivariate distributions
- Transformed distributions
- Mixture distributions
- Indep distributions

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

Higher-Order Distributions
--------------------------

.. autosummary::
   :toctree: generated

   transformed
   mixture
    indep

Bijective Transforms
--------------------

.. autosummary::
   :toctree: generated

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
