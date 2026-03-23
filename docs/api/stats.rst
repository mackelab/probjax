Stats Module
============

The stats module provides a comprehensive set of probability distributions with a SciPy-like API.

.. module:: probjax.stats

This module includes:

- Continuous and discrete distributions
- Multivariate distributions
- Transformed distributions
- Mixture distributions
- Independent distributions

Base Classes
------------

.. autosummary::
   :toctree: generated

   base.rv_generic
   base.rv_continuous
   base.rv_discrete
   base.rv_multivariate
   base.rv_exponential_family
   base.rv_spherical

Continuous Distributions
------------------------

.. autosummary::
   :toctree: generated

   continuous.norm.norm
   continuous.beta.beta
   continuous.gamma.gamma
   continuous.cauchy.cauchy
   continuous.chi2.chi2
   continuous.expon.expon
   continuous.laplace.laplace
   continuous.logistic.logistic
   continuous.uniform.uniform
   continuous.t.t
   continuous.pareto.pareto
   continuous.skewnorm.skewnorm
   continuous.vonmises.vonmises
   continuous.truncnorm.truncnorm
   continuous.gennorm.gennorm
   continuous.genpareto.genpareto
   continuous.watson.watson
   continuous.wrapcauchy.wrapcauchy
   continuous.bingham.bingham
   continuous.dirichlet.dirichlet
   continuous.multivariate_normal.multivariate_normal

Discrete Distributions
----------------------

.. autosummary::
   :toctree: generated

   discrete.bernoulli.bernoulli
   discrete.binomial.binomial
   discrete.categorical.categorical
   discrete.poisson.poisson
   discrete.geometric.geometric
   discrete.dirac.dirac
   discrete.empirical.empirical

Higher-Order Distributions
--------------------------

.. autosummary::
   :toctree: generated

   transformed.transformed
   mixture.mixture
   independent.independent

Detailed Documentation
----------------------

.. automodule:: probjax.stats.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.stats.continuous
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: probjax.stats.discrete
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

.. automodule:: probjax.stats.independent
   :members:
   :undoc-members:
   :show-inheritance:
