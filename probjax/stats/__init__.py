"""
Statistical Distributions (:mod:`probjax.stats`)
===============================================

This module contains various probability distributions implemented
in JAX with a SciPy-like API.
"""

from probjax.stats.base import (
    rv_generic,
    rv_continuous,
    rv_discrete,
    rv_exponential_family,
    rv_continuous_frozen,
    rv_discrete_frozen,
)

# Import all implemented distributions
from probjax.stats.norm import norm
from probjax.stats.gamma import gamma
from probjax.stats.beta import beta
from probjax.stats.expon import expon
from probjax.stats.laplace import laplace
from probjax.stats.uniform import uniform

# Import discrete distributions
from probjax.stats.bernoulli import bernoulli
from probjax.stats.binomial import binomial
from probjax.stats.categorical import categorical
from probjax.stats.poisson import poisson
from probjax.stats.geometric import geometric
from probjax.stats.dirac import dirac
from probjax.stats.empirical import empirical

# Import higher-order distributions
from probjax.stats.independent import independent
from probjax.stats.transformed import transformed
from probjax.stats.mixture import mixture

__all__ = [
    # Base classes
    'rv_generic',
    'rv_continuous',
    'rv_discrete',
    'rv_exponential_family',
    'rv_continuous_frozen',
    'rv_discrete_frozen',
    # Continuous distributions
    'norm',
    'gamma',
    'beta',
    'expon',
    'laplace',
    'uniform',
    # Discrete distributions
    'bernoulli',
    'binomial',
    'categorical',
    'poisson',
    'geometric',
    'dirac',
    'empirical',
    # Higher-order distributions
    'independent',
    'transformed',
    'mixture',
]
