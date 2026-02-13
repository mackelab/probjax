"""
Statistical Distributions (:mod:`probjax.stats`)
===============================================

This module contains various probability distributions implemented
in JAX with a SciPy-like API.
"""

from probjax.stats.base import (
    rv_continuous,
    rv_continuous_frozen,
    rv_discrete,
    rv_discrete_frozen,
    rv_exponential_family,
    rv_generic,
    rv_multivariate,
    rv_multivariate_frozen,
    rv_spherical,
    rv_spherical_frozen,
)
from probjax.stats.continuous.beta import beta
from probjax.stats.continuous.bingham import bingham
from probjax.stats.continuous.cauchy import cauchy
from probjax.stats.continuous.chi2 import chi2
from probjax.stats.continuous.dirichlet import dirichlet
from probjax.stats.continuous.expon import expon
from probjax.stats.continuous.gamma import gamma
from probjax.stats.continuous.gennorm import gennorm
from probjax.stats.continuous.genpareto import genpareto
from probjax.stats.continuous.laplace import laplace
from probjax.stats.continuous.logistic import logistic
from probjax.stats.continuous.multivariate_normal import multivariate_normal

# Import all implemented distributions
from probjax.stats.continuous.norm import norm
from probjax.stats.continuous.pareto import pareto
from probjax.stats.continuous.skewnorm import skewnorm
from probjax.stats.continuous.t import t
from probjax.stats.continuous.truncnorm import truncnorm
from probjax.stats.continuous.uniform import uniform
from probjax.stats.continuous.vonmises import vonmises
from probjax.stats.continuous.watson import watson
from probjax.stats.continuous.wrapcauchy import wrapcauchy

# Import discrete distributions
from probjax.stats.discrete.bernoulli import bernoulli
from probjax.stats.discrete.binomial import binomial
from probjax.stats.discrete.categorical import categorical
from probjax.stats.discrete.dirac import dirac
from probjax.stats.discrete.geometric import geometric
from probjax.stats.discrete.poisson import poisson
from probjax.stats.discrete.empirical import empirical

# Import higher-order distributions
from probjax.stats.independent import independent
from probjax.stats.mixture import mixture
from probjax.stats.transformed import transformed

__all__ = [
    # Base classes
    'rv_generic',
    'rv_continuous',
    'rv_discrete',
    'rv_exponential_family',
    'rv_continuous_frozen',
    'rv_multivariate',
    'rv_multivariate_frozen',
    'rv_spherical',
    'rv_spherical_frozen',
    'rv_discrete_frozen',
    # Continuous distributions
    'norm',
    'gamma',
    'gennorm',
    'genpareto',
    'beta',
    'expon',
    'laplace',
    'logistic',
    'uniform',
    'chi2',
    't',
    'cauchy',
    'dirichlet',
    'multivariate_normal',
    'vonmises',
    'truncnorm',
    'pareto',
    'skewnorm',
    'watson',
    'bingham',
    'wrapcauchy',
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
