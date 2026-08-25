# Distributions

SciPy-shaped distributions. Each generator is used directly
(`norm.rvs(key, loc, scale)`) or frozen with fixed parameters
(`norm(loc=..., scale=...)`).

## Base classes

::: probjax.stats.rv_generic
::: probjax.stats.rv_continuous
::: probjax.stats.rv_discrete
::: probjax.stats.rv_multivariate
::: probjax.stats.rv_frozen
::: probjax.stats.rv_exponential_family
::: probjax.stats.rv_spherical

## Continuous

::: probjax.stats.norm
::: probjax.stats.gamma
::: probjax.stats.beta
::: probjax.stats.expon
::: probjax.stats.laplace
::: probjax.stats.logistic
::: probjax.stats.uniform
::: probjax.stats.cauchy
::: probjax.stats.chi2
::: probjax.stats.t
::: probjax.stats.pareto
::: probjax.stats.genpareto
::: probjax.stats.gennorm
::: probjax.stats.skewnorm
::: probjax.stats.truncnorm

## Flexible univariate families

Parameterised densities intended as conditional heads for autoregressive models,
or as flexible marginals in their own right.

::: probjax.stats.mixture_kernel
::: probjax.stats.logistic_mixture_kernel
::: probjax.stats.histogram
::: probjax.stats.tailed_histogram
::: probjax.stats.spline_normal

## Multivariate and directional

::: probjax.stats.multivariate_normal
::: probjax.stats.dirichlet
::: probjax.stats.vonmises
::: probjax.stats.watson
::: probjax.stats.bingham
::: probjax.stats.wrapcauchy

## Discrete

::: probjax.stats.bernoulli
::: probjax.stats.binomial
::: probjax.stats.categorical
::: probjax.stats.poisson
::: probjax.stats.geometric
::: probjax.stats.dirac
::: probjax.stats.empirical

## Higher-order

::: probjax.stats.transformed
::: probjax.stats.mixture
::: probjax.stats.indep

## Fitting

::: probjax.stats.fit
::: probjax.stats.FitMixin
::: probjax.stats.is_batch_stream
::: probjax.stats.take_batches

## Transform protocols

Used to build higher-order distributions and normalizing flows.

::: probjax.stats.TransformedDistribution
::: probjax.stats.forward_and_logdet
::: probjax.stats.ensure_invertible
