"""
Mixture Distribution (:mod:`probjax.stats.mixture`)
=================================================

This module implements mixture distributions that combine multiple component distributions
with mixing probabilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, ArrayLike

from probjax.stats.base import rv_continuous, rv_continuous_frozen
from probjax.stats.constraints import simplex, distribution

__all__ = ["mixture", "mixture_frozen"]


class mixture(rv_continuous):
    """A mixture distribution that combines multiple component distributions."""

    parameters = {
        "mixing_probs": simplex,
        "components": distribution,
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def _parse_args(cls, mixing_probs, components, **kwds):
        """Parse arguments for the mixture distribution."""
        return (mixing_probs, components), kwds

    @classmethod
    def _get_support(cls, mixing_probs, components, **kwds):
        """Get the support of the mixture distribution."""
        # The support of a mixture is the union of the supports of its components.
        # Since we can't easily compute this in general, we return the full real line
        # as a conservative estimate.
        return (-jnp.inf, jnp.inf)

    @classmethod
    def _get_batch_shape(cls, mixing_probs, components, **kwds):
        """Get the batch shape of the mixture distribution."""
        return mixing_probs.shape[:-1]

    @classmethod
    def _get_event_shape(cls, mixing_probs, components, **kwds):
        """Get the event shape of the mixture distribution."""
        return components[0].event_shape

    @classmethod
    def pdf(cls, x: ArrayLike, mixing_probs, components, **kwds):
        """Probability density function of the mixture distribution."""
        x = jnp.asarray(x)
        pdfs = jnp.stack([comp.pdf(x) for comp in components], axis=-1)
        return jnp.sum(mixing_probs * pdfs, axis=-1)

    @classmethod
    def logpdf(cls, x: ArrayLike, mixing_probs, components, **kwds):
        """Log probability density function of the mixture distribution."""
        x = jnp.asarray(x)
        log_pdfs = jnp.stack([comp.logpdf(x) for comp in components], axis=-1)
        return jax.scipy.special.logsumexp(jnp.log(mixing_probs) + log_pdfs, axis=-1)

    @classmethod
    def cdf(cls, x: ArrayLike, mixing_probs, components, **kwds):
        """Cumulative distribution function of the mixture distribution."""
        x = jnp.asarray(x)
        cdfs = jnp.stack([comp.cdf(x) for comp in components], axis=-1)
        return jnp.sum(mixing_probs * cdfs, axis=-1)

    @classmethod
    def ppf(cls, q: ArrayLike, mixing_probs, components, **kwds):
        """Percent point function of the mixture distribution."""
        # For mixtures, the PPF is not easily computable in general.
        # We use a numerical approximation by finding the root of CDF(x) - q = 0.
        q = jnp.asarray(q)
        x0 = jnp.mean([comp.ppf(q) for comp in components], axis=0)
        return jax.scipy.optimize.root(
            lambda x: cls.cdf(x, mixing_probs, components) - q,
            x0,
        ).x

    @classmethod
    def rvs(
        cls,
        rng: PRNGKeyArray,
        shape: Tuple[int, ...] = (),
        mixing_probs=None,
        components=None,
        **kwds,
    ):
        """Random variates of the mixture distribution."""
        # First sample the component index
        component_idx = random.categorical(rng, mixing_probs, shape=shape)
        # Then sample from the selected component
        samples = []
        for i, comp in enumerate(components):
            mask = component_idx == i
            if jnp.any(mask):
                comp_samples = comp.rvs(rng, shape=mask.shape)
                samples.append(jnp.where(mask[..., None], comp_samples, 0))
        return jnp.sum(jnp.stack(samples), axis=0)

    @classmethod
    def mean(cls, mixing_probs, components, **kwds):
        """Mean of the mixture distribution."""
        means = jnp.stack([comp.mean() for comp in components], axis=-1)
        return jnp.sum(mixing_probs * means, axis=-1)

    @classmethod
    def var(cls, mixing_probs, components, **kwds):
        """Variance of the mixture distribution."""
        means = jnp.stack([comp.mean() for comp in components], axis=-1)
        vars = jnp.stack([comp.var() for comp in components], axis=-1)
        mean = jnp.sum(mixing_probs * means, axis=-1)
        return jnp.sum(mixing_probs * (vars + (means - mean[..., None]) ** 2), axis=-1)

    def freeze(self, mixing_probs, components, **kwds):
        """Freeze the mixture distribution with the given parameters."""
        return mixture_frozen(self, mixing_probs, components, **kwds)


class mixture_frozen(rv_continuous_frozen):
    """Frozen mixture distribution."""

    def __init__(self, dist, mixing_probs, components, **kwds):
        super().__init__(dist, mixing_probs=mixing_probs, components=components, **kwds)
        self.mixing_probs = mixing_probs
        self.components = components

    def pdf(self, x: ArrayLike):
        """Probability density function of the frozen mixture distribution."""
        return self.dist.pdf(x, self.mixing_probs, self.components, **self.kwds)

    def logpdf(self, x: ArrayLike):
        """Log probability density function of the frozen mixture distribution."""
        return self.dist.logpdf(x, self.mixing_probs, self.components, **self.kwds)

    def cdf(self, x: ArrayLike):
        """Cumulative distribution function of the frozen mixture distribution."""
        return self.dist.cdf(x, self.mixing_probs, self.components, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function of the frozen mixture distribution."""
        return self.dist.ppf(q, self.mixing_probs, self.components, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of the frozen mixture distribution."""
        return self.dist.rvs(
            rng, shape, self.mixing_probs, self.components, **self.kwds
        )

    def mean(self):
        """Mean of the frozen mixture distribution."""
        return self.dist.mean(self.mixing_probs, self.components, **self.kwds)

    def var(self):
        """Variance of the frozen mixture distribution."""
        return self.dist.var(self.mixing_probs, self.components, **self.kwds)
