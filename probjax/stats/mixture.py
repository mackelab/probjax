"""
Mixture Distribution (:mod:`probjax.stats.mixture`)
=================================================

This module implements mixture distributions that combine multiple component distributions
with mixing probabilities.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import ArrayLike, PRNGKeyArray

from probjax.stats.base import rv_continuous_frozen, rv_discrete_frozen, rv_generic
from probjax.stats.constraints import distribution, simplex

__all__ = ["mixture"]


class mixture_frozen(rv_continuous_frozen, rv_discrete_frozen):
    """Frozen mixture distribution."""

    def __init__(self, dist, mixing_probs, components, **kwds):
        super().__init__(dist, mixing_probs=mixing_probs, components=components, **kwds)

    def _compute_batch_and_event_shape(self, mixing_probs, components, **kwds):
        """Compute the batch and event shape of the distribution."""
        batch_shape1 = mixing_probs.shape[:-1]
        num_components = mixing_probs.shape[-1]
        assert len(components) == num_components, (
            "Number of components must match number of mixing probabilities"
        )
        event_shape = components[0].event_shape
        assert all(comp.event_shape == event_shape for comp in components), (
            "All components must have the same event shape"
        )
        batch_shape2 = components[0].batch_shape
        assert all(comp.batch_shape == batch_shape2 for comp in components), (
            "All components must have the same batch shape"
        )
        batch_shape = jnp.broadcast_shapes(batch_shape1, batch_shape2)
        return batch_shape, event_shape


class mixture_gen(rv_generic):
    """A mixture distribution that combines multiple component distributions."""

    parameters = {
        "mixing_probs": simplex,
        "components": distribution,
    }

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, mixing_probs, components, **kwargs):
        """Create a frozen mixture distribution."""
        return self.freeze(mixing_probs=mixing_probs, components=components, **kwargs)

    def freeze(self, mixing_probs, components, **kwargs):
        """Freeze the mixture distribution with the given parameters."""
        return mixture_frozen(
            self, mixing_probs=mixing_probs, components=components, **kwargs
        )

    @classmethod
    def support(cls, mixing_probs, components, **kwds):
        """Get the support of the mixture distribution."""
        supports = [comp.support() for comp in components]
        if all(isinstance(s, tuple) and len(s) == 2 for s in supports):
            return (min(s[0] for s in supports), max(s[1] for s in supports))
        return tuple(
            set().union(*[s if isinstance(s, tuple) else (s,) for s in supports])
        )

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
        key_sample, key_cluster_membership = random.split(rng, 2)

        # Sample from all components at once
        component_samples = jnp.stack(
            [comp.rvs(key_sample, shape=shape) for comp in components], axis=-1
        )

        # Sample cluster membership
        cluster_membership = random.categorical(
            key_cluster_membership,
            mixing_probs,
            shape=shape,
        )
        while cluster_membership.ndim < component_samples.ndim:
            cluster_membership = jnp.expand_dims(cluster_membership, axis=-1)
        # Select samples based on cluster membership
        samples = jnp.take_along_axis(component_samples, cluster_membership, axis=-1)

        return jnp.squeeze(samples, axis=-1)

    @classmethod
    def mean(cls, mixing_probs, components, **kwds):
        """Mean of the mixture distribution."""
        means = jnp.stack([jnp.asarray(comp.mean()) for comp in components], axis=-1)
        return jnp.sum(mixing_probs * means, axis=-1)

    @classmethod
    def var(cls, mixing_probs, components, **kwds):
        """Variance of the mixture distribution."""
        means = jnp.stack([jnp.asarray(comp.mean()) for comp in components], axis=-1)
        vars = jnp.stack([jnp.asarray(comp.var()) for comp in components], axis=-1)
        mean = jnp.sum(mixing_probs * means, axis=-1)
        return jnp.sum(mixing_probs * (vars + (means - mean[..., None]) ** 2), axis=-1)

    @classmethod
    def mode(cls, mixing_probs, components, **kwds):
        """Mode of the mixture distribution.

        For a mixture distribution with unimodal components, the true mode lies within
        the convex hull of the component modes. We use this fact to constrain our
        optimization search space.
        """
        # Actually not that straightforward to compute the mode of a mixture distribution
        raise NotImplementedError("Mode not implemented for mixture distribution")
        # Get component modes as vertices of the convex hull
        # modes = jnp.stack([comp.mode() for comp in components], axis=0)

        # # Define objective function (negative log probability)
        # def objective(x):
        #     return -cls.logpdf(x, mixing_probs, components).sum()

        # # Use BFGS optimization to find the mode
        # minimize_fn = partial(
        #     minimize, objective, method='BFGS', options={'maxiter': 10}
        # )
        # result = jax.vmap(minimize_fn)(modes)
        # modes = result.x
        # logpdfs = result.fun
        # idxs = jnp.argmax(logpdfs, axis=0)
        # while idxs.ndim < modes.ndim:
        #     idxs = idxs[..., None]
        # mode = jnp.take_along_axis(modes, idxs, axis=0)

        # return jnp.squeeze(mode, axis=-1)

    @classmethod
    def entropy(cls, mixing_probs, components, **kwds):
        """Entropy of the mixture distribution."""
        raise NotImplementedError("Entropy not implemented for mixture distribution")

    @classmethod
    def fit(
        cls,
        x: ArrayLike,
        components,
        max_iter: int = 100,
        tol: float = 1e-4,
        rng_key: Optional[PRNGKeyArray] = None,
    ):
        """Fit the mixture distribution to data using the EM algorithm.

        Args:
            x: Array of observations
            components: List of component distributions to fit
            max_iter: Maximum number of EM iterations
            tol: Convergence tolerance for log-likelihood
            rng_key: Random key for initialization

        Returns:
            Tuple of (mixing_probs, fitted_components)
        """
        x = jnp.asarray(x)
        n_samples = x.shape[0]
        n_components = len(components)

        # Initialize mixing probabilities uniformly
        mixing_probs = jnp.ones(n_components) / n_components

        # Initialize component parameters using their fit methods
        fitted_components = [comp.fit(x) for comp in components]

        # Initialize log-likelihood
        prev_log_likelihood = -jnp.inf

        def em_step(state):
            mixing_probs, fitted_components, prev_log_likelihood = state

            # E-step: Compute responsibilities
            log_pdfs = jnp.stack(
                [comp.logpdf(x) for comp in fitted_components], axis=-1
            )
            log_responsibilities = jnp.log(mixing_probs) + log_pdfs
            responsibilities = jnp.exp(
                log_responsibilities
                - jax.scipy.special.logsumexp(
                    log_responsibilities, axis=-1, keepdims=True
                )
            )

            # M-step: Update mixing probabilities
            mixing_probs = jnp.mean(responsibilities, axis=0)

            # M-step: Update component parameters
            fitted_components = [
                comp.fit(x, weights=responsibilities[:, i])
                for i, comp in enumerate(components)
            ]

            # Compute log-likelihood
            log_likelihood = jnp.mean(
                jax.scipy.special.logsumexp(log_responsibilities, axis=-1)
            )

            return (mixing_probs, fitted_components, log_likelihood)

        def convergence_check(state):
            mixing_probs, fitted_components, log_likelihood = state
            return jnp.abs(log_likelihood - prev_log_likelihood) > tol

        # Run EM algorithm
        state = (mixing_probs, fitted_components, prev_log_likelihood)
        for _ in range(max_iter):
            state = em_step(state)
            if not convergence_check(state):
                break

        mixing_probs, fitted_components, _ = state
        return mixing_probs, fitted_components


mixture = mixture_gen(name="mixture")
