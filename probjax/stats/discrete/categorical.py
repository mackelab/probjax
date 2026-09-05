"""
Categorical Distribution (:mod:`probjax.stats.categorical`)
========================================================

This module implements the Categorical distribution.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_discrete, rv_discrete_frozen, rv_exponential_family
from probjax.stats.constraints import simplex
from probjax.stats.utils import flatten_samples, normalize_sample_weights
from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["categorical"]


class categorical_gen(rv_discrete, rv_exponential_family):
    """A Categorical distribution."""

    parameters = {
        "probs": simplex,
    }

    @classmethod
    def support(cls, probs, **kwds):
        """Support of the Categorical distribution."""
        return (0, probs.shape[-1] - 1)

    @classmethod
    def pmf(cls, k: ArrayLike, probs, **kwds):
        """Probability mass function of the Categorical distribution."""
        return jnp.exp(cls.logpmf(k, probs, **kwds))

    @classmethod
    def logpmf(cls, k: ArrayLike, probs, **kwds):
        """Log probability mass function of the Categorical distribution."""
        k = jnp.asarray(k).astype(jnp.int32)
        k = jax.nn.one_hot(k, probs.shape[-1])
        log_probs = jax.scipy.special.xlogy(k, probs).sum(axis=-1)
        return log_probs

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        probs=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the Categorical distribution."""
        probs = jnp.asarray(probs)
        event_shape = probs.shape[
            :-1
        ]  # Remove the last dimension which is the number of categories
        # random.categorical expects *logits*; passing probs straight in would
        # silently sample from softmax(probs) instead of probs.
        logits = jnp.log(jnp.clip(probs, jnp.finfo(probs.dtype).tiny))
        return random.categorical(rng, logits, shape=shape + event_shape, axis=-1)

    @classmethod
    def mean(cls, probs, **kwds):
        """Mean of the Categorical distribution."""
        return jnp.sum(probs * jnp.arange(probs.shape[-1]), axis=-1)

    @classmethod
    def var(cls, probs, **kwds):
        """Variance of the Categorical distribution."""
        mean = cls.mean(probs, **kwds)
        return jnp.sum(probs * (jnp.arange(probs.shape[-1]) - mean) ** 2, axis=-1)

    @classmethod
    def entropy(cls, probs, **kwds):
        """Entropy of the Categorical distribution."""
        return -jnp.sum(probs * jnp.log(probs), axis=-1)

    @classmethod
    def natural_parameters(cls, probs, **kwds):
        """Natural parameters of the Categorical distribution."""
        return jnp.log(probs)

    @classmethod
    def sufficient_statistics(cls, x, probs, **kwds):
        """Sufficient statistics of the Categorical distribution."""
        return jax.nn.one_hot(x, probs.shape[-1])

    @classmethod
    def log_partition(cls, probs, **kwds):
        """Log partition function of the Categorical distribution."""
        return 0.0

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        num_classes: Optional[int] = None,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Maximum likelihood estimation of Categorical distribution parameters.

        The MLE for the Categorical distribution has a closed-form solution:
        probs = empirical probabilities of each category

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        num_classes : int, optional
            Number of classes. If None, inferred from data.
        **kwds : dict, optional
            Additional parameters (ignored)

        Returns
        -------
        params : tuple
            The fitted probabilities
        """
        data = flatten_samples(data)
        if num_classes is None:
            num_classes = int(jnp.max(data)) + 1
        count_dtype = jnp.result_type(data.dtype, jnp.float32)

        weights_arr = normalize_sample_weights(
            weights,
            n_samples=data.shape[0],
            dtype=count_dtype,
        )
        if weights_arr is not None:
            counts = jnp.zeros((num_classes,), dtype=count_dtype)
            counts = counts.at[data].add(weights_arr)
        else:
            counts = jnp.bincount(data, length=num_classes).astype(count_dtype)
            counts = counts / jnp.sum(counts)
        probs = jnp.clip(counts, jnp.asarray(1e-12, dtype=count_dtype), None)
        probs = probs / jnp.sum(probs)
        return (probs,)

    @classmethod
    def freeze(cls, probs, **kwds):
        """Freeze the Categorical distribution."""
        rv = rv_discrete_frozen(cls, probs, **kwds)
        rv._batch_shape = probs.shape[:-1]
        rv._event_shape = ()
        return rv


categorical = categorical_gen(name="categorical")
