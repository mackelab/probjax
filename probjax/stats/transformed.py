"""
Transformed Distribution (:mod:`probjax.stats.transformed`)
=========================================================

This module implements transformed distributions that apply a bijective transformation
to a base distribution.
"""

import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from probjax.core import inverse_and_logabsdet
from probjax.stats.base import rv_continuous, rv_continuous_frozen
from probjax.stats.constraints import distribution, real
from probjax.utils.typing import ArrayLike, RngKey

__all__ = ["transformed"]


class transformed_frozen(rv_continuous_frozen):
    """Frozen transformed distribution with base-shape metadata."""

    def __init__(self, dist, base_dist, bijector, **kwds):
        super().__init__(dist, base_dist=base_dist, bijector=bijector, **kwds)

    def _compute_batch_and_event_shape(self, base_dist, bijector, **kwds):
        del bijector, kwds
        batch_shape = tuple(int(dim) for dim in base_dist.batch_shape)
        event_shape = tuple(int(dim) for dim in base_dist.event_shape)
        return batch_shape, event_shape


class transformed_gen(rv_continuous):
    """A transformed distribution that applies a bijective transformation to a base distribution."""

    parameters = {
        "base_dist": distribution,
        "bijector": callable,  # type: ignore[dict-item]
    }
    extra_frozen_kwds = frozenset({"inverse_and_logdet"})

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @classmethod
    def _parse_args(cls, base_dist, bijector, **kwds):
        """Parse arguments for the transformed distribution."""
        return (base_dist, bijector), kwds

    def freeze(self, base_dist, bijector, **kwargs):
        """Freeze the transformed distribution with the given parameters."""
        return transformed_frozen(
            self, base_dist=base_dist, bijector=bijector, **kwargs
        )

    @classmethod
    def support(cls, base_dist, bijector, **kwds):
        """Support of the transformed distribution."""
        return real

    @classmethod
    @functools.cache
    def _get_vmapped_bijector(cls, bijector):
        """Get a single-axis vmapped bijector."""
        return jax.vmap(bijector)

    @classmethod
    @functools.cache
    def _get_inverse_and_logdet(cls, bijector):
        """Build inverse+logabsdet function for a bijector."""
        return inverse_and_logabsdet(bijector)

    @classmethod
    @functools.cache
    def _get_vmapped_inverse_and_logdet(cls, bijector):
        """Get a single-axis vmapped inverse+logabsdet function."""
        return jax.vmap(cls._get_inverse_and_logdet(bijector))

    @classmethod
    def _get_vmapped_inverse_and_logdet_with_override(
        cls, bijector, inverse_and_logdet_fn=None
    ):
        """Get vmapped inverse+logabsdet, optionally using a custom function."""
        del bijector
        if inverse_and_logdet_fn is None:
            return None
        return jax.vmap(inverse_and_logdet_fn)

    @staticmethod
    def _flatten_by_event_shape(x: ArrayLike, event_shape: Tuple[int, ...]):
        """Flatten all leading dimensions into one axis while preserving event dims."""
        x_arr = jnp.asarray(x)
        event_shape = tuple(event_shape)

        if event_shape:
            event_ndim = len(event_shape)
            if x_arr.ndim < event_ndim:
                raise ValueError(
                    "Input has fewer dimensions than the distribution event shape."
                )
            trailing_shape = tuple(x_arr.shape[-event_ndim:])
            if trailing_shape != event_shape:
                raise ValueError(
                    "Trailing dimensions of the input must match the distribution event shape."
                )
            leading_shape = tuple(x_arr.shape[:-event_ndim])
            x_flat = jnp.reshape(x_arr, (-1,) + event_shape)
        else:
            leading_shape = tuple(x_arr.shape)
            x_flat = jnp.reshape(x_arr, (-1,))

        return x_arr, x_flat, leading_shape

    @staticmethod
    def _unflatten_by_event_shape(x_flat, leading_shape: Tuple[int, ...], event_shape):
        """Restore flattened values back to leading and event dimensions."""
        event_shape = tuple(event_shape)
        if event_shape:
            return jnp.reshape(x_flat, leading_shape + event_shape)
        return jnp.reshape(x_flat, leading_shape)

    @staticmethod
    def _ensure_univariate_event(event_shape: Tuple[int, ...]):
        """Restrict operations that only support univariate events."""
        if event_shape not in ((), (1,)):
            raise NotImplementedError(
                "This method currently supports only univariate transformed distributions."
            )

    @classmethod
    def logpdf(cls, x: ArrayLike, base_dist, bijector, inverse_and_logdet=None, **kwds):
        """Log probability density function of the transformed distribution."""
        event_shape = tuple(base_dist.event_shape)
        _, x_flat, leading_shape = cls._flatten_by_event_shape(x, event_shape)

        vmapped_inverse_and_logdet = cls._get_vmapped_inverse_and_logdet_with_override(
            bijector, inverse_and_logdet
        )
        if vmapped_inverse_and_logdet is None:
            vmapped_inverse_and_logdet = cls._get_vmapped_inverse_and_logdet(bijector)

        inv_flat, log_det_flat = vmapped_inverse_and_logdet(x_flat)
        inv_value = cls._unflatten_by_event_shape(inv_flat, leading_shape, event_shape)
        log_det = jnp.reshape(
            log_det_flat, leading_shape + tuple(log_det_flat.shape[1:])
        )
        return base_dist.logpdf(inv_value) + log_det

    @classmethod
    def pdf(cls, x: ArrayLike, base_dist, bijector, **kwds):
        """Probability density function of the transformed distribution."""
        return jnp.exp(cls.logpdf(x, base_dist, bijector, **kwds))

    @classmethod
    def cdf(cls, x: ArrayLike, base_dist, bijector, inverse_and_logdet=None, **kwds):
        """Cumulative distribution function of the transformed distribution."""
        event_shape = tuple(base_dist.event_shape)
        cls._ensure_univariate_event(event_shape)
        _, x_flat, leading_shape = cls._flatten_by_event_shape(x, event_shape)

        vmapped_inverse_and_logdet = cls._get_vmapped_inverse_and_logdet_with_override(
            bijector, inverse_and_logdet
        )
        if vmapped_inverse_and_logdet is None:
            vmapped_inverse_and_logdet = cls._get_vmapped_inverse_and_logdet(bijector)

        inv_flat, _ = vmapped_inverse_and_logdet(x_flat)
        inv_value = cls._unflatten_by_event_shape(inv_flat, leading_shape, event_shape)
        return base_dist.cdf(inv_value)

    @classmethod
    def ppf(cls, q: ArrayLike, base_dist, bijector, inverse_and_logdet=None, **kwds):
        """Percent point function of the transformed distribution."""
        del inverse_and_logdet, kwds
        event_shape = tuple(base_dist.event_shape)
        cls._ensure_univariate_event(event_shape)

        base_ppf = base_dist.ppf(q)
        _, base_ppf_flat, leading_shape = cls._flatten_by_event_shape(
            base_ppf, event_shape
        )
        vmapped_bijector = cls._get_vmapped_bijector(bijector)
        transformed_flat = vmapped_bijector(base_ppf_flat)
        return cls._unflatten_by_event_shape(
            transformed_flat, leading_shape, event_shape
        )

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        base_dist=None,
        bijector=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ):
        """Random variates of the transformed distribution."""
        if base_dist is None or bijector is None:
            raise ValueError("Both base_dist and bijector must be provided.")
        samples = base_dist.rvs(rng, shape=shape)
        event_shape = tuple(base_dist.event_shape)
        _, samples_flat, leading_shape = cls._flatten_by_event_shape(
            samples, event_shape
        )
        vmapped_bijector = cls._get_vmapped_bijector(bijector)
        transformed_flat = vmapped_bijector(samples_flat)
        return cls._unflatten_by_event_shape(
            transformed_flat, leading_shape, event_shape
        )

    @classmethod
    def mean(cls, base_dist, bijector, **kwds):
        """Mean of the transformed distribution."""
        raise NotImplementedError("Mean not implemented for transformed distribution")

    @classmethod
    def var(cls, base_dist, bijector, **kwds):
        """Variance of the transformed distribution."""
        raise NotImplementedError(
            "Variance not implemented for transformed distribution"
        )

    @classmethod
    def entropy(cls, base_dist, bijector, inverse_and_logdet=None, **kwds):
        """Entropy of the transformed distribution."""
        raise NotImplementedError(
            "Entropy not implemented for transformed distribution"
        )

    @classmethod
    def mode(cls, base_dist, bijector, **kwds):
        """Mode of the transformed distribution."""
        raise NotImplementedError("Mode not implemented for transformed distribution")


transformed = transformed_gen(name="transformed")
