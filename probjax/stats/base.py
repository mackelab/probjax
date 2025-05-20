"""
Statistical Distributions Base Classes (:mod:`probjax.stats.base`)
=================================================================

This module contains the base classes for continuous and discrete random variables
that provide a SciPy-like API. This closely follows the structure of scipy.stats._distn_infrastructure.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union, Callable
import functools
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, ArrayLike
from abc import ABC, abstractmethod

from probjax.stats.constraints import Constraint

__all__ = [
    "rv_generic",
    "rv_continuous",
    "rv_discrete",
    "rv_continuous_frozen",
    "rv_discrete_frozen",
]


class rv_generic(ABC):
    """Generic random variable class for common functionality."""

    name: Optional[str] = None
    parameters: Dict[str, Constraint]

    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __call__(self, *args, **kwds):
        """Call the distribution with the given arguments."""
        return self.freeze(*args, **kwds)

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments."""
        return type('FrozenDistribution', (rv_frozen,), {}, dist=self)(
            self, *args, **kwds
        )

    @classmethod
    @abstractmethod
    def support(cls, *args, **kwds):
        """Support of the distribution."""
        ...

    @classmethod
    @abstractmethod
    def rvs(cls, rng: PRNGKeyArray, shape: Tuple[int, ...] = (), *args, **kwds):
        """Random variates of given shape.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            The random key used for sampling
        shape : tuple of ints
            The shape of the samples to draw
        *args : array_like
            Shape parameters for the distribution
        **kwds : dict, optional
            Additional parameters (loc, scale, etc.)

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        ...

    @classmethod
    def mean(cls, *args, **kwds):
        """Mean of the distribution."""
        raise NotImplementedError("Mean is not implemented for this distribution.")

    @classmethod
    def mode(cls, *args, **kwds):
        """Mode of the distribution."""
        raise NotImplementedError("Mode is not implemented for this distribution.")

    @classmethod
    def var(cls, *args, **kwds):
        """Variance of the distribution."""
        raise NotImplementedError("Variance is not implemented for this distribution.")

    @classmethod
    def std(cls, *args, **kwds):
        """Standard deviation of the distribution."""
        return jnp.sqrt(cls.var(*args, **kwds))

    @classmethod
    def cdf(cls, *args, **kwds):
        """Cumulative distribution function of the RV."""
        raise NotImplementedError("CDF is not implemented for this distribution.")

    @classmethod
    def logcdf(cls, x: ArrayLike, *args, **kwds):
        """Log of the cumulative distribution function at x of the given RV."""
        return jnp.log(cls.cdf(x, *args, **kwds))

    @classmethod
    def sf(cls, x: ArrayLike, *args, **kwds):
        """Survival function (1 - cdf) at x of the given RV."""
        return 1.0 - cls.cdf(x, *args, **kwds)

    @classmethod
    def logsf(cls, x: ArrayLike, *args, **kwds):
        """Log of the survival function at x of the given RV."""
        return jnp.log(cls.sf(x, *args, **kwds))

    @classmethod
    def ppf(cls, *args, **kwds):
        """Percent point function (inverse of cdf) of the RV."""
        raise NotImplementedError("PPF is not implemented for this distribution.")

    @classmethod
    def isf(cls, q: ArrayLike, *args, **kwds):
        """Inverse survival function (1 - ppf) of the RV."""
        return cls.ppf(1 - q, *args, **kwds)

    @classmethod
    def entropy(cls, *args, **kwds):
        """Entropy of the RV."""
        raise NotImplementedError("Entropy is not implemented for this distribution.")

    @classmethod
    def median(cls, *args, **kwds):
        """Median of the distribution."""
        args, kwds = cls._parse_args(*args, **kwds)
        return cls.ppf(0.5, *args, **kwds)

    @classmethod
    def interval(cls, alpha: ArrayLike, *args, **kwds):
        """Confidence interval with equal areas around the median."""
        args, kwds = cls._parse_args(*args, **kwds)
        alpha = jnp.asarray(alpha)
        q1 = (1.0 - alpha) / 2
        q2 = (1.0 + alpha) / 2
        a = cls.ppf(q1, *args, **kwds)
        b = cls.ppf(q2, *args, **kwds)
        return a, b

    @classmethod
    def moment(cls, n: int, *args, **kwds):
        """n-th non-central moment of the distribution.

        Parameters
        ----------
        n : int
            Order of the moment
        *args : array_like
            Shape parameters for the distribution
        **kwds : dict, optional
            Additional parameters (loc, scale, etc.)

        Returns
        -------
        moment : float or ndarray
            n-th non-central moment
        """
        raise NotImplementedError("Moment is not implemented for this distribution.")

    @classmethod
    def skew(cls, *args, **kwds):
        """Skewness of the distribution.

        Parameters
        ----------
        *args : array_like
            Shape parameters for the distribution
        **kwds : dict, optional
            Additional parameters (loc, scale, etc.)

        Returns
        -------
        skew : float or ndarray
            Skewness of the distribution
        """
        raise NotImplementedError("Skewness is not implemented for this distribution.")

    @classmethod
    def kurtosis(cls, *args, **kwds):
        """Kurtosis of the distribution.

        Parameters
        ----------
        *args : array_like
            Shape parameters for the distribution
        **kwds : dict, optional
            Additional parameters (loc, scale, etc.)

        Returns
        -------
        kurtosis : float or ndarray
            Kurtosis of the distribution (Fisher's definition, kurtosis - 3)
        """
        raise NotImplementedError("Kurtosis is not implemented for this distribution.")


class rv_exponential_family(rv_generic):
    """Base class for exponential family random variables."""

    @classmethod
    def natural_parameters(cls, *args, **kwds):
        """Natural parameters of the distribution."""
        raise NotImplementedError(
            "Natural parameters are not implemented for this distribution."
        )

    @classmethod
    def sufficient_statistics(cls, x: ArrayLike, *args, **kwds):
        """Sufficient statistics of the distribution."""
        raise NotImplementedError(
            "Sufficient statistics are not implemented for this distribution."
        )

    @classmethod
    def log_partition(cls, *args, **kwds):
        """Log partition function of the distribution."""
        raise NotImplementedError(
            "Log partition function is not implemented for this distribution."
        )


class rv_continuous(rv_generic):
    """Base class for continuous random variables."""

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments."""
        return type(
            'FrozenContinuousDistribution', (rv_continuous_frozen,), {}, dist=self
        )(self, *args, **kwds)

    @classmethod
    def pdf(cls, x: ArrayLike, *args, **kwds):
        """Probability density function at x of the given RV."""
        args, kwds = cls._parse_args(*args, **kwds)
        return jnp.exp(cls.logpdf(x, *args, **kwds))

    @classmethod
    def logpdf(cls, x: ArrayLike, *args, **kwds):
        """Log of the probability density function at x of the given RV."""
        raise NotImplementedError("Logpdf is not implemented for this distribution.")


class rv_discrete(rv_generic):
    """Base class for discrete random variables."""

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments."""
        return type('FrozenDiscreteDistribution', (rv_discrete_frozen,), {}, dist=self)(
            self, *args, **kwds
        )

    @classmethod
    @abstractmethod
    def pmf(cls, k: ArrayLike, *args, **kwds):
        """Probability mass function at k of the given RV."""
        ...

    @classmethod
    def logpmf(cls, k: ArrayLike, *args, **kwds):
        """Log of the probability mass function at k of the given RV."""
        return jnp.log(cls.pmf(k, *args, **kwds))

    @classmethod
    def sf(cls, k: ArrayLike, *args, **kwds):
        """Survival function (1 - cdf) at k of the given RV."""
        return 1.0 - cls.cdf(k, *args, **kwds)

    @classmethod
    def logsf(cls, k: ArrayLike, *args, **kwds):
        """Log of the survival function at k of the given RV."""
        return jnp.log(cls.sf(k, *args, **kwds))


class FrozenDistributionMeta(type):
    """Metaclass for frozen distributions that inherits name and docstrings."""

    def __new__(mcs, name, bases, namespace, dist=None):
        if dist is not None:
            # Inherit name if available
            if hasattr(dist, 'name'):
                namespace['name'] = dist.name

            # Inherit docstrings from the distribution
            if dist.__doc__:
                namespace['__doc__'] = dist.__doc__

            # Inherit method docstrings
            for attr_name in dir(dist):
                if not attr_name.startswith('_'):
                    dist_attr = getattr(dist, attr_name)
                    if callable(dist_attr) and hasattr(dist_attr, '__doc__'):
                        if attr_name in namespace:
                            namespace[attr_name].__doc__ = dist_attr.__doc__

        return super().__new__(mcs, name, bases, namespace)


class rv_frozen(metaclass=FrozenDistributionMeta):
    def __init__(
        self,
        dist,
        *args,
        **kwds,
    ):
        self.args = args
        self.kwds = kwds
        self.dist = dist

        # Get shapes from the distribution arguments
        batch_shapes = []

        # Check args for shapes
        for arg in args:
            if isinstance(arg, (jnp.ndarray, np.ndarray)):
                batch_shapes.append(arg.shape)

        # Check kwargs for shapes
        for v in kwds.values():
            if isinstance(v, (jnp.ndarray, np.ndarray)):
                batch_shapes.append(v.shape)

        # Compute final shapes
        if len(batch_shapes) > 0:
            self._batch_shape = jnp.broadcast_shapes(*batch_shapes)
        else:
            self._batch_shape = ()

        self._event_shape = ()

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return self._batch_shape

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return self._event_shape

    def _format_arg(self, arg):
        """Format argument for string representation, showing shapes for arrays."""
        # Check if arg is a JAX array
        if isinstance(arg, (jnp.ndarray, np.ndarray)):
            return f"array(shape={arg.shape}, dtype={arg.dtype})"
        # For other types, just use normal string representation
        return str(arg)

    def __str__(self):
        """String representation with array shapes instead of content."""
        args_str = ', '.join(self._format_arg(arg) for arg in self.args)
        kwargs_str = ', '.join(
            f'{k}={self._format_arg(v)}' for k, v in self.kwds.items()
        )

        if args_str and kwargs_str:
            params_str = f"{args_str}, {kwargs_str}"
        elif args_str:
            params_str = args_str
        elif kwargs_str:
            params_str = kwargs_str
        else:
            params_str = ""

        return f"{self.name or self.dist.__class__.__name__}({params_str})"

    def __repr__(self):
        """Representation for debugging."""
        return self.__str__()

    def tree_flatten(self):
        """Return a flattened representation for JAX pytree."""
        # Flatten the distribution
        children = (self.args, self.kwds)
        # Auxiliary data that can be useful for reconstruction
        aux_data = {'dist': self.dist}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct an instance from flattened representation."""
        args, kwds = children
        dist = aux_data['dist']
        # Create and return a new frozen instance
        instance = cls(dist, *args, **kwds)
        return instance

    def cdf(self, x: ArrayLike):
        """Cumulative distribution function of the RV."""
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x: ArrayLike):
        """Log of the cumulative distribution function of the RV."""
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function (inverse of cdf) of the RV."""
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q: ArrayLike):
        """Inverse survival function of the RV."""
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of given shape."""
        return self.dist.rvs(rng, shape, *self.args, **self.kwds)

    def sf(self, x: ArrayLike):
        """Survival function (1 - cdf) at x of the given RV."""
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x: ArrayLike):
        """Log of the survival function at x of the given RV."""
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments: str = 'mv'):
        """Returns mean, variance, skew, or kurtosis."""
        kwds = self.kwds.copy()
        kwds.update({'moments': moments})
        return self.dist.stats(*self.args, **kwds)

    def median(self):
        """Median of the distribution."""
        return self.dist.median(*self.args, **self.kwds)

    def mean(self):
        """Mean of the distribution."""
        return self.dist.mean(*self.args, **self.kwds)

    def var(self):
        """Variance of the distribution."""
        return self.dist.var(*self.args, **self.kwds)

    def std(self):
        """Standard deviation of the distribution."""
        return self.dist.std(*self.args, **self.kwds)

    def moment(self, order: Optional[int] = None):
        """Non-central moment of the distribution."""
        return self.dist.moment(order, *self.args, **self.kwds)

    def entropy(self):
        """Entropy of the RV."""
        return self.dist.entropy(*self.args, **self.kwds)

    def interval(self, confidence: Optional[ArrayLike] = None):
        """Confidence interval with equal areas around the median."""
        return self.dist.interval(confidence, *self.args, **self.kwds)

    def support(self):
        """Support of the distribution."""
        return self.dist.support(*self.args, **self.kwds)


class rv_discrete_frozen(rv_frozen):
    def pmf(self, k: ArrayLike):
        """Probability mass function at k of the given RV."""
        return self.dist.pmf(k, *self.args, **self.kwds)

    def logpmf(self, k: ArrayLike):
        """Log of the probability mass function at k of the given RV."""
        return self.dist.logpmf(k, *self.args, **self.kwds)


class rv_continuous_frozen(rv_frozen):
    def pdf(self, x: ArrayLike):
        """Probability density function at x of the given RV."""
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x: ArrayLike):
        """Log of the probability density function at x of the given RV."""
        return self.dist.logpdf(x, *self.args, **self.kwds)

    def cdf(self, x: ArrayLike):
        """Cumulative distribution function of the RV."""
        return self.dist.cdf(x, *self.args, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function (inverse of cdf) of the RV."""
        return self.dist.ppf(q, *self.args, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = ()):
        """Random variates of given shape."""
        return self.dist.rvs(rng, shape, *self.args, **self.kwds)

    def mean(self):
        """Mean of the distribution."""
        return self.dist.mean(*self.args, **self.kwds)

    def var(self):
        """Variance of the distribution."""
        return self.dist.var(*self.args, **self.kwds)

    def entropy(self):
        """Entropy of the distribution."""
        return self.dist.entropy(*self.args, **self.kwds)

    def mode(self):
        """Mode of the distribution."""
        return self.dist.mode(*self.args, **self.kwds)

    def support(self):
        """Support of the distribution."""
        return self.dist.support(*self.args, **self.kwds)

    @property
    def batch_shape(self):
        """Batch shape of the distribution."""
        return self.dist._get_batch_shape(*self.args, **self.kwds)

    @property
    def event_shape(self):
        """Event shape of the distribution."""
        return self.dist._get_event_shape(*self.args, **self.kwds)


# Register frozen classes as JAX PyTrees
jax.tree_util.register_pytree_node_class(rv_frozen)
jax.tree_util.register_pytree_node_class(rv_continuous_frozen)
jax.tree_util.register_pytree_node_class(rv_discrete_frozen)
