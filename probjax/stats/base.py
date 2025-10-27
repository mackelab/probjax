"""
Statistical Distributions Base Classes (:mod:`probjax.stats.base`)
=================================================================

This module contains the base classes for continuous and discrete random variables
that provide a SciPy-like API. This closely follows the structure of scipy.stats._distn_infrastructure.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, PRNGKeyArray

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

    def _parse_args(cls, *args, **kwds):
        """Parse arguments for the distribution."""
        # Move important kwds to args
        for param_name in cls.parameters:
            if param_name in kwds:
                args = args + (kwds.pop(param_name),)
        return args, kwds

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments."""
        # Create the frozen class
        frozen_cls = type('FrozenDistribution', (rv_frozen,), {'dist': self})
        return frozen_cls(self, *args, **kwds)

    @classmethod
    @abstractmethod
    def support(cls, *args, **kwds):
        """Support of the distribution."""
        ...

    @classmethod
    @abstractmethod
    def rvs(cls, rng: PRNGKeyArray, *args, shape: Tuple[int, ...] = (), **kwargs):
        """Random variates of given shape.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            The random key used for sampling
        *args : array_like
            Shape parameters for the distribution
        shape : tuple of ints, optional
            The shape of the samples to draw
        **kwargs : dict, optional
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

    @classmethod
    def fit(cls, data: ArrayLike, **kwds):
        """Maximum likelihood estimation of distribution parameters.

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters for the optimization

        Returns
        -------
        params : tuple
            The fitted parameters of the distribution
        """
        raise NotImplementedError("Not implemented for this distribution.")


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

    @classmethod
    def fit(cls, data: ArrayLike, **kwds):
        """Maximum likelihood estimation of distribution parameters using sufficient
        statistics.

        For exponential family distributions, the MLE can be computed efficiently using
        sufficient statistics. The natural parameters are found by solving the equation:

            E[T(X)] = T(x)

        where T(X) is the sufficient statistic and T(x) is the observed sufficient
        statistic.

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters for the optimization

        Returns
        -------
        params : tuple
            The fitted parameters of the distribution
        """
        data = jnp.asarray(data)

        # Compute sufficient statistics
        T = cls.sufficient_statistics(data)

        # Get initial parameters from kwds or use defaults
        init_params = {}
        for param_name, constraint in cls.parameters.items():
            if param_name in kwds:
                init_params[param_name] = kwds.pop(param_name)
            else:
                # Use default value from constraint
                init_params[param_name] = constraint.default_value

        # Convert to flat array for optimization
        init_flat = jnp.concatenate([jnp.ravel(v) for v in init_params.values()])

        def neg_log_likelihood(params_flat):
            # Reshape parameters according to their original shapes
            start_idx = 0
            params = {}
            for param_name, param_value in init_params.items():
                param_size = jnp.size(param_value)
                param_shape = jnp.shape(param_value)
                param = params_flat[start_idx : start_idx + param_size].reshape(
                    param_shape
                )
                params[param_name] = param
                start_idx += param_size

            # Get natural parameters
            eta = cls.natural_parameters(**params)

            # Compute expected sufficient statistics
            E_T = jax.grad(cls.log_partition)(eta)

            # Compute negative log likelihood using sufficient statistics
            return -jnp.sum(eta * T - cls.log_partition(eta))

        # Optimize
        from jax.scipy.optimize import minimize

        result = minimize(neg_log_likelihood, init_flat, method="BFGS", **kwds)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Reshape parameters back to their original shapes
        start_idx = 0
        fitted_params = {}
        for param_name, param_value in init_params.items():
            param_size = jnp.size(param_value)
            param_shape = jnp.shape(param_value)
            param = result.x[start_idx : start_idx + param_size].reshape(param_shape)
            fitted_params[param_name] = param
            start_idx += param_size

        return tuple(fitted_params.values())


class rv_continuous(rv_generic):
    """Base class for continuous random variables."""

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments."""
        # Create the frozen class
        frozen_cls = type(
            'FrozenContinuousDistribution', (rv_continuous_frozen,), {'dist': self}
        )
        return frozen_cls(self, *args, **kwds)

    @classmethod
    def pdf(cls, x: ArrayLike, *args, **kwds):
        """Probability density function at x of the given RV."""
        args, kwds = cls._parse_args(*args, **kwds)
        return jnp.exp(cls.logpdf(x, *args, **kwds))

    @classmethod
    def logpdf(cls, x: ArrayLike, *args, **kwds):
        """Log of the probability density function at x of the given RV."""
        raise NotImplementedError("Logpdf is not implemented for this distribution.")

    @classmethod
    def fit(cls, data: ArrayLike, **kwds):
        """Maximum likelihood estimation of distribution parameters.

        For discrete distributions, the MLE is found by maximizing the log-likelihood
        using the logpmf function.

        Parameters
        ----------
        data : array_like
            Data to fit the distribution to
        **kwds : dict, optional
            Additional parameters for the optimization

        Returns
        -------
        params : tuple
            The fitted parameters of the distribution
        """
        data = jnp.asarray(data)

        # Get initial parameters from kwds or use defaults
        init_params = {}
        for param_name, constraint in cls.parameters.items():
            if param_name in kwds:
                init_params[param_name] = kwds.pop(param_name)
            else:
                # Use default value from constraint
                init_params[param_name] = constraint.default_value

        # Convert to flat array for optimization
        init_flat = jnp.concatenate([jnp.ravel(v) for v in init_params.values()])

        def neg_log_likelihood(params_flat):
            # Reshape parameters according to their original shapes
            start_idx = 0
            params = {}
            for param_name, param_value in init_params.items():
                param_size = jnp.size(param_value)
                param_shape = jnp.shape(param_value)
                param = params_flat[start_idx : start_idx + param_size].reshape(
                    param_shape
                )
                params[param_name] = param
                start_idx += param_size

            # Compute negative log likelihood using logpmf
            return -jnp.sum(cls.logpdf(data, **params))

        # Optimize
        from jax.scipy.optimize import minimize

        result = minimize(neg_log_likelihood, init_flat, method="BFGS", **kwds)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Reshape parameters back to their original shapes
        start_idx = 0
        fitted_params = {}
        for param_name, param_value in init_params.items():
            param_size = jnp.size(param_value)
            param_shape = jnp.shape(param_value)
            param = result.x[start_idx : start_idx + param_size].reshape(param_shape)
            fitted_params[param_name] = param
            start_idx += param_size

        return tuple(fitted_params.values())


class rv_discrete(rv_generic):
    """Base class for discrete random variables."""

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments."""
        # Create the frozen class
        frozen_cls = type(
            'FrozenDiscreteDistribution', (rv_discrete_frozen,), {'dist': self}
        )
        return frozen_cls(self, *args, **kwds)

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
    def logpdf(cls, x: ArrayLike, *args, **kwds):
        """Log of the probability density function at x of the given RV."""
        return cls.logpmf(x, *args, **kwds)

    @classmethod
    def pdf(cls, x: ArrayLike, *args, **kwds):
        """Probability density function at x of the given RV."""
        return jnp.exp(cls.logpdf(x, *args, **kwds))


class FrozenDistributionMeta(type):
    """Metaclass for frozen distributions that inherits name and docstrings."""

    def __new__(mcs, name, bases, namespace, dist=None):
        if dist is not None:
            # Get the concrete implementation class
            impl_class = dist.__class__

            # Inherit name if available
            if hasattr(impl_class, 'name'):
                namespace['name'] = impl_class.name

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

        self._batch_shape, self._event_shape = self._compute_batch_and_event_shape(
            *args, **kwds
        )

        super().__init__()

    def _compute_batch_and_event_shape(self, *args, **kwds):
        """Compute the batch and event shape of the distribution."""
        # Get shapes from the distribution arguments
        batch_shapes = []

        num_params = len(self.dist.parameters)

        # Check args for shapes
        for arg in args:
            if isinstance(arg, (jnp.ndarray, np.ndarray)):
                batch_shapes.append(arg.shape)

        # Check kwargs for shapes
        for v in kwds.values():
            if isinstance(v, (jnp.ndarray, np.ndarray)):
                batch_shapes.append(v.shape)

        if len(batch_shapes) > num_params:
            raise ValueError(
                f"Too many args/kwargs provided for distribution {self.dist.__class__.__name__}."
                f"Expected {self.dist.parameters} shapes, got {len(batch_shapes)}."
            )

        # Compute final shapes assuming rv is univariate
        if len(batch_shapes) > 0:
            batch_shape = jnp.broadcast_shapes(*batch_shapes)
        else:
            batch_shape = ()

        event_shape = ()

        return batch_shape, event_shape

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

        # Get name from class or distribution
        name = (
            getattr(self, 'name', None)
            or getattr(self.dist, 'name', None)
            or self.dist.__class__.__name__
        )
        return f"{name}({params_str})"

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
        """Cumulative distribution function (i.e. P(X <= x)).

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at x
        """
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x: ArrayLike):
        """Log of the cumulative distribution function (i.e. log(P(X <= x))).

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.

        Returns
        -------
        logcdf : ndarray or scalar
            Log of the cumulative distribution function evaluated at x
        """
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function (inverse of cdf).

        Parameters
        ----------
        q : array_like
            Probability at which to evaluate the inverse cumulative distribution function.

        Returns
        -------
        ppf : ndarray or scalar
            Percent point function evaluated at q
        """
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q: ArrayLike):
        """Inverse survival function (1 - ppf) of the frozen distribution."""
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, shape: Tuple[int, ...] = (), **kwargs):
        """Random variates of the frozen distribution.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            The random key used for sampling
        shape : tuple of ints, optional
            The shape of the samples to draw. Default is ().

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        return self.dist.rvs(rng, *self.args, shape=shape, **self.kwds)

    def sf(self, x: ArrayLike):
        """Survival function (1 - cdf)."""
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x: ArrayLike):
        """Log of the survival function (1 - cdf)."""
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments: str = 'mv'):
        """Returns mean, variance, skew, or kurtosis of the frozen distribution.

        Parameters
        ----------
        moments : str, optional
            Which moments to compute: 'mv' (default), 'v', 's', 'k'.

        Returns
        -------
        stats : ndarray or scalar
            Mean, variance, skew, or kurtosis of the distribution
        """
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
        """Entropy of the distribution."""
        return self.dist.entropy(*self.args, **self.kwds)

    def interval(self, confidence: Optional[ArrayLike] = None):
        """Confidence interval with equal areas around the median of the distribution.

        Parameters
        ----------
        confidence : array_like, optional
            Confidence level for the interval. Default is 0.95.

        Returns
        """
        return self.dist.interval(confidence, *self.args, **self.kwds)

    def support(self):
        """Support of the frozen distribution."""
        return self.dist.support(*self.args, **self.kwds)


class rv_discrete_frozen(rv_frozen):
    def pmf(self, x: ArrayLike):
        """Probability mass function of the distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability mass function.

        Returns
        -------
        pmf : ndarray or scalar
            Probability mass function evaluated at x
        """
        return self.dist.pmf(x, *self.args, **self.kwds)

    def logpmf(self, x: ArrayLike):
        """Log of the probability mass function of the distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability mass function.

        Returns
        -------
        logpmf : ndarray or scalar
            Log of the probability mass function evaluated at x
        """
        return self.dist.logpmf(x, *self.args, **self.kwds)

    def mode(self):
        """Mode of the distribution.

        Returns
        -------
        mode : ndarray or scalar
            Mode of the distribution
        """
        return self.dist.mode(*self.args, **self.kwds)

    def logpdf(self, x: ArrayLike):
        """Log of the probability density function of the distribution."""
        return self.dist.logpdf(x, *self.args, **self.kwds)

    def pdf(self, x: ArrayLike):
        """Probability density function of the distribution."""
        return self.dist.pdf(x, *self.args, **self.kwds)


class rv_continuous_frozen(rv_frozen):
    def pdf(self, x: ArrayLike):
        """Probability density function of the distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at x
        """
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x: ArrayLike):
        """Log of the probability density function of the distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.

        Returns
        -------
        logpdf : ndarray or scalar
            Log of the probability density function evaluated at x
        """
        return self.dist.logpdf(x, *self.args, **self.kwds)

    def mode(self):
        """Mode of the distribution.

        Returns
        -------
        mode : ndarray or scalar
            Mode of the distribution
        """
        return self.dist.mode(*self.args, **self.kwds)

    def cdf(self, x: ArrayLike):
        """Cumulative distribution function of the distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.

        Returns
        -------
        cdf : ndarray or scalar
            Cumulative distribution function evaluated at x
        """
        return self.dist.cdf(x, *self.args, **self.kwds)

    def ppf(self, q: ArrayLike):
        """Percent point function (inverse of cdf) of the distribution.

        Parameters
        ----------
        q : array_like
            Probability at which to evaluate the inverse cumulative distribution function.

        Returns
        -------
        ppf : ndarray or scalar
            Percent point function evaluated at q
        """
        return self.dist.ppf(q, *self.args, **self.kwds)

    def rvs(self, rng: PRNGKeyArray, *args, shape: Tuple[int, ...] = (), **kwargs):
        """Random variates of the distribution.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            The random key used for sampling
        shape : tuple of ints, optional
            The shape of the samples to draw. Default is ().

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        return self.dist.rvs(rng, *self.args, shape=shape, **self.kwds)


# Register frozen classes as JAX PyTrees
jax.tree_util.register_pytree_node_class(rv_frozen)
jax.tree_util.register_pytree_node_class(rv_continuous_frozen)
jax.tree_util.register_pytree_node_class(rv_discrete_frozen)
