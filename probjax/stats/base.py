"""
Statistical Distributions Base Classes (:mod:`probjax.stats.base`)
=================================================================

This module contains the base classes for continuous and discrete random variables
that provide a SciPy-like API. This closely follows the structure of scipy.stats._distn_infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from probjax.stats.constraints import Constraint
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = [
    "rv_generic",
    "rv_continuous",
    "rv_multivariate",
    "rv_spherical",
    "rv_discrete",
    "rv_continuous_frozen",
    "rv_multivariate_frozen",
    "rv_spherical_frozen",
    "rv_discrete_frozen",
]


class rv_generic(ABC):
    """Generic random variable class for common functionality."""

    name: ClassVar[Optional[str]] = None
    parameters: ClassVar[Mapping[str, Constraint]] = {}
    parameter_aliases: ClassVar[Mapping[str, str]] = {}

    def __init__(self, name: Optional[str] = None):
        if name is not None:
            type(self).name = name

    def __call__(self, *args: Any, **kwds: Any) -> "rv_frozen":
        """Call the distribution with the given arguments."""
        return self.freeze(*args, **kwds)

    @staticmethod
    def _format_frozen_class_name(name: Optional[str], fallback: str) -> str:
        """Create a stable class name for generated frozen distributions."""
        raw_name = name or fallback
        parts = "".join(ch if ch.isalnum() else "_" for ch in str(raw_name)).split("_")
        camel = "".join(part.capitalize() for part in parts if part)
        return f"Frozen{camel or 'Distribution'}"

    def _get_or_create_frozen_class(
        self, base_frozen_cls: type["rv_frozen"]
    ) -> type["rv_frozen"]:
        """Get a cached frozen class for this distribution instance."""
        cache = getattr(self, "_frozen_cls_cache", None)
        if cache is not None and issubclass(cache, base_frozen_cls):
            return cast(type["rv_frozen"], cache)

        class_name = self._format_frozen_class_name(
            getattr(self, "name", None), self.__class__.__name__
        )
        frozen_cls = cast(
            type["rv_frozen"],
            FrozenDistributionMeta(
                class_name,
                (base_frozen_cls,),
                {},
                dist=self,
            ),
        )
        setattr(self, "_frozen_cls_cache", frozen_cls)
        return frozen_cls

    def _freeze_as(
        self, base_frozen_cls: type["rv_frozen"], *args: Any, **kwds: Any
    ) -> "rv_frozen":
        """Freeze a distribution into the requested frozen base class."""
        frozen_cls = self._get_or_create_frozen_class(base_frozen_cls)
        return frozen_cls(self, *args, **kwds)

    @classmethod
    def _parse_args(
        cls, *args: Any, **kwds: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Parse arguments for the distribution."""
        # Move important kwds to args
        params = dict(cls.parameters)
        for param_name in params:
            if param_name in kwds:
                args = args + (kwds.pop(param_name),)
        return args, kwds

    def freeze(self, *args: Any, **kwds: Any) -> "rv_frozen":
        """Freeze the distribution for the given arguments."""
        return self._freeze_as(rv_frozen, *args, **kwds)

    @classmethod
    @abstractmethod
    def support(cls, *args, **kwds) -> Constraint:
        """Support of the distribution."""
        ...

    @classmethod
    @abstractmethod
    def rvs(
        cls, rng: RngKey, *args: Any, shape: Tuple[int, ...] = (), **kwargs: Any
    ) -> Array:
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
    def mean(cls, *args: Any, **kwds: Any) -> ArrayLike:
        """Mean of the distribution."""
        raise NotImplementedError("Mean is not implemented for this distribution.")

    @classmethod
    def mode(cls, *args: Any, **kwds: Any) -> ArrayLike:
        """Mode of the distribution."""
        raise NotImplementedError("Mode is not implemented for this distribution.")

    @classmethod
    def var(cls, *args: Any, **kwds: Any) -> ArrayLike:
        """Variance of the distribution."""
        raise NotImplementedError("Variance is not implemented for this distribution.")

    @classmethod
    def std(cls, *args: Any, **kwds: Any) -> ArrayLike:
        """Standard deviation of the distribution."""
        return jnp.sqrt(cls.var(*args, **kwds))

    @classmethod
    def cdf(cls, *args: Any, **kwds: Any) -> ArrayLike:
        """Cumulative distribution function of the RV."""
        raise NotImplementedError("CDF is not implemented for this distribution.")

    @classmethod
    @abstractmethod
    def logpdf(cls, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Log probability of the distribution at the given value."""
        ...

    @classmethod
    def logcdf(cls, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Log of the cumulative distribution function at x of the given RV."""
        return jnp.log(cls.cdf(x, *args, **kwds))

    @classmethod
    def sf(cls, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Survival function (1 - cdf) at x of the given RV."""
        cdf_val = jnp.asarray(cls.cdf(x, *args, **kwds))
        return jnp.ones_like(cdf_val) - cdf_val

    @classmethod
    def logsf(cls, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Log of the survival function at x of the given RV."""
        return jnp.log(cls.sf(x, *args, **kwds))

    @classmethod
    def ppf(cls, *args: Any, **kwds: Any) -> ArrayLike:
        """Percent point function (inverse of cdf) of the RV."""
        raise NotImplementedError("PPF is not implemented for this distribution.")

    @classmethod
    def isf(cls, q: ArrayLike, *args: Any, **kwds: Any) -> ArrayLike:
        """Inverse survival function (1 - ppf) of the RV."""
        q = jnp.asarray(q)
        return cls.ppf(1.0 - q, *args, **kwds)

    @classmethod
    def entropy(cls, *args: Any, **kwds: Any) -> ArrayLike:
        """Entropy of the RV."""
        raise NotImplementedError("Entropy is not implemented for this distribution.")

    @classmethod
    def median(cls, *args: Any, **kwds: Any) -> ArrayLike:
        """Median of the distribution."""
        args, kwds = cls._parse_args(*args, **kwds)
        return cls.ppf(0.5, *args, **kwds)

    @classmethod
    def interval(
        cls, alpha: ArrayLike, *args: Any, **kwds: Any
    ) -> tuple[ArrayLike, ArrayLike]:
        """Confidence interval with equal areas around the median."""
        args, kwds = cls._parse_args(*args, **kwds)
        alpha = jnp.asarray(alpha)
        q1 = (1.0 - alpha) / 2
        q2 = (1.0 + alpha) / 2
        a = cls.ppf(q1, *args, **kwds)
        b = cls.ppf(q2, *args, **kwds)
        return a, b

    @classmethod
    def moment(cls, n: int, *args: Any, **kwds: Any) -> ArrayLike:
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
    def skew(cls, *args: Any, **kwds: Any) -> ArrayLike:
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
    def kurtosis(cls, *args: Any, **kwds: Any) -> ArrayLike:
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
    def fit(cls, data: ArrayLike, **kwds: Any) -> tuple[Array, ...]:
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
    def natural_parameters(cls, *args: Any, **kwds: Any) -> Array:
        """Natural parameters of the distribution."""
        raise NotImplementedError(
            "Natural parameters are not implemented for this distribution."
        )

    @classmethod
    def sufficient_statistics(cls, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Sufficient statistics of the distribution."""
        raise NotImplementedError(
            "Sufficient statistics are not implemented for this distribution."
        )

    @classmethod
    def log_partition(cls, *args: Any, **kwds: Any) -> Array:
        """Log partition function of the distribution."""
        raise NotImplementedError(
            "Log partition function is not implemented for this distribution."
        )

    @classmethod
    def fit(cls, data: ArrayLike, **kwds: Any) -> tuple[Array, ...]:
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
        init_params: dict[str, Array] = {}
        for param_name, constraint in cls.parameters.items():
            if param_name in kwds:
                init_params[param_name] = kwds.pop(param_name)
            else:
                # Use default value from constraint
                default_value = getattr(constraint, "default_value", None)
                if default_value is None:
                    raise ValueError(
                        f"Constraint {constraint!r} lacks a default value for parameter"
                        f" '{param_name}'. Provide an explicit initial value."
                    )
                init_params[param_name] = jnp.asarray(default_value)

        # Convert to flat array for optimization
        init_flat = jnp.concatenate([jnp.ravel(v) for v in init_params.values()])

        def neg_log_likelihood(params_flat: Array) -> Array:
            # Reshape parameters according to their original shapes
            start_idx = 0
            params: dict[str, Array] = {}
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

            # Compute negative log likelihood using sufficient statistics
            return -jnp.sum(eta * T - cls.log_partition(eta))

        # Optimize
        from jax.scipy.optimize import minimize

        result = minimize(neg_log_likelihood, init_flat, method="BFGS", **kwds)

        if not result.success:
            message = getattr(result, "message", "unknown error")
            raise ValueError(f"Optimization failed: {message}")

        # Reshape parameters back to their original shapes
        start_idx = 0
        fitted_params: dict[str, Array] = {}
        for param_name, param_value in init_params.items():
            param_size = jnp.size(param_value)
            param_shape = jnp.shape(param_value)
            param = result.x[start_idx : start_idx + param_size].reshape(param_shape)
            fitted_params[param_name] = param
            start_idx += param_size

        return tuple(fitted_params.values())


class rv_continuous(rv_generic):
    """Base class for continuous random variables."""

    def freeze(self, *args: Any, **kwds: Any) -> "rv_continuous_frozen":
        """Freeze the distribution for the given arguments."""
        return cast(
            "rv_continuous_frozen",
            self._freeze_as(rv_continuous_frozen, *args, **kwds),
        )

    @classmethod
    def pdf(cls, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Probability density function at x of the given RV."""
        args, kwds = cls._parse_args(*args, **kwds)
        return jnp.exp(cls.logpdf(x, *args, **kwds))

    @classmethod
    def logpdf(cls, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Log of the probability density function at x of the given RV."""
        raise NotImplementedError("Logpdf is not implemented for this distribution.")

    @classmethod
    def fit(cls, data: ArrayLike, **kwds: Any) -> tuple[Array, ...]:
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
        init_params: dict[str, Array] = {}
        for param_name, constraint in cls.parameters.items():
            if param_name in kwds:
                init_params[param_name] = kwds.pop(param_name)
            else:
                # Use default value from constraint
                default_value = getattr(constraint, "default_value", None)
                if default_value is None:
                    raise ValueError(
                        f"Constraint {constraint!r} lacks a default value for parameter"
                        f" '{param_name}'. Provide an explicit initial value."
                    )
                init_params[param_name] = jnp.asarray(default_value)

        # Convert to flat array for optimization
        init_flat = jnp.concatenate([jnp.ravel(v) for v in init_params.values()])

        def neg_log_likelihood(params_flat: Array) -> Array:
            # Reshape parameters according to their original shapes
            start_idx = 0
            params: dict[str, Array] = {}
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
            message = getattr(result, "message", "unknown error")
            raise ValueError(f"Optimization failed: {message}")

        # Reshape parameters back to their original shapes
        start_idx = 0
        fitted_params: dict[str, Array] = {}
        for param_name, param_value in init_params.items():
            param_size = jnp.size(param_value)
            param_shape = jnp.shape(param_value)
            param = result.x[start_idx : start_idx + param_size].reshape(param_shape)
            fitted_params[param_name] = param
            start_idx += param_size

        return tuple(fitted_params.values())


class rv_multivariate(rv_continuous):
    """Base class for multivariate continuous random variables."""

    multivariate: ClassVar[bool] = True

    def freeze(self, *args: Any, **kwds: Any) -> "rv_continuous_frozen":
        """Freeze the multivariate distribution for the given arguments."""
        frozen_cls = cast(type["rv_frozen"], globals()["rv_multivariate_frozen"])
        return cast(
            "rv_continuous_frozen",
            self._freeze_as(frozen_cls, *args, **kwds),
        )

    @classmethod
    @abstractmethod
    def _multivariate_batch_event_shape(
        cls, *args: Any, **kwds: Any
    ) -> tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Infer batch and event shape for multivariate distributions."""
        ...


class rv_spherical(rv_multivariate):
    """Base class for spherical distributions on the unit sphere."""

    def freeze(self, *args: Any, **kwds: Any) -> "rv_continuous_frozen":
        """Freeze the spherical distribution for the given arguments."""
        frozen_cls = cast(type["rv_frozen"], globals()["rv_spherical_frozen"])
        return cast(
            "rv_continuous_frozen",
            self._freeze_as(frozen_cls, *args, **kwds),
        )

    @classmethod
    @abstractmethod
    def mean_direction_vector(cls, *args: Any, **kwds: Any) -> ArrayLike:
        """Representative principal direction of the spherical distribution."""
        ...

    @classmethod
    @abstractmethod
    def mean_direction_dyad(cls, *args: Any, **kwds: Any) -> Array:
        """Expected dyadic product :math:`E[XX^T]` for spherical random vectors."""
        ...

    @classmethod
    def dispersion(cls, *args: Any, **kwds: Any) -> Array:
        """Dispersion matrix defined as :math:`E[XX^T] - I/d`."""
        dyad = jnp.asarray(cls.mean_direction_dyad(*args, **kwds))
        dim = dyad.shape[-1]
        identity = jnp.eye(dim, dtype=dyad.dtype) / jnp.asarray(dim, dtype=dyad.dtype)
        identity = jnp.broadcast_to(identity, dyad.shape)
        return dyad - identity

    @classmethod
    def axial_dispersion(cls, *args: Any, **kwds: Any) -> Array:
        """Dispersion along the principal axis :math:`1 - mu^T E[XX^T] mu`."""
        mean_vec = jnp.asarray(cls.mean_direction_vector(*args, **kwds))
        dyad = jnp.asarray(cls.mean_direction_dyad(*args, **kwds))
        axial_moment = jnp.einsum("...i,...ij,...j->...", mean_vec, dyad, mean_vec)
        one = jnp.asarray(1.0, dtype=axial_moment.dtype)
        zero = jnp.asarray(0.0, dtype=axial_moment.dtype)
        return jnp.maximum(zero, one - axial_moment)


class rv_discrete(rv_generic):
    """Base class for discrete random variables."""

    def freeze(self, *args: Any, **kwds: Any) -> "rv_discrete_frozen":
        """Freeze the distribution for the given arguments."""
        return cast(
            "rv_discrete_frozen",
            self._freeze_as(rv_discrete_frozen, *args, **kwds),
        )

    @classmethod
    @abstractmethod
    def pmf(cls, k: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Probability mass function at k of the given RV."""
        ...

    @classmethod
    def logpmf(cls, k: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Log of the probability mass function at k of the given RV."""
        return jnp.log(cls.pmf(k, *args, **kwds))

    @classmethod
    def logpdf(cls, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Log of the probability density function at x of the given RV."""
        return cls.logpmf(x, *args, **kwds)

    @classmethod
    def pdf(cls, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Probability density function at x of the given RV."""
        return jnp.exp(cls.logpdf(x, *args, **kwds))


class FrozenDistributionMeta(type):
    """Metaclass for frozen distributions that inherits name and docstrings."""

    def __new__(mcs, name, bases, namespace, dist=None):
        if dist is not None:
            impl_class = dist if isinstance(dist, type) else dist.__class__
            dist_name = getattr(dist, "name", None) or getattr(impl_class, "name", None)
            if dist_name is not None:
                namespace.setdefault("name", dist_name)
            namespace.setdefault("dist", dist)

        return super().__new__(mcs, name, bases, namespace)


class rv_frozen(metaclass=FrozenDistributionMeta):
    def __init__(
        self,
        dist,
        *args,
        **kwds,
    ):
        self.args = tuple(args)
        self.kwds = dict(kwds)
        self.dist = dist

        self._refresh_parameter_state()

        self._batch_shape, self._event_shape = self._compute_batch_and_event_shape(
            *self.args, **self.kwds
        )

        for param_name, value in self._parameter_values.items():
            if hasattr(type(self), param_name):
                continue
            object.__setattr__(self, param_name, value)

        super().__init__()

    def _build_parameter_dict(self) -> dict[str, Any]:
        """Bind parameter names to values based on the distribution signature."""
        parameters = tuple(getattr(self.dist, "parameters", {}).keys())
        values: dict[str, Any] = {}
        for idx, name in enumerate(parameters):
            if idx < len(self.args):
                values[name] = self.args[idx]
        for name in parameters:
            if name in self.kwds:
                values[name] = self.kwds[name]
        return values

    def _build_parameter_aliases(self) -> dict[str, str]:
        """Build alias lookup table for distribution parameters."""
        aliases = dict(getattr(self.dist, "parameter_aliases", {}))
        if (
            "alpha" in self._parameter_values
            and "concentration" not in self._parameter_values
        ):
            aliases.setdefault("concentration", "alpha")
        if "beta" in self._parameter_values and "rate" not in self._parameter_values:
            aliases.setdefault("rate", "beta")
        if (
            "cov" in self._parameter_values
            and "covariance_matrix" not in self._parameter_values
        ):
            aliases.setdefault("covariance_matrix", "cov")
        return aliases

    def _build_call_kwargs(self) -> dict[str, Any]:
        """Build canonical kwargs for calling distribution methods."""
        call_kwds = dict(self.kwds)
        parameters = tuple(getattr(self.dist, "parameters", {}).keys())
        for idx, name in enumerate(parameters):
            if idx < len(self.args) and name not in call_kwds:
                call_kwds[name] = self.args[idx]
        return call_kwds

    def _refresh_parameter_state(self) -> None:
        """Refresh cached parameter mappings after state updates."""
        self._parameter_values = self._build_parameter_dict()
        self._parameter_aliases = self._build_parameter_aliases()
        self._call_kwds = self._build_call_kwargs()

    def _call_dist(self, method_name: str, *args: Any, **kwds: Any) -> Any:
        """Call a distribution method using canonical frozen parameters."""
        self._refresh_parameter_state()
        method = getattr(self.dist, method_name)
        call_kwds = dict(self._call_kwds)
        call_kwds.update(kwds)
        return method(*args, **call_kwds)

    def _compute_batch_and_event_shape(
        self, *args: Any, **kwds: Any
    ) -> tuple[Tuple[int, ...], Tuple[int, ...]]:
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
            batch_shape_raw = jnp.broadcast_shapes(*batch_shapes)
        else:
            batch_shape_raw = ()

        batch_shape: Tuple[int, ...] = tuple(int(dim) for dim in batch_shape_raw)
        event_shape: Tuple[int, ...] = tuple()

        return batch_shape, event_shape

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return self._batch_shape

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return self._event_shape

    def __getattr__(self, name: str) -> Any:
        """Expose frozen parameters as attributes for SciPy-like ergonomics."""
        self._refresh_parameter_state()
        if name in self._parameter_values:
            return self._parameter_values[name]
        alias = self._parameter_aliases.get(name)
        if alias is not None and alias in self._parameter_values:
            return self._parameter_values[alias]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

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
        return self._call_dist("cdf", x)

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
        return self._call_dist("logcdf", x)

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
        return self._call_dist("ppf", q)

    def isf(self, q: ArrayLike):
        """Inverse survival function (1 - ppf) of the frozen distribution."""
        return self._call_dist("isf", q)

    def rvs(self, rng: RngKey, shape: Tuple[int, ...] = (), **kwargs):
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
        return self._call_dist("rvs", rng, shape=shape, **kwargs)

    def sf(self, x: ArrayLike):
        """Survival function (1 - cdf)."""
        return self._call_dist("sf", x)

    def logsf(self, x: ArrayLike):
        """Log of the survival function (1 - cdf)."""
        return self._call_dist("logsf", x)

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
        return self._call_dist("stats", moments=moments)

    def median(self):
        """Median of the distribution."""
        return self._call_dist("median")

    def mean(self):
        """Mean of the distribution."""
        return self._call_dist("mean")

    def var(self):
        """Variance of the distribution."""
        return self._call_dist("var")

    def std(self):
        """Standard deviation of the distribution."""
        return self._call_dist("std")

    def moment(self, order: Optional[int] = None):
        """Non-central moment of the distribution."""
        return self._call_dist("moment", order)

    def entropy(self):
        """Entropy of the distribution."""
        return self._call_dist("entropy")

    def interval(self, confidence: Optional[ArrayLike] = None):
        """Confidence interval with equal areas around the median of the distribution.

        Parameters
        ----------
        confidence : array_like, optional
            Confidence level for the interval. Default is 0.95.

        Returns
        """
        return self._call_dist("interval", confidence)

    def support(self):
        """Support of the frozen distribution."""
        return self._call_dist("support")


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
        return self._call_dist("pmf", x)

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
        return self._call_dist("logpmf", x)

    def mode(self):
        """Mode of the distribution.

        Returns
        -------
        mode : ndarray or scalar
            Mode of the distribution
        """
        return self._call_dist("mode")

    def logpdf(self, x: ArrayLike):
        """Log of the probability density function of the distribution."""
        return self._call_dist("logpdf", x)

    def pdf(self, x: ArrayLike):
        """Probability density function of the distribution."""
        return self._call_dist("pdf", x)


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
        return self._call_dist("pdf", x)

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
        return self._call_dist("logpdf", x)

    def mode(self):
        """Mode of the distribution.

        Returns
        -------
        mode : ndarray or scalar
            Mode of the distribution
        """
        return self._call_dist("mode")

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
        return self._call_dist("cdf", x)

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
        return self._call_dist("ppf", q)

    def rvs(self, rng: RngKey, *args, shape: Tuple[int, ...] = (), **kwargs):
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
        return self._call_dist("rvs", rng, shape=shape, **kwargs)


class rv_multivariate_frozen(rv_continuous_frozen):
    """Frozen multivariate distribution with explicit shape inference."""

    def _compute_batch_and_event_shape(
        self, *args: Any, **kwds: Any
    ) -> tuple[Tuple[int, ...], Tuple[int, ...]]:
        shape_fn = getattr(self.dist, "_multivariate_batch_event_shape", None)
        if not callable(shape_fn):
            raise NotImplementedError(
                f"{self.dist.__class__.__name__} must implement "
                "_multivariate_batch_event_shape for multivariate freezing."
            )

        batch_shape, event_shape = cast(
            tuple[Tuple[int, ...], Tuple[int, ...]],
            shape_fn(*args, **kwds),
        )
        return (
            tuple(int(dim) for dim in batch_shape),
            tuple(int(dim) for dim in event_shape),
        )


class rv_spherical_frozen(rv_multivariate_frozen):
    """Frozen spherical distribution exposing directional statistics."""

    def mean_direction_vector(self):
        """Representative principal direction of the spherical distribution."""
        return self._call_dist("mean_direction_vector")

    def mean_direction_dyad(self):
        """Expected dyadic product :math:`E[XX^T]` of the spherical distribution."""
        return self._call_dist("mean_direction_dyad")

    def dispersion(self):
        """Dispersion matrix defined as :math:`E[XX^T] - I/d`."""
        return self._call_dist("dispersion")

    def axial_dispersion(self):
        """Dispersion along the principal axis."""
        return self._call_dist("axial_dispersion")


# Register frozen classes as JAX PyTrees
jax.tree_util.register_pytree_node_class(rv_frozen)
jax.tree_util.register_pytree_node_class(rv_continuous_frozen)
jax.tree_util.register_pytree_node_class(rv_multivariate_frozen)
jax.tree_util.register_pytree_node_class(rv_spherical_frozen)
jax.tree_util.register_pytree_node_class(rv_discrete_frozen)
