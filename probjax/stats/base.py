"""
Statistical Distributions Base Classes (:mod:`probjax.stats.base`)
=================================================================

This module contains the base classes for continuous and discrete random variables
that provide a SciPy-like API. It follows scipy.stats._distn_infrastructure.
"""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.optimize import minimize

import probjax.stats.constraints as stats_constraints
from probjax.stats.constraint_registry import biject_to
from probjax.utils.guards import is_array_like
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = [
    "DistributionParams",
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


@dataclass(frozen=True)
class _FrozenArgs:
    args: tuple[Any, ...]
    kwds: dict[str, Any]
    parameter_values: dict[str, ArrayLike]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DistributionParams:
    """Unconstrained parameters with a distribution stored as static metadata."""

    dist: rv_generic
    params: Mapping[str, ArrayLike]

    def tree_flatten(self):
        return (dict(self.params),), self.dist

    @classmethod
    def tree_unflatten(cls, dist, children):
        return cls(dist, children[0])

    def constrain(self) -> "rv_frozen":
        return self.dist.from_params(self.dist.params_from_unconstrained(self.params))


def _is_frozen_dist(component: Any) -> bool:
    """PyTree leaf predicate: a frozen distribution instance."""
    # Deferred reference: ``rv_frozen`` is defined below.
    return isinstance(component, _RV_FROZEN_TYPES)


def _is_distribution_params(component: Any) -> bool:
    """PyTree leaf predicate: unconstrained distribution parameters."""
    return isinstance(component, DistributionParams)


def _frozen_to_unconstrained(component: Any) -> DistributionParams:
    return DistributionParams(component.dist, component.unconstrained_params)


def _unconstrained_to_frozen(component: DistributionParams) -> Any:
    return component.constrain()


_RV_FROZEN_TYPES: tuple = ()


def _rv_bind(rng, args, shape, dist, name, rvs_fn, logpdf_fn, kwds):
    """Bind a sample through the ``rv_p`` primitive (tracing + eager paths)."""
    from probjax.core.custom_primitives.random_variable import rv_p

    return rv_p.bind(
        rng,
        *args,
        shape=shape,
        dist=dist,
        name=name,
        rvs_fn=rvs_fn,
        logpdf_fn=logpdf_fn,
        kwds=kwds,
    )


class rv_generic(ABC):
    """Generic random variable class for common functionality."""

    name: ClassVar[Optional[str]] = None
    parameters: ClassVar[Mapping[str, stats_constraints.Constraint]] = {}
    parameter_aliases: ClassVar[Mapping[str, str]] = {}
    extra_frozen_kwds: ClassVar[frozenset[str]] = frozenset()

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
        self._frozen_cls_cache = frozen_cls
        return frozen_cls

    @staticmethod
    def _distribution_display_name(dist: Any) -> str:
        return (
            getattr(dist, "name", None)
            or getattr(dist, "__name__", None)
            or dist.__class__.__name__
        )

    @classmethod
    def _bind_frozen_args_for_dist(
        cls, dist: Any, args: tuple[Any, ...], kwds: Mapping[str, Any]
    ) -> _FrozenArgs:
        """Validate and bind frozen distribution arguments."""
        parameters = tuple(getattr(dist, "parameters", {}).keys())
        extra_kwds = set(getattr(dist, "extra_frozen_kwds", ()))
        kwds_dict = dict(kwds)
        dist_name = cls._distribution_display_name(dist)

        if len(args) > len(parameters):
            raise TypeError(
                f"{dist_name} expected at most {len(parameters)} positional "
                f"arguments, got {len(args)}."
            )

        positional_names = set(parameters[: len(args)])
        duplicate_names = sorted(positional_names.intersection(kwds_dict))
        if duplicate_names:
            names = ", ".join(repr(name) for name in duplicate_names)
            raise TypeError(f"{dist_name} got multiple values for argument {names}.")

        valid_kwds = set(parameters).union(extra_kwds)
        unexpected = sorted(set(kwds_dict).difference(valid_kwds))
        if unexpected:
            names = ", ".join(repr(name) for name in unexpected)
            raise TypeError(f"{dist_name} got unexpected keyword argument {names}.")

        parameter_values = {
            name: value for name, value in zip(parameters, args, strict=False)
        }
        parameter_values.update({
            name: kwds_dict[name] for name in parameters if name in kwds_dict
        })
        return _FrozenArgs(args=args, kwds=kwds_dict, parameter_values=parameter_values)

    def _freeze_as(
        self, base_frozen_cls: type["rv_frozen"], *args: Any, **kwds: Any
    ) -> "rv_frozen":
        """Freeze a distribution into the requested frozen base class."""
        frozen_cls = self._get_or_create_frozen_class(base_frozen_cls)
        return frozen_cls(self, *args, **kwds)

    def _freeze_as_named(self, cls_name: str, *args: Any, **kwds: Any) -> "rv_frozen":
        """Freeze using a frozen class looked up by name (avoids import cycles)."""
        frozen_cls = cast(type["rv_frozen"], globals()[cls_name])
        return self._freeze_as(frozen_cls, *args, **kwds)

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

    def from_params(
        self, params: Optional[Mapping[str, Any]] = None, **kwds: Any
    ) -> "rv_frozen":
        """Create a frozen distribution from name-keyed parameters."""
        values = dict(params or {})
        values.update(kwds)
        args = []
        for name in self.parameters:
            if name not in values:
                break
            args.append(values.pop(name))
        return self.freeze(*args, **values)

    @classmethod
    def _convert_params(cls, params: Mapping[str, Any], *, to_unconstrained: bool):
        """Shared loop for ``params_to/from_unconstrained`` conversions."""
        converted = {}
        for name, value in params.items():
            constraint = cls.parameters.get(name)
            if isinstance(constraint, stats_constraints.Distribution):
                if to_unconstrained:
                    converted[name] = jax.tree_util.tree_map(
                        _frozen_to_unconstrained,
                        value,
                        is_leaf=_is_frozen_dist,
                    )
                else:
                    converted[name] = jax.tree_util.tree_map(
                        _unconstrained_to_frozen,
                        value,
                        is_leaf=_is_distribution_params,
                    )
                continue
            if not isinstance(constraint, stats_constraints.Constraint):
                converted[name] = value
                continue
            try:
                bijection = biject_to(constraint)
                converted[name] = (
                    bijection.inv(value) if to_unconstrained else bijection(value)
                )
            except NotImplementedError:
                converted[name] = value
        return converted

    @classmethod
    def params_to_unconstrained(cls, params: Mapping[str, Any]) -> dict[str, Any]:
        """Map constrained parameters to an optimization-friendly pytree."""
        return cls._convert_params(params, to_unconstrained=True)

    @classmethod
    def params_from_unconstrained(cls, params: Mapping[str, Any]) -> dict[str, Any]:
        """Map unconstrained parameters back to their declared supports."""
        return cls._convert_params(params, to_unconstrained=False)

    def rvs(
        self,
        rng: RngKey,
        *args: Any,
        shape: Tuple[int, ...] = (),
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Array:
        """Random variates of given shape.

        Calls through the `rv_p` primitive so traced execution records a random
        variable site while eager execution remains a direct sample.
        """
        return _rv_bind(
            rng,
            args,
            shape,
            self,
            name,
            type(self)._rvs_impl,
            type(self).logpdf,
            kwargs,
        )

    @classmethod
    @abstractmethod
    def support(cls, *args, **kwds) -> stats_constraints.Constraint:
        """Support of the distribution."""
        ...

    @classmethod
    @abstractmethod
    def _rvs_impl(
        cls, rng: RngKey, *args: Any, shape: Tuple[int, ...] = (), **kwargs: Any
    ) -> Array:
        """Implementation for random variate sampling.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            The random key used for sampling
        *args : array_like
            Shape parameters for the distribution
        shape : tuple of ints, optional
            The shape of the samples to draw
        name : str, optional
            Optional site name used when tracing probabilistic programs.
            If omitted, a unique name is generated from the distribution name.
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

    @classmethod
    def fit_params(cls, data: ArrayLike, **kwds: Any) -> dict[str, Any]:
        """Fit and return parameters keyed by their declared names."""
        fitted = cls.fit(data, **kwds)
        values = fitted if isinstance(fitted, tuple) else (fitted,)
        return dict(zip(cls.parameters, values, strict=False))

    @classmethod
    def _fit_mle(cls, data: ArrayLike, **kwds: Any) -> tuple[Array, ...]:
        """Fit by optimizing the likelihood in unconstrained parameter space."""
        init_params = {}
        for name in cls.parameters:
            if name not in kwds:
                raise ValueError(f"Provide an initial value for parameter {name!r}.")
            init_params[name] = jnp.asarray(kwds.pop(name))

        initial, unravel = ravel_pytree(cls.params_to_unconstrained(init_params))

        def objective(flat_params):
            params = cls.params_from_unconstrained(unravel(flat_params))
            return -jnp.sum(cls.logpdf(data, **params))

        result = minimize(objective, initial, method="BFGS", **kwds)
        if not bool(jnp.all(jnp.isfinite(result.x))):
            message = getattr(result, "message", "unknown error")
            raise ValueError(f"Optimization failed: {message}")

        fitted = cls.params_from_unconstrained(unravel(result.x))
        return tuple(fitted[name] for name in cls.parameters)


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
        """Maximum likelihood estimation in unconstrained parameter space."""
        return cls._fit_mle(data, **kwds)


class rv_continuous(rv_generic):
    """Base class for continuous random variables."""

    @classmethod
    def support(cls, *args: Any, **kwds: Any) -> stats_constraints.Constraint:
        """Support of the distribution; the whole real line unless overridden."""
        return stats_constraints.real

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
        """Maximum likelihood estimation in unconstrained parameter space."""
        return cls._fit_mle(data, **kwds)


class rv_multivariate(rv_continuous):
    """Base class for multivariate continuous random variables."""

    multivariate: ClassVar[bool] = True

    def freeze(self, *args: Any, **kwds: Any) -> "rv_continuous_frozen":
        """Freeze the multivariate distribution for the given arguments."""
        return cast(
            "rv_continuous_frozen",
            self._freeze_as_named("rv_multivariate_frozen", *args, **kwds),
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
        return cast(
            "rv_continuous_frozen",
            self._freeze_as_named("rv_spherical_frozen", *args, **kwds),
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
    def pmf(cls, k: ArrayLike, *args: Any, **kwds: Any) -> Array:
        """Probability mass function at k of the given RV."""
        return jnp.exp(cls.logpmf(k, *args, **kwds))

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


class DistributionAPI(ABC):
    """Common distribution interface shared by frozen and module-backed dists."""

    @property
    @abstractmethod
    def batch_shape(self) -> Tuple[int, ...]:
        """Shape of the batch (non-event) dimensions of the distribution.

        Returns
        -------
        tuple of int
            Batch dimensions of the distribution.
        """
        ...

    @property
    @abstractmethod
    def event_shape(self) -> Tuple[int, ...]:
        """Shape of the event dimensions of the distribution.

        Returns
        -------
        tuple of int
            Event dimensions of the distribution.
        """
        ...

    @abstractmethod
    def logpdf(self, x: ArrayLike) -> Array:
        """Log probability density (or mass) function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log density.

        Returns
        -------
        Array
            Log probability density evaluated at ``x``.
        """
        ...

    @abstractmethod
    def rvs(
        self,
        rng: RngKey,
        shape: Tuple[int, ...] = (),
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Array:
        """Draw random variates from the distribution.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            Random key used for sampling.
        shape : tuple of int, optional
            Shape of the sample to draw. Default is ``()``.
        name : str, optional
            Optional site name used when tracing probabilistic programs.
        **kwargs : Any
            Additional distribution-specific sampling arguments.

        Returns
        -------
        Array
            Random variates of the requested shape.
        """
        ...

    def pdf(self, x: ArrayLike):
        """Probability density (or mass) function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the density.

        Returns
        -------
        Array
            Probability density evaluated at ``x``.
        """
        return jnp.exp(self.logpdf(x))

    def cdf(self, x: ArrayLike):
        """Cumulative distribution function, ``P(X <= x)``.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the CDF.

        Returns
        -------
        Array
            Cumulative probability evaluated at ``x``.
        """
        raise NotImplementedError("CDF not implemented for this distribution")

    def logcdf(self, x: ArrayLike) -> Array:
        """Log of the cumulative distribution function, ``log P(X <= x)``.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log CDF.

        Returns
        -------
        Array
            Log cumulative probability evaluated at ``x``.
        """
        return jnp.log(self.cdf(x))

    def ppf(self, q: ArrayLike) -> ArrayLike:
        """Percent point function (inverse of the CDF).

        Parameters
        ----------
        q : array_like
            Probabilities at which to evaluate the inverse CDF.

        Returns
        -------
        Array
            Inverse CDF values.
        """
        raise NotImplementedError("PPF not implemented for this distribution")

    def sf(self, x: ArrayLike) -> Array:
        """Survival function, ``1 - cdf(x)``.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the survival function.

        Returns
        -------
        Array
            Survival probability evaluated at ``x``.
        """
        return 1.0 - self.cdf(x)

    def logsf(self, x: ArrayLike) -> Array:
        """Log of the survival function, ``log(1 - cdf(x))``.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log survival function.

        Returns
        -------
        Array
            Log survival probability evaluated at ``x``.
        """
        return jnp.log(self.sf(x))

    def isf(self, q: ArrayLike) -> ArrayLike:
        """Inverse survival function (inverse of ``sf``).

        Parameters
        ----------
        q : array_like
            Probabilities at which to evaluate the inverse survival function.

        Returns
        -------
        Array
            Inverse survival function values.
        """
        return self.ppf(1.0 - q)

    def mean(self):
        """Mean of the distribution.

        Returns
        -------
        ArrayLike
            Distribution mean.
        """
        raise NotImplementedError("Mean not implemented for this distribution")

    def mode(self):
        """Mode of the distribution.

        Returns
        -------
        ArrayLike
            Distribution mode.
        """
        raise NotImplementedError("Mode not implemented for this distribution")

    def var(self):
        """Variance of the distribution.

        Returns
        -------
        ArrayLike
            Distribution variance.
        """
        raise NotImplementedError("Variance not implemented for this distribution")

    def std(self):
        """Standard deviation of the distribution.

        Returns
        -------
        ArrayLike
            Distribution standard deviation.
        """
        return jnp.sqrt(self.var())

    def entropy(self):
        """Differential entropy of the distribution.

        Returns
        -------
        ArrayLike
            Differential entropy.
        """
        raise NotImplementedError("Entropy not implemented for this distribution")

    def median(self):
        """Median of the distribution.

        Returns
        -------
        ArrayLike
            Distribution median.
        """
        return self.ppf(0.5)

    def interval(self, confidence: Optional[ArrayLike] = None):
        """Confidence interval with equal areas around the median.

        Parameters
        ----------
        confidence : array_like, optional
            Confidence level for the interval. Default is 0.95.

        Returns
        -------
        tuple
            Lower and upper bounds of the confidence interval.
        """
        confidence = 0.95 if confidence is None else confidence
        alpha = (1.0 - confidence) / 2.0
        return self.ppf(alpha), self.ppf(1.0 - alpha)

    def moment(self, order: Optional[int] = None):
        """Raw (non-central) moment of the distribution.

        Parameters
        ----------
        order : int, optional
            Order of the moment to compute.

        Returns
        -------
        ArrayLike
            Raw moment of the requested order.
        """
        raise NotImplementedError("Moment not implemented for this distribution")

    def skew(self):
        """Skewness of the distribution.

        Returns
        -------
        ArrayLike
            Distribution skewness.
        """
        raise NotImplementedError("Skew not implemented for this distribution")

    def kurtosis(self):
        """Excess kurtosis of the distribution.

        Returns
        -------
        ArrayLike
            Distribution excess kurtosis.
        """
        raise NotImplementedError("Kurtosis not implemented for this distribution")

    def stats(self, moments: str = "mv"):
        """Compute several summary statistics at once.

        Parameters
        ----------
        moments : str, optional
            String selecting which moments to compute. Each character selects
            one moment: ``"m"`` (mean), ``"v"`` (variance), ``"s"`` (skew),
            ``"k"`` (kurtosis). Default is ``"mv"``.

        Returns
        -------
        tuple
            Requested moment values in the order they appear in ``moments``.
        """
        values = []
        for m in moments:
            if m == "m":
                values.append(self.mean())
            elif m == "v":
                values.append(self.var())
            elif m == "s":
                values.append(self.skew())
            elif m == "k":
                values.append(self.kurtosis())
            else:
                raise ValueError(
                    "moments must contain only the letters 'm', 'v', 's', and 'k'."
                )
        if not values:
            return ()
        return values[0] if len(values) == 1 else tuple(values)

    def sample(self, rng: RngKey, shape: Tuple[int, ...] = ()) -> Array:
        """Draw random samples from the distribution.

        Alias of :meth:`rvs` that omits the optional ``name`` argument.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            Random key used for sampling.
        shape : tuple of int, optional
            Shape of the sample to draw. Default is ``()``.

        Returns
        -------
        Array
            Random samples of the requested shape.
        """
        return self.rvs(rng, shape=shape)

    def support(self):
        """Support (domain) of the distribution.

        Returns
        -------
        Constraint
            Constraint describing valid values of the random variable.
        """
        raise NotImplementedError("Support not implemented for this distribution")


class FrozenDistributionMeta(ABCMeta):
    """Metaclass for frozen distributions that inherits name and docstrings."""

    def __new__(mcs, name, bases, namespace, dist=None, **kwargs):
        if dist is not None:
            impl_class = dist if isinstance(dist, type) else dist.__class__
            dist_name = getattr(dist, "name", None) or getattr(impl_class, "name", None)
            if dist_name is not None:
                namespace.setdefault("name", dist_name)
            namespace.setdefault("dist", dist)

        return super().__new__(mcs, name, bases, namespace, **kwargs)


class rv_frozen(DistributionAPI, metaclass=FrozenDistributionMeta):
    """Frozen distribution with fixed parameter values.

    Wraps an unfrozen :class:`rv_generic` distribution together with concrete
    parameter values, exposing the SciPy-like :class:`DistributionAPI`. Create
    instances by calling an unfrozen distribution with its parameters rather
    than by constructing this class directly.

    Parameters
    ----------
    dist : rv_generic
        The unfrozen distribution instance to freeze.
    *args : array_like
        Positional parameter values for the distribution.
    **kwds : dict, optional
        Keyword parameter values for the distribution.
    """

    def __init__(
        self,
        dist,
        *args,
        **kwds,
    ):
        frozen_args = rv_generic._bind_frozen_args_for_dist(dist, tuple(args), kwds)
        self.args = frozen_args.args
        self.kwds = frozen_args.kwds
        self.dist = dist

        self._parameter_values = dict(frozen_args.parameter_values)
        self._parameter_aliases = dict(getattr(self.dist, "parameter_aliases", {}))
        self._call_kwds = self._build_call_kwargs()

        self._batch_shape, self._event_shape = self._compute_batch_and_event_shape(
            *self.args, **self.kwds
        )

        for param_name, value in self._parameter_values.items():
            if hasattr(type(self), param_name):
                continue
            object.__setattr__(self, param_name, value)

        super().__init__()

    def _build_call_kwargs(self) -> dict[str, Any]:
        """Build canonical kwargs for calling distribution methods."""
        call_kwds = dict(self.kwds)
        parameters = tuple(getattr(self.dist, "parameters", {}).keys())
        for idx, name in enumerate(parameters):
            if idx < len(self.args) and name not in call_kwds:
                call_kwds[name] = self.args[idx]
        return call_kwds

    def _call_dist(self, method_name: str, *args: Any, **kwds: Any) -> Any:
        """Call a distribution method using canonical frozen parameters."""
        method = getattr(self.dist, method_name)
        call_kwds = dict(self._call_kwds)
        call_kwds.update(kwds)
        return method(*args, **call_kwds)

    def _bind_rvs(
        self,
        rng: RngKey,
        shape: Tuple[int, ...],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Array:
        call_kwds = dict(self._call_kwds)
        call_kwds.update(kwargs)

        rvs_fn = getattr(self.dist, "_rvs_impl", None)
        if rvs_fn is None:
            rvs_fn = self.dist.rvs

        logpdf_fn = getattr(self.dist, "logpdf", None)
        if logpdf_fn is None:
            logpdf_fn = getattr(type(self.dist), "logpdf", None)

        return _rv_bind(rng, (), shape, self.dist, name, rvs_fn, logpdf_fn, call_kwds)

    def _compute_batch_and_event_shape(
        self, *args: Any, **kwds: Any
    ) -> tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Compute the batch and event shape of the distribution."""
        # Get shapes from the distribution arguments
        batch_shapes = []

        num_params = len(self.dist.parameters)

        # Check args for shapes
        for arg in args:
            if is_array_like(arg):
                batch_shapes.append(arg.shape)

        # Check kwargs for shapes
        for v in kwds.values():
            if is_array_like(v):
                batch_shapes.append(v.shape)

        if len(batch_shapes) > num_params:
            raise ValueError(
                "Too many args/kwargs provided for distribution "
                f"{self.dist.__class__.__name__}."
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

    @property
    def params(self) -> dict[str, Any]:
        """Name-keyed constrained parameters."""
        return dict(self._parameter_values)

    @property
    def unconstrained_params(self) -> dict[str, Any]:
        """Name-keyed parameters mapped through the constraint registry."""
        return self.dist.params_to_unconstrained(self.params)

    def __getattr__(self, name: str) -> Any:
        """Expose frozen parameters as attributes for SciPy-like ergonomics."""
        if name in self._parameter_values:
            return self._parameter_values[name]
        alias = self._parameter_aliases.get(name)
        if alias is not None and alias in self._parameter_values:
            return self._parameter_values[alias]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def _format_arg(self, arg):
        """Format argument for string representation, showing shapes for arrays."""
        # Check if arg is a JAX array
        if is_array_like(arg):
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
            Probability at which to evaluate the inverse cumulative distribution
            function.

        Returns
        -------
        ppf : ndarray or scalar
            Percent point function evaluated at q
        """
        return self._call_dist("ppf", q)

    def isf(self, q: ArrayLike):
        """Inverse survival function (1 - ppf) of the frozen distribution."""
        return self._call_dist("isf", q)

    def rvs(
        self,
        rng: RngKey,
        shape: Tuple[int, ...] = (),
        name: Optional[str] = None,
        **kwargs,
    ):
        """Random variates of the frozen distribution.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            The random key used for sampling
        shape : tuple of ints, optional
            The shape of the samples to draw. Default is ().
        name : str, optional
            Optional site name used when tracing probabilistic programs.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        return self._bind_rvs(rng, shape=shape, name=name, **kwargs)

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
        -------
        interval : tuple of ndarray or scalar
            Lower and upper bounds of the confidence interval.
        """
        return self._call_dist("interval", confidence)

    def support(self):
        """Support of the frozen distribution."""
        return self._call_dist("support")


class rv_discrete_frozen(rv_frozen):
    """Frozen discrete distribution with pmf/logpmf support."""

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
    """Frozen continuous distribution with pdf/logpdf support."""

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

    def rvs(
        self,
        rng: RngKey,
        *args,
        shape: Tuple[int, ...] = (),
        name: Optional[str] = None,
        **kwargs,
    ):
        """Random variates of the distribution.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            The random key used for sampling
        shape : tuple of ints, optional
            The shape of the samples to draw. Default is ().
        name : str, optional
            Optional site name used when tracing probabilistic programs.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given shape
        """
        return self._bind_rvs(rng, shape=shape, name=name, **kwargs)


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

_RV_FROZEN_TYPES = (rv_frozen,)
