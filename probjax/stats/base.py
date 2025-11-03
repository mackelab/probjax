"""
Statistical Distributions Base Classes (:mod:`probjax.stats.base`)
=================================================================

This module contains the base classes for continuous and discrete random variables
that provide a SciPy-like API. This closely follows the structure of scipy.stats._distn_infrastructure.
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from probjax.stats.constraints import Constraint

__all__ = [
    "rv_generic",
    "rv_continuous",
    "rv_discrete",
    "rv_continuous_frozen",
    "rv_discrete_frozen",
]

_ArrayDict = Dict[str, Array]
_FlatMetadata = List[Tuple[str, Tuple[int, ...]]]


def _as_array(value: ArrayLike, dtype: Optional[jnp.dtype] = None) -> Array:
    """Convert an input value to a JAX array with optional dtype coercion."""
    arr = jnp.asarray(value)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def _flatten_param_dict(param_dict: Mapping[str, ArrayLike]) -> Tuple[Array, _FlatMetadata]:
    """Flatten a dictionary of parameter arrays into a single vector."""
    flat_pieces: List[Array] = []
    metadata: _FlatMetadata = []

    for name, value in param_dict.items():
        arr = _as_array(value)
        metadata.append((name, arr.shape))
        flat_pieces.append(arr.reshape(-1))

    if not flat_pieces:
        return jnp.array([], dtype=jnp.float32), metadata

    if len(flat_pieces) == 1:
        flat = flat_pieces[0]
    else:
        flat = jnp.concatenate(flat_pieces)

    return flat, metadata


def _unflatten_param_vector(
    vector: Array, metadata: _FlatMetadata
) -> Dict[str, Array]:
    """Reconstruct a parameter dictionary from a flattened representation."""
    params: Dict[str, Array] = {}
    cursor = 0
    for name, shape in metadata:
        size = int(np.prod(shape)) if shape else 1
        slice_ = vector[cursor : cursor + size]
        params[name] = slice_.reshape(shape) if shape else jnp.asarray(slice_)
        cursor += size
    return params


def _default_initial_guess(name: str, data: Array) -> Array:
    """Heuristic initial guesses for parameter optimisation."""
    dtype = data.dtype
    if name in {"loc", "mean", "mu"}:
        return jnp.mean(data, axis=0, dtype=dtype)
    if name in {"scale", "std", "sigma"}:
        return jnp.maximum(jnp.std(data, axis=0, dtype=dtype), jnp.asarray(1e-6, dtype))
    if name in {"variance"}:
        return jnp.maximum(jnp.var(data, axis=0, dtype=dtype), jnp.asarray(1e-6, dtype))
    if name in {"p", "prob"}:
        clipped = jnp.clip(jnp.mean(data, axis=0, dtype=dtype), 1e-6, 1 - 1e-6)
        return clipped
    if name in {"rate", "lambda"}:
        return jnp.reciprocal(jnp.maximum(jnp.mean(data, axis=0, dtype=dtype), jnp.asarray(1e-6, dtype)))
    if name in {"alpha", "beta", "kappa", "shape", "concentration"}:
        return jnp.asarray(1.0, dtype)
    return jnp.asarray(1.0, dtype)


def _normalize_sample_weights(
    num_samples: int,
    weights: Optional[ArrayLike],
    dtype: jnp.dtype,
) -> Optional[Array]:
    """Normalize sample weights ensuring positivity and unit sum."""
    if weights is None:
        return None
    w = jnp.asarray(weights, dtype=dtype)
    if w.ndim != 1 or w.shape[0] != num_samples:
        raise ValueError(
            f"weights must be a one-dimensional array of length {num_samples}, got shape {w.shape}."
        )
    w = jnp.clip(w, a_min=jnp.asarray(0.0, dtype=dtype))
    total = jnp.sum(w)
    fallback = jnp.asarray(num_samples, dtype=dtype)
    total = jnp.where(total > 0, total, fallback)
    return w / total


def _prepare_initial_parameters(
    cls: "rv_generic",
    data: Array,
    overrides: Optional[Mapping[str, ArrayLike]] = None,
    fixed: Optional[Mapping[str, ArrayLike]] = None,
) -> Dict[str, Array]:
    overrides = overrides or {}
    fixed = fixed or {}
    dtype = data.dtype
    guessed: Dict[str, Array] = {}

    # Allow subclass hook
    subclass_guess: Dict[str, ArrayLike] = {}
    subclass_guess_raw = getattr(cls, "_fit_initial_guess", None)
    if callable(subclass_guess_raw):
        maybe_guess = subclass_guess_raw(data)
        if maybe_guess is not None:
            subclass_guess = dict(maybe_guess)
    for name in cls.parameters:
        if name in fixed:
            guessed[name] = _as_array(fixed[name], dtype)
        elif name in overrides:
            guessed[name] = _as_array(overrides[name], dtype)
        elif name in subclass_guess:
            guessed[name] = _as_array(subclass_guess[name], dtype)
        else:
            guessed[name] = _default_initial_guess(name, data)
    return guessed


class rv_generic(ABC):
    """Generic random variable class for common functionality."""

    name: Optional[str] = None
    parameters: Dict[str, Constraint] = {}

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
    def _fit_closed_form(
        cls,
        data: Array,
        *,
        weights: Optional[Array] = None,
        fixed: Optional[Mapping[str, ArrayLike]] = None,
        **kwargs: Any,
    ) -> Optional[Tuple[Array, ...]]:
        """Optional closed-form estimator hook. Subclasses may override."""
        return None

    @classmethod
    def _build_parameter_dict(
        cls,
        varying: Mapping[str, Array],
        fixed: Optional[Mapping[str, ArrayLike]] = None,
    ) -> Dict[str, Array]:
        params: Dict[str, Array] = {}
        fixed = fixed or {}
        for name in cls.parameters:
            if name in fixed:
                params[name] = _as_array(fixed[name])
            else:
                params[name] = varying[name]
        return params

    @classmethod
    def _fit_numeric(
        cls,
        data: Array,
        *,
        initial: Optional[Mapping[str, ArrayLike]] = None,
        fixed: Optional[Mapping[str, ArrayLike]] = None,
        optimizer: str = "BFGS",
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        weights: Optional[Array] = None,
        **logpdf_kwargs: Any,
    ) -> Tuple[Array, ...]:
        if not cls.parameters:
            raise ValueError(
                f"Distribution {cls.__name__} does not expose parameters to fit."
            )

        fixed = fixed or {}
        initial_params = _prepare_initial_parameters(
            cls,
            data,
            overrides=initial,
            fixed=fixed,
        )

        free_names = [name for name in cls.parameters if name not in fixed]
        varying_initial = {name: initial_params[name] for name in free_names}

        if not varying_initial:
            fitted = cls._build_parameter_dict({}, fixed=fixed)
            return tuple(fitted[name] for name in cls.parameters)

        flat_init, metadata = _flatten_param_dict(varying_initial)
        target_dtype = jnp.result_type(data.dtype, jnp.float32)
        flat_init = flat_init.astype(target_dtype)

        def objective(theta: Array) -> Array:
            varying = _unflatten_param_vector(theta, metadata)
            params = cls._build_parameter_dict(varying, fixed=fixed)
            log_prob = cls.logpdf(data, **params, **logpdf_kwargs)
            if weights is not None:
                log_prob = log_prob * weights
            return -jnp.sum(log_prob).astype(target_dtype)

        grad_fn = jax.grad(objective)
        optimizer_kwargs = dict(optimizer_kwargs or {})

        from jax.scipy.optimize import minimize

        result = minimize(
            objective,
            flat_init,
            method=optimizer,
            jac=grad_fn,
            **optimizer_kwargs,
        )
        if not getattr(result, "success", False):
            message = getattr(result, "message", "Unknown optimisation failure")
            raise ValueError(f"Optimization failed for {cls.__name__}: {message}")

        fitted_varying = _unflatten_param_vector(result.x, metadata)
        fitted = cls._build_parameter_dict(fitted_varying, fixed=fixed)
        ordered = tuple(fitted[name] for name in cls.parameters)
        return ordered

    @classmethod
    def fit(
        cls,
        data: ArrayLike,
        *,
        method: str = "auto",
        initial: Optional[Mapping[str, ArrayLike]] = None,
        fixed: Optional[Mapping[str, ArrayLike]] = None,
        optimizer: str = "BFGS",
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        weights: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Tuple[Array, ...]:
        """
        Estimate distribution parameters from data.

        Parameters
        ----------
        data:
            Observations drawn from the distribution.
        method:
            Either ``"auto"`` (default), ``"closed_form"`` to force closed-form
            estimation, or ``"numeric"`` to skip analytic attempts.
        initial:
            Optional mapping providing initial guesses for the optimiser.
        fixed:
            Optional mapping of parameter names to values that should be held
            constant during fitting.
        optimizer:
            Optimiser name passed to :func:`jax.scipy.optimize.minimize` when a
            numeric routine is required.
        optimizer_kwargs:
            Additional keyword arguments forwarded to the optimiser.
        weights:
            Optional non-negative sample weights. When provided, closed-form
            implementations may exploit them; otherwise a
            ``NotImplementedError`` is raised.
        **kwargs:
            Additional keyword arguments. If any share a name with distribution
            parameters they are interpreted as fixed parameters; remaining keys
            are merged into ``optimizer_kwargs``.
        """
        data_arr = _as_array(data)
        if data_arr.size == 0:
            raise ValueError("Cannot fit distribution with empty data.")
        if data_arr.ndim == 0:
            data_arr = jnp.reshape(data_arr, (1,))

        num_samples = int(data_arr.shape[0])
        weights_arr = _normalize_sample_weights(num_samples, weights, data_arr.dtype)

        fixed_map: Dict[str, ArrayLike] = dict(fixed or {})
        extra_opt_kwargs: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in cls.parameters:
                fixed_map[key] = value
            else:
                extra_opt_kwargs[key] = value

        if extra_opt_kwargs:
            base_opts = dict(optimizer_kwargs or {})
            base_opts.update(extra_opt_kwargs)
            optimizer_kwargs = base_opts

        if method not in {"auto", "closed_form", "numeric"}:
            raise ValueError(f"Unknown fit method '{method}'")

        if method in {"auto", "closed_form"}:
            closed = cls._fit_closed_form(
                data_arr,
                weights=weights_arr,
                fixed=fixed_map if fixed_map else None,
            )
            if closed is not None:
                return tuple(_as_array(param, data_arr.dtype) for param in closed)
            if method == "closed_form":
                raise NotImplementedError(
                    f"No closed-form fit available for {cls.__name__}."
                )

        return cls._fit_numeric(
            data_arr,
            initial=initial,
            fixed=fixed_map,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            weights=weights_arr,
        )


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
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwargs,
    ):
        """Default fit relies on :meth:`rv_generic.fit`."""
        return super().fit(data, weights=weights, **kwargs)


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
    def fit(
        cls,
        data: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        **kwds,
    ):
        """Delegate to :meth:`rv_generic.fit` for numerical/analytic estimation."""
        return super().fit(data, weights=weights, **kwds)


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
