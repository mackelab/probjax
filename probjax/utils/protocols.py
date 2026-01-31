from typing import Any, Optional, Protocol

import jax
from jaxtyping import Array

__all__ = [
    "ModelFn",
    "TimeDependentModelFn",
    "TimeDependentFn",
    "WeightFn",
    "ReductionFn",
    "LossFn",
    "InterpolationFn",
    "InterpolationNoiseFn",
]


class ModelFn(Protocol):
    """A callable that predicts output at a given position."""

    def __call__(
        self, *args: Any, rng: Optional[jax.random.PRNGKey] = None, **kwargs: Any
    ) -> Array:
        """Predict output at a given position.

        Args:
            *args: Position and any additional arguments
            rng: Optional random number generator key
            **kwargs: Additional keyword arguments

        Returns:
            Array of predicted outputs
        """
        ...


class TimeDependentModelFn(Protocol):
    """A callable that predicts a velocity field at a given position and time."""

    def __call__(
        self,
        t: Array,
        *args: Any,
        rng: Optional[jax.random.PRNGKey] = None,
        **kwargs: Any,
    ) -> Array:
        """Predict velocity field at a given position and time.

        Args:
            t: Time value
            *args: Position and any additional arguments
            rng: Optional random number generator key
            **kwargs: Additional keyword arguments

        Returns:
            Array of predicted velocities
        """
        ...


class TimeDependentFn(Protocol):
    """A callable that depends on time and input data."""

    def __call__(self, t: Array, *args: Any, **kwargs: Any) -> Array:
        """Compute time-dependent value.

        Args:
            t: Time value
            *args: Input data and any additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Array of computed values
        """
        ...


class WeightFn(Protocol):
    """A callable that computes weights based on time."""

    def __call__(self, t: Array) -> Array:
        """Compute weights based on time.

        Args:
            t: Time value

        Returns:
            Array of weights
        """
        ...


class ReductionFn(Protocol):
    """A callable that reduces an array to a scalar or smaller array."""

    def __call__(self, x: Array, *args, **kwargs: Any) -> Array:
        """Reduce array to scalar or smaller array.

        Args:
            x: Array to reduce
            **kwargs: Additional keyword arguments

        Returns:
            Reduced array
        """
        ...


class LossFn(Protocol):
    """A callable that computes a scalar loss value."""

    def __call__(
        self, *args: Any, rng: Optional[jax.random.PRNGKey] = None, **kwargs: Any
    ) -> Array:
        """Compute scalar loss value.

        Args:
            *args: Input data and any additional arguments
            rng: Optional random number generator key
            **kwargs: Additional keyword arguments

        Returns:
            Scalar loss value
        """
        ...


# Define additional protocols for flow matching


class InterpolationFn(Protocol):
    """A callable that interpolates between two points at a given time."""

    def __call__(self, t: Array, x0: Array, x1: Array) -> Array:
        """Interpolate between two points at a given time.

        Args:
            t: Time value
            x0: Starting point
            x1: Ending point

        Returns:
            Array of interpolated points
        """
        ...


class InterpolationNoiseFn(Protocol):
    """A callable that provides noise scale for interpolation."""

    def __call__(self, t: Array, x0: Array, x1: Array) -> Array:
        """Compute noise scale for interpolation.

        Args:
            t: Time value
            x0: Starting point
            x1: Ending point

        Returns:
            Array of noise scales
        """
        ...
