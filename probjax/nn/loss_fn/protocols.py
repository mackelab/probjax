from typing import Protocol, Optional
from jaxtyping import Array
from jax.typing import ArrayLike

__all__ = [
    "ScoreModelFn",
    "TimeDependentFn",
    "WeightFn",
    "ReductionFn",
    "LossFn",
]


class ScoreModelFn(Protocol):
    """Protocol for score model functions."""

    def __call__(self, *args, **kwargs) -> Array:
        """
        A score function that computes the gradient of log-density.

        Returns:
            Score values (gradient of log-density)
        """
        ...


class TimeDependentFn(Protocol):
    """Protocol for time-dependent functions."""

    def __call__(self, times: Array, x: Array, *args, **kwargs) -> Array:
        """
        A function that depends on time and input data.

        Args:
            times: Time values
            x: Input data

        Returns:
            Output values
        """
        ...


class WeightFn(Protocol):
    """Protocol for weight functions."""

    def __call__(self, times: Array) -> Array:
        """
        A function that computes weights based on time.

        Args:
            times: Time values

        Returns:
            Weight values
        """
        ...


class ReductionFn(Protocol):
    """Protocol for reduction functions."""

    def __call__(self, x: Array) -> Array:
        """
        A function that reduces an array to a scalar or smaller array.

        Args:
            x: Input array

        Returns:
            Reduced value
        """
        ...


class LossFn(Protocol):
    """Protocol for loss functions."""

    def __call__(self, *args, rng: Optional[Array] = None, **kwargs) -> Array:
        """
        A loss function that computes a scalar loss value.

        Args:
            rng: Random number generator key

        Returns:
            Loss value
        """
        ...


# Define additional protocols for flow matching


class InterpolationFn(Protocol):
    """Protocol for interpolation functions."""

    def __call__(self, x0: Array, x1: Array, t: Array) -> Array:
        """
        A function that interpolates between x0 and x1 at time t.

        Args:
            x0: Starting point
            x1: Ending point
            t: Interpolation time(s)

        Returns:
            Interpolated value
        """
        ...


class InterpolationNoiseFn(Protocol):
    """Protocol for interpolation noise functions."""

    def __call__(self, x0: Array, x1: Array, t: Array) -> Array:
        """
        A function that provides noise scale for interpolation.

        Args:
            x0: Starting point
            x1: Ending point
            t: Interpolation time(s)

        Returns:
            Noise scale
        """
        ...


class VelocityModelFn(Protocol):
    """Protocol for velocity model functions."""

    def __call__(self, t: Array, x: Array, *args, **kwargs) -> Array:
        """
        A function that predicts the velocity field at point x and time t.

        Args:
            t: Time
            x: Position

        Returns:
            Velocity vector
        """
        ...
