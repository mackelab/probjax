from typing import NamedTuple, Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from probjax.utils.odeutil.util import mean_error_ratio, optimal_step_size


class AdaptiveParams(NamedTuple):
    """Parameters for adaptive ODE integration."""

    rtol: float = 1e-4
    atol: float = 1e-5
    mxstep: int = jnp.inf
    dtmin: float = 0.0
    dtmax: float = jnp.inf
    maxerror: float = 1.2
    safety: float = 0.95
    ifactor: float = 10.0
    dfactor: float = 0.1
    error_norm: float = 2
    order: int = 5


class StepSizeAdapter:
    """Handles step size adaptation for ODE solvers."""

    def __init__(self, params: AdaptiveParams):
        """Initialize the step size adapter with the given parameters.

        Args:
            params: Parameters for adaptive integration
        """
        self.params = params

    def error_ratio(self, error, y, y_next):
        """Compute the error ratio for step size control.

        Args:
            error: The estimated error
            y: The current state
            y_next: The next state

        Returns:
            The computed error ratio
        """
        return mean_error_ratio(
            error, y, y_next, self.params.rtol, self.params.atol, self.params.error_norm
        )

    def next_step_size(self, dt, error_ratio):
        """Compute the next step size based on the current error ratio.

        Args:
            dt: Current step size
            error_ratio: Current error ratio

        Returns:
            Next step size
        """
        return jnp.clip(
            optimal_step_size(
                dt,
                error_ratio,
                maxerror=self.params.maxerror,
                safety=self.params.safety,
                ifactor=self.params.ifactor,
                dfactor=self.params.dfactor,
                order=self.params.order,
            ),
            self.params.dtmin,
            self.params.dtmax,
        )

    def accept_step(self, error_ratio, dt):
        """Determine whether to accept the step based on the error ratio.

        Args:
            error_ratio: Current error ratio
            dt: Current step size

        Returns:
            Boolean indicating whether to accept the step
        """
        return (
            (error_ratio <= self.params.maxerror)
            | (dt == self.params.dtmin)
            | (dt == self.params.dtmax)
        )
