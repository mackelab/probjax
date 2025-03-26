"""Adaptive ODE solver with modular adapter and parameter management."""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Union, Dict, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from probjax.utils.odeutil.solvers.base import ODESolver, ODEState
from probjax.utils.odeutil.integrate_adaptive import odeint_adaptive
from probjax.utils.odeutil.params import AdaptiveParams
from probjax.utils.odeutil.util import (
    initial_step_size,
    interp_fit,
    mean_error_ratio,
    optimal_step_size,
)


class AdaptationState(NamedTuple):
    """State of the adaptive step size control."""

    dt: float
    accepted: bool
    error_ratio: float
    num_steps: int


class StepSizeAdapter:
    """Adapter for adaptive step size control in ODE solvers."""

    def __init__(self, params: Optional[AdaptiveParams] = None):
        """Initialize the step size adapter.

        Args:
            params: Parameters for adaptive step size control
        """
        self.params = AdaptiveParams.create(params)

    def init_state(
        self, drift: Callable, y0: Array, t0: float = 0.0
    ) -> AdaptationState:
        """Initialize the adaptation state.

        Args:
            drift: ODE drift function
            y0: Initial state
            t0: Initial time

        Returns:
            Initial adaptation state
        """
        # Calculate initial step size if not provided
        if self.params.dtinit is None:
            f0 = drift(t0, y0)
            dt = jnp.clip(
                initial_step_size(
                    drift,
                    (),
                    t0,
                    y0,
                    f0,
                    self.params.order,
                    self.params.rtol,
                    self.params.atol,
                ),
                min=0.0,
                max=jnp.inf,
            )
        else:
            dt = self.params.dtinit

        # Clip step size to allowed range
        dt = jnp.clip(dt, self.params.dtmin, self.params.dtmax)

        return AdaptationState(dt=dt, accepted=True, error_ratio=0.0, num_steps=0)

    def step(
        self, state: AdaptationState, y: Array, y_new: Array, y_error: Array
    ) -> AdaptationState:
        """Compute the next step size based on the error estimate.

        Args:
            state: Current adaptation state
            y: Current state
            y_new: New state
            y_error: Error estimate

        Returns:
            Updated adaptation state
        """
        error_ratio = mean_error_ratio(
            y_error,
            y,
            y_new,
            self.params.rtol,
            self.params.atol,
            self.params.error_norm,
        )

        # Calculate new step size using controller
        dt_new = jnp.clip(
            optimal_step_size(
                state.dt,
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

        # Step is accepted if error is small enough or if we're at the smallest step size
        accepted = (error_ratio <= self.params.maxerror) | (
            state.dt <= self.params.dtmin
        )

        # If step is rejected, try again with smaller step size but don't increment step counter
        dt = jnp.where(accepted, dt_new, dt_new)
        num_steps = state.num_steps + jnp.where(accepted, 1, 0)

        return AdaptationState(
            dt=dt, accepted=accepted, error_ratio=error_ratio, num_steps=num_steps
        )


def adaptive_integrate(
    method: Callable, drift: Callable, y0: Array, ts: ArrayLike, *args, **kwargs
) -> Array:
    """Integrate ODE using adaptive step sizes.

    Args:
        method: ODE solver method
        drift: ODE drift function
        y0: Initial state
        ts: Time points to evaluate the solution
        *args: Additional arguments to pass to the drift function
        **kwargs: Additional keyword arguments for adaptive parameters

    Returns:
        Solution evaluated at the requested time points
    """
    # Extract parameters or create default ones
    params = AdaptiveParams.create(kwargs.pop("adaptive_params", None))

    # Add parameters to kwargs for the integration function
    kwargs["adaptive_params"] = params

    # Call the adaptive integrator
    return odeint_adaptive(method, drift, kwargs, y0, jnp.asarray(ts), *args)
