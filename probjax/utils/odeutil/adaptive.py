"""Step-size adaptor for adaptive ODE solvers.

Subclass :class:`StepSizeAdaptor` to plug in a different controller (e.g. a
PI controller, dead-band, learned policy). The default reproduces the
classic Hairer–Wanner controller used by Dormand–Prince / Cash–Karp.
"""

import copy

import jax.numpy as jnp

from probjax.utils.odeutil.util import (
    initial_step_size as _initial_step_size,
    mean_error_ratio,
    optimal_step_size,
)


class StepSizeAdaptor:
    """Standardized step-size controller for adaptive ODE solvers.

    Constructor arguments are the classic adaptive-RK tuning knobs:

    Args:
        rtol: Relative tolerance (per-component).
        atol: Absolute tolerance (per-component).
        mxstep: Hard cap on inner adaptive iterations between two output
            grid points. Defaults to ``jnp.inf`` (no cap).
        dtmin: Minimum admissible step size; the controller clips below
            this and accepts the step regardless of error.
        dtmax: Maximum admissible step size.
        maxerror: Accept-step threshold on the normalized error ratio.
        safety, ifactor, dfactor: Hairer–Wanner safety factor and
            grow/shrink limits applied around the optimal step.
        error_norm: Order of the error norm (``2`` = RMS).
        order: Local-error order. Set automatically by the integrator to
            match the chosen method, so there is rarely a reason to set
            this at construction time.

    Subclass to customize the strategy:

        >>> class HairerOnly(StepSizeAdaptor):
        ...     def accept_step(self, error_ratio, dt):
        ...         return error_ratio <= self.maxerror

    Note:
        This object is only consulted by **adaptive** solver methods
        (``"dopri5"``, ``"dopri8"``, ``"bosh3"``, etc.). Fixed-step methods
        (``"euler"``, ``"rk4"``, ``"midpoint"``, ...) ignore it entirely.
    """

    def __init__(
        self,
        rtol: float = 1e-4,
        atol: float = 1e-5,
        mxstep: int = jnp.inf,
        dtmin: float = 0.0,
        dtmax: float = jnp.inf,
        maxerror: float = 1.2,
        safety: float = 0.95,
        ifactor: float = 10.0,
        dfactor: float = 0.1,
        error_norm: float = 2,
        order: int = 5,
    ) -> None:
        self.rtol = rtol
        self.atol = atol
        self.mxstep = mxstep
        self.dtmin = dtmin
        self.dtmax = dtmax
        self.maxerror = maxerror
        self.safety = safety
        self.ifactor = ifactor
        self.dfactor = dfactor
        self.error_norm = error_norm
        self.order = order

    def with_order(self, order: int) -> "StepSizeAdaptor":
        """Return a shallow copy with the local-error ``order`` set.

        The adaptive integrator calls this once after resolving the method's
        local order, so subclasses do not need to track ``order`` themselves.
        """
        new = copy.copy(self)
        new.order = order
        return new

    def initial_step_size(self, drift, args, t0, y0, f0):
        """Pick the initial step ``dt0`` from drift / state magnitudes."""
        return jnp.clip(
            _initial_step_size(
                drift, args, t0, y0, f0, self.order, self.rtol, self.atol
            ),
            min=0.0,
            max=jnp.inf,
        )

    def error_ratio(self, error, y, y_next):
        """Compute ``||error / (atol + rtol·max(|y|,|y_next|))||_p``."""
        return mean_error_ratio(
            error, y, y_next, self.rtol, self.atol, self.error_norm
        )

    def next_step_size(self, dt, error_ratio):
        """Hairer–Wanner optimal step, clipped to ``[dtmin, dtmax]``."""
        return jnp.clip(
            optimal_step_size(
                dt,
                error_ratio,
                maxerror=self.maxerror,
                safety=self.safety,
                ifactor=self.ifactor,
                dfactor=self.dfactor,
                order=self.order,
            ),
            self.dtmin,
            self.dtmax,
        )

    def accept_step(self, error_ratio, dt):
        """Accept iff the error is below ``maxerror`` or ``dt`` hit a clip."""
        return (
            (error_ratio <= self.maxerror)
            | (dt == self.dtmin)
            | (dt == self.dtmax)
        )
