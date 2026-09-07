from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp

from probjax.utils.jaxutils import API
from probjax.utils.typing import ArrayLike, Callable

METHOD_STEP_FN: dict[str, Any] = {}
METHOD_INFO: dict[str, dict] = {}


ODEState = NamedTuple
ODEInfo = NamedTuple


class ODESolver(NamedTuple):
    """This is a NamedTuple that represents a ODE kernel.
    given by the drift.
    """

    init: Callable
    step: Callable

    def __call__(self, state: ODEState, dt: ArrayLike) -> tuple[ODEState, ODEInfo]:
        return self.step(state, dt)


class ODESolverAPI(metaclass=API):
    @staticmethod
    def init(position: ArrayLike, **kwargs) -> ODEState:
        """Initialize ODE solver state.

        Args:
            position: Initial position/state
            **kwargs: Additional keyword arguments (e.g., drift function)

        Returns:
            Initial ODEState
        """
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def build_step(
        *args, **kwargs
    ) -> Callable[[ODEState, ArrayLike], tuple[ODEState, ODEInfo]]:
        """Build the step function for the ODE solver.

        Returns:
            Step function that takes (state, dt, *args) and returns
            (new_state, info)
        """
        raise NotImplementedError("build_step method must be implemented")

    def __new__(cls, drift: Callable[..., Any], **kwargs) -> ODESolver:
        """Create an ODESolver instance.

        Args:
            drift: Drift function f(t, y, *args) -> dy/dt
            **kwargs: Additional keyword arguments for build_step

        Returns:
            ODESolver instance with init and step functions
        """
        return ODESolver(
            init=partial(cls.init, drift=drift),
            step=cls.build_step(drift, **kwargs),
        )


def register_method(name: str, step_fn: ODESolverAPI, info: Optional[dict] = None):  # noqa: F821
    """General method to register a step_fn for an ODE solver, thereby creating a
    new method.

    Args:
        name (str): Name of the method
        step_fn (Callable): Step function.
        info (Optional[dict], optional): Some information about your method.
            Defaults to None.

    Returns:
        _type_: _description_
    """
    if info is None:
        info = {}
    METHOD_STEP_FN[name] = step_fn
    METHOD_INFO[name] = info
    return step_fn


def get_method(name: str) -> tuple[ODESolverAPI, dict]:
    return METHOD_STEP_FN[name], METHOD_INFO[name]


def get_methods():
    return list(METHOD_STEP_FN.keys())


def rk_combine(dt, y0, k, b_sol, b_error, b_mid):
    """Shared explicit/implicit Runge-Kutta tail.

    Returns ``(y1, y1_error, y1_mid)`` from the stage matrix ``k``; callers
    reshape and attach solver-specific state/info.
    """
    y1 = y0 + dt * jnp.dot(b_sol, k)
    y1_error = None if b_error is None else dt * jnp.dot(b_error, k)
    y1_mid = None if b_mid is None else dt * jnp.dot(b_mid, k) + y0
    return y1, y1_error, y1_mid


def make_cached_init(state_cls):
    """Init caching ``f0`` for solver states carrying ``(t0, y0, f0)``."""

    def init(t0, y0, *args, drift=None):
        t0 = jnp.asarray(t0)
        y0 = jnp.asarray(y0)
        f0 = drift(t0, y0, *args) if drift is not None else None
        return state_cls(t0=t0, y0=y0, f0=f0)

    return init


def make_method_init(last_equals_next, init_fn, *, always_bind_drift=False):
    """Method-level init dispatching on FSAL (``last_equals_next``).

    With ``always_bind_drift`` (implicit methods), ``drift`` is forwarded
    even for non-FSAL methods, matching the historical implicit behavior
    where ``f0`` is always evaluated at init.
    """

    if last_equals_next or always_bind_drift:

        def init_method(t0, y0, *args, drift=None, **kwargs):
            return init_fn(t0, y0, *args, drift=drift)

    else:

        def init_method(t0, y0, *args, **kwargs):
            return init_fn(t0, y0, *args)

    return init_method
