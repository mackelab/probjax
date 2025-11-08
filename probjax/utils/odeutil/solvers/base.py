from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple, Optional

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
