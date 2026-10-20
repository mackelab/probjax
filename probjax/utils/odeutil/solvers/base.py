from __future__ import annotations

import inspect
from functools import partial
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp

from probjax.utils._solver_common import ConfiguredSolver, resolve_solver
from probjax.utils.jaxutils import KernelAPI
from probjax.utils.typing import Array, ArrayLike, Callable

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


class ODESolverAPI(KernelAPI):
    """Base class for ODE solvers.

    Subclass it and set ``init`` and ``build_step`` as static methods.
    ``cls(drift, **options)`` builds the solver; ``cls(**options)`` returns a
    :class:`~probjax.utils.jaxutils.Configured` solver to pass as
    ``odeint(..., solver=...)``.
    """

    _kernel_type = ODESolver

    @staticmethod
    def init(*args: Any, **kwargs: Any) -> Any:
        """Initialize ODE solver state.

        Args:
            position: Initial position/state
            **kwargs: Additional keyword arguments (e.g., drift function)

        Returns:
            Initial ODEState
        """
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def init_params(*args: Any, **kwargs: Any) -> Any:
        """ODE solvers have no tunable parameters."""
        return None

    @staticmethod
    def build_step(*args: Any, **kwargs: Any) -> Any:
        """Build the step function for the ODE solver.

        Returns:
            Step function that takes (state, dt, *args) and returns
            (new_state, info)
        """
        raise NotImplementedError("build_step method must be implemented")

    info: dict = {}  # set by register_method

    def __new__(
        cls, drift: Optional[Callable[..., Any]] = None, dtype: Any = None, **kwargs
    ):
        """Configure or build the solver.

        ``cls(**options)`` returns a :class:`ConfiguredSolver` to pass as
        ``odeint(..., solver=...)``, e.g. ``bdf2(newton_iters=3)``.
        ``cls(drift, **options)`` builds the :class:`ODESolver` for ``drift``.

        Args:
            drift: Drift function f(t, y, *args) -> dy/dt
            dtype: Computation dtype. Forwarded to ``build_step`` for methods
                that build coefficient tableaux in a given dtype (otherwise
                they would stay in float32 and cap the attainable accuracy).
            **kwargs: Options for build_step

        Returns:
            ConfiguredSolver without ``drift``, else ODESolver with init and
            step functions
        """
        if drift is None:
            if dtype is not None:
                kwargs["dtype"] = dtype
            return ConfiguredSolver(cls, tuple(sorted(kwargs.items())))
        if (
            dtype is not None
            and "dtype" in inspect.signature(cls.build_step).parameters
        ):
            kwargs["dtype"] = dtype
        return cls.kernel_type(
            init=partial(cls.init, drift=drift),
            step=cls.build_step(drift, **kwargs),
        )


def register_method(
    name: str, step_fn: type[ODESolverAPI], info: Optional[dict] = None
) -> type[ODESolverAPI]:
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
    step_fn.info = info
    return step_fn


def get_method(solver) -> tuple[Any, dict]:
    """``(solver_factory, info)`` for a name, solver class or configured solver."""
    return resolve_solver(solver, METHOD_STEP_FN, METHOD_INFO)


def get_methods():
    return list(METHOD_STEP_FN.keys())


def rk_combine(
    dt: ArrayLike,
    y0: Array,
    k: Array,
    b_sol: Any,
    b_error: Any,
    b_mid: Any,
) -> tuple[Array, Array | None, Array | None]:
    """Shared explicit/implicit Runge-Kutta tail.

    Returns ``(y1, y1_error, y1_mid)`` from the stage matrix ``k``; callers
    reshape and attach solver-specific state/info.
    """
    y1 = jnp.asarray(y0 + dt * jnp.dot(b_sol, k))
    y1_error = None if b_error is None else jnp.asarray(dt * jnp.dot(b_error, k))
    y1_mid = None if b_mid is None else jnp.asarray(dt * jnp.dot(b_mid, k) + y0)
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

        def init_with_drift(t0, y0, *args, drift=None, **kwargs):
            return init_fn(t0, y0, *args, drift=drift)

        return init_with_drift

    def init_without_drift(t0, y0, *args, **kwargs):
        return init_fn(t0, y0, *args)

    return init_without_drift
