from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

from jax.typing import ArrayLike

from probjax.utils._solver_common import ConfiguredSolver, resolve_solver
from probjax.utils.jaxutils import API

METHOD_STEP_FN = {}
METHOD_INFO = {}


SDEState = NamedTuple
SDEInfo = NamedTuple


class SDESolver(NamedTuple):
    """This is a NamedTuple that represents a ODE kernel.
    given by the drift.
    """

    init: Callable
    step: Callable

    def __call__(self, key, state: SDEState, dt: ArrayLike) -> Tuple[SDEState, SDEInfo]:
        return self.step(key, state, dt)


class SDESolverAPI(metaclass=API):
    @staticmethod
    def init(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def build_step(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("build_step method must be implemented")

    info: dict = {}  # set by register_method

    def __new__(
        cls,
        drift: Optional[Callable] = None,
        diffusion: Optional[Callable] = None,
        **kwargs,
    ):
        """``cls(**options)`` configures the solver for ``sdeint(..., solver=...)``;
        ``cls(drift, diffusion, **options)`` builds it."""
        if drift is None:
            return ConfiguredSolver(cls, tuple(sorted(kwargs.items())))
        return SDESolver(
            init=partial(cls.init, drift=drift, diffusion=diffusion),
            step=cls.build_step(drift, diffusion, **kwargs),
        )


def register_method(
    name: str, step_fn: type[SDESolverAPI], info: Optional[dict] = None
) -> type[SDESolverAPI]:
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


def get_method(solver):
    """``(solver_factory, info)`` for a name, solver class or configured solver."""
    return resolve_solver(solver, METHOD_STEP_FN, METHOD_INFO)


def get_methods():
    return list(METHOD_STEP_FN.keys())


def infer_noise_dim(g0, default_dim: int, configured_dim: int | None = None) -> int:
    """Infer Brownian dimension from diagonal or dense diffusion coefficients."""
    if configured_dim is not None:
        return int(configured_dim)
    return int(default_dim if g0.ndim <= 1 else g0.shape[-1])
