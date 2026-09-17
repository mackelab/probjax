from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

from jax.typing import ArrayLike

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
    def init(position, **kwargs) -> SDEState:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def build_step(*args, **kwargs) -> Callable:
        raise NotImplementedError("build_step method must be implemented")

    def __new__(cls, drift: Callable, diffusion: Callable, **kwargs) -> SDESolver:
        return SDESolver(
            init=partial(cls.init, drift=drift, diffusion=diffusion),
            step=cls.build_step(drift, diffusion, **kwargs),
        )


def register_method(name: str, step_fn: SDESolverAPI, info: Optional[dict] = None):  # noqa: F821
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
    METHOD_STEP_FN[name] = step_fn
    METHOD_INFO[name] = info
    return step_fn


def get_method(name: str):
    return METHOD_STEP_FN[name], METHOD_INFO[name]


def get_methods():
    return list(METHOD_STEP_FN.keys())


def infer_noise_dim(g0, default_dim: int, configured_dim: int | None = None) -> int:
    """Infer Brownian dimension from diagonal or dense diffusion coefficients."""
    if configured_dim is not None:
        return int(configured_dim)
    return int(default_dim if g0.ndim <= 1 else g0.shape[-1])
