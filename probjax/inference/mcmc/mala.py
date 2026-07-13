from typing import Callable, NamedTuple

import blackjax

from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel
from probjax.inference.mcmc.hmc import _scale_step_size_by_grad
from probjax.utils.typing import PyTree


class MALAParams(NamedTuple):
    step_size: float


def build_step(logdensity_fn: Callable) -> Callable:
    return make_step_from_kernel(
        logdensity_fn,
        blackjax.mala.build_kernel,
    )


def init_params(state: PyTree, step_size: float = 1e-2) -> MALAParams:
    step_size = _scale_step_size_by_grad(state, step_size)
    return MALAParams(step_size=step_size)


mala = make_kernel_api(
    name="mala",
    init_fn=blackjax.mala.init,
    init_params_fn=init_params,
    build_step_fn=build_step,
)
