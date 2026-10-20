from __future__ import annotations

from typing import Callable, NamedTuple

import blackjax

from probjax.inference.mcmc.adaptation import step_size_adaptor
from probjax.inference.mcmc.base import (
    MarkovKernelAPI,
    make_step_from_kernel,
    scale_step_size_by_grad,
)
from probjax.utils.typing import PyTree, Scalar


class MALAParams(NamedTuple):
    step_size: Scalar


def build_step(logdensity_fn: Callable) -> Callable:
    return make_step_from_kernel(
        logdensity_fn,
        blackjax.mala.build_kernel,
    )


def init_params(state: PyTree, step_size: float = 1e-2) -> MALAParams:
    scaled_step_size = scale_step_size_by_grad(state, step_size)
    return MALAParams(step_size=scaled_step_size)


class mala(MarkovKernelAPI):
    """Metropolis-adjusted Langevin algorithm.

    Proposes a gradient step plus Gaussian noise and accepts it with a
    Metropolis-Hastings correction. Tunable parameters come from
    ``kernel.init_params(state, step_size=1e-2)`` (see :class:`MALAParams`);
    the step size is scaled by the initial gradient norm. Default adaptation:
    :func:`step_size_adaptor`.

    Args:
        logdensity_fn: Log density of the target.
    """

    init = staticmethod(blackjax.mala.init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_step)
    default_adaptation_fn = staticmethod(step_size_adaptor)
