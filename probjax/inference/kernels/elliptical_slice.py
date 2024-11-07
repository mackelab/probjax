from typing import Callable, NamedTuple, Optional, Tuple

import blackjax
from blackjax.mcmc.elliptical_slice import EllipSliceInfo, EllipSliceState
from chex import PRNGKey
from jaxtyping import Array, PyTree

from probjax.inference.kernels.base import MarkovKernelAPI


class EllipticalSliceParams(NamedTuple):
    pass


def init_params(position: PyTree) -> EllipticalSliceParams:
    return EllipticalSliceParams()


def build_eliptical_slice_step(
    logdensity_fn: Callable,
    *,
    cov_matrix: Array,
    mean: Array,
) -> Callable:
    kernel = blackjax.elliptical_slice.build_kernel(cov_matrix, mean)

    def step(
        key: PRNGKey,
        state: EllipSliceState,
        params: Optional[EllipticalSliceParams] = None,
    ) -> Tuple[EllipSliceState, EllipSliceInfo]:
        return kernel(key, state, logdensity_fn=logdensity_fn)

    return step


class EllipticalSlice(MarkovKernelAPI):
    init = blackjax.elliptical_slice.init
    build_step = build_eliptical_slice_step
    init_params = init_params
