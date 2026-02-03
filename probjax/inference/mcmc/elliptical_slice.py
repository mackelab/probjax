from typing import Callable, NamedTuple, Optional, Tuple

import blackjax
from blackjax.mcmc.elliptical_slice import EllipSliceInfo, EllipSliceState
from probjax.utils.typing import Array, PyTree, RngKey

from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel


class EllipticalSliceParams(NamedTuple):
    pass


def init_params(state: PyTree) -> EllipticalSliceParams:
    return EllipticalSliceParams()


def build_eliptical_slice_step(
    logdensity_fn: Callable,
    *,
    cov_matrix: Array,
    mean: Array,
) -> Callable:
    kernel_builder = lambda: blackjax.elliptical_slice.build_kernel(cov_matrix, mean)
    return make_step_from_kernel(logdensity_fn, kernel_builder)


elliptical_slice = make_kernel_api(
    name="elliptical_slice",
    init_fn=blackjax.elliptical_slice.init,
    init_params_fn=init_params,
    build_step_fn=build_eliptical_slice_step,
)
