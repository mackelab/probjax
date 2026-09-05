from typing import Callable, NamedTuple, Optional

import blackjax
import jax
import jax.numpy as jnp

from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel
from probjax.utils.typing import Array, PyTree


class EllipticalSliceParams(NamedTuple):
    pass


def init_params(state: PyTree) -> EllipticalSliceParams:
    return EllipticalSliceParams()


def build_eliptical_slice_step(
    logdensity_fn: Callable,
    *,
    cov_matrix: Array,
    mean: Array,
    loglikelihood_fn: Optional[Callable] = None,
) -> Callable:
    if loglikelihood_fn is None:
        loglikelihood_fn = _make_loglikelihood(logdensity_fn, cov_matrix, mean)
    kernel_builder = lambda: blackjax.elliptical_slice.build_kernel(cov_matrix, mean)
    return make_step_from_kernel(loglikelihood_fn, kernel_builder)


def _make_loglikelihood(
    logdensity_fn: Callable, cov_matrix: Array, mean: Array
) -> Callable:
    ndim = jnp.ndim(cov_matrix)

    if ndim == 1:
        log_std = 0.5 * jnp.log(cov_matrix)

        def loglikelihood_fn(x):
            x_flat = jnp.ravel(x)
            return logdensity_fn(x) - jnp.sum(
                jax.scipy.stats.norm.logpdf(x_flat, mean, jnp.exp(log_std))
            )

    elif ndim == 2:

        def loglikelihood_fn(x):
            x_flat = jnp.ravel(x)
            return logdensity_fn(x) - jax.scipy.stats.multivariate_normal.logpdf(
                x_flat, mean, cov_matrix
            )

    else:
        raise ValueError(
            f"cov_matrix must be 1D (diagonal) or 2D (full), got ndim={ndim}"
        )

    return loglikelihood_fn


elliptical_slice = make_kernel_api(
    name="elliptical_slice",
    init_fn=blackjax.elliptical_slice.init,
    init_params_fn=init_params,
    build_step_fn=build_eliptical_slice_step,
)
