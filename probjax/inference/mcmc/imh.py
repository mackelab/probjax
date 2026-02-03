from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.random_walk import RWInfo, RWState
from probjax.utils.typing import Array, PyTree, RngKey

from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel


class IMHParams(NamedTuple):
    pass


def wrap_logpdf(logpdf: Callable) -> Callable:
    """Wrap the logpdf function to work with the IMH kernel."""

    def wrapped_logpdf(x: Any, y: Any, *args, **kwargs) -> Array:
        return logpdf(y, *args, **kwargs)

    return wrapped_logpdf


def build_imh_step(
    logdensity_fn: Callable,
    proposal_fn: Callable,
    proposal_logpdf: Callable,
) -> Callable:
    """A function to build the Independent Metropolis-Hastings kernel.

    Args:
        logdensity_fn (Callable): The log density function.
        proposal_fn (Callable): Proposal function.
        proposal_logpdf (Callable): Proposal log pdf.

    Returns:
        Callable: Step function for the IMH kernel.
    """
    kernel = blackjax.irmh.build_kernel()

    def step(
        key: RngKey, state: RWState, params: Optional[IMHParams] = None
    ) -> Tuple[RWState, RWInfo]:
        _proposal_fn = partial(proposal_fn, params=params)
        _proposal_logpdf = partial(wrap_logpdf(proposal_logpdf), params=params)
        return kernel(
            key,
            state,
            logdensity_fn,
            proposal_distribution=_proposal_fn,
            proposal_logdensity_fn=_proposal_logpdf,
        )

    return step


def init_imh_params(state: PyTree, rng_key=None) -> IMHParams:
    """Generally, there are no parameters to initialize for the IMH kernel."""
    return IMHParams()


imh = make_kernel_api(
    name="imh",
    init_fn=blackjax.irmh.init,
    init_params_fn=init_imh_params,
    build_step_fn=build_imh_step,
)


class GaussianIMHParams(NamedTuple):
    """Parameters for the Gaussian IMH kernel."""

    mean: Array
    cov: Array
    unflatten: Callable


def init_gaussian_imh_params(
    state: PyTree, mean: Optional[Array] = None, cov: Optional[Array] = None
) -> GaussianIMHParams:
    """Initialize the parameters for the Gaussian IMH kernel."""
    position = state.position if hasattr(state, "position") else state
    flat_position, unflatten = jax.flatten_util.ravel_pytree(position)
    if mean is None:
        mean = flat_position
    if cov is None:
        cov = jnp.ones_like(flat_position)
    return GaussianIMHParams(mean=mean, cov=cov, unflatten=unflatten)


def proposal_gaussian(key: RngKey, *, params: GaussianIMHParams):
    """Generate a new position from a Gaussian proposal."""
    mean = params.mean
    cov = params.cov
    eps = jax.random.normal(key, mean.shape)
    if cov.ndim == 1:
        new_position = mean + jnp.sqrt(cov) * eps
    else:
        new_position = mean + jnp.dot(jnp.linalg.cholesky(cov), eps)
    return params.unflatten(new_position)


def proposal_gaussian_logpdf(state, *, params: GaussianIMHParams):
    """Log pdf of the Gaussian proposal."""
    x = state.position
    flat_position, _ = jax.flatten_util.ravel_pytree(x)
    mean = params.mean
    cov = params.cov
    if cov.ndim == 1:
        return jax.scipy.stats.norm.logpdf(flat_position, mean, cov).sum()
    else:
        return jax.scipy.stats.multivariate_normal.logpdf(flat_position, mean, cov)


gaussian_imh = make_kernel_api(
    name="gaussian_imh",
    init_fn=blackjax.irmh.init,
    init_params_fn=init_gaussian_imh_params,
    build_step_fn=partial(
        build_imh_step,
        proposal_fn=proposal_gaussian,
        proposal_logpdf=proposal_gaussian_logpdf,
    ),
)
