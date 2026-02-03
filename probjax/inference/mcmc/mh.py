from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.random_walk import RWInfo, RWState
from jax.flatten_util import ravel_pytree
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey

from probjax.inference.mcmc.adaptation import (
    step_size_adaption,
    step_size_and_scale_adaption,
)
from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel


class RWParams(NamedTuple):
    pass


def build_mh_step(
    logdensity_fn: Callable,
    transition_proposal_fn: Callable,
    transition_proposal_logpdf: Optional[Callable] = None,
) -> Callable:
    kernel = blackjax.rmh.build_kernel()

    def step(
        key: RngKey,
        state: RWState,
        params: RWParams,
    ) -> Tuple[RWState, RWInfo]:
        _transition_proposal_fn = partial(transition_proposal_fn, params=params)
        if transition_proposal_logpdf is not None:
            _transition_proposal_logpdf = partial(
                transition_proposal_logpdf, params=params
            )
        else:
            _transition_proposal_logpdf = None

        return kernel(
            key,
            state,
            logdensity_fn,
            _transition_proposal_fn,
            _transition_proposal_logpdf,
        )

    return step


def init_params_mh(state: PyTree) -> RWParams:
    return RWParams()


mh = make_kernel_api(
    name="mh",
    init_fn=blackjax.rmh.init,
    init_params_fn=init_params_mh,
    build_step_fn=build_mh_step,
)


class RWParamsGauss(NamedTuple):
    step_size: float
    scale: Optional[ArrayLike]


def init_params_gaussian_rw(
    state: PyTree, step_size: float = 0.5, scale: Optional[Array] = None
) -> RWParamsGauss:
    position = state.position if hasattr(state, "position") else state
    flat_position, _ = ravel_pytree(position)
    dim = flat_position.shape[0]
    if scale is None:
        scale = jnp.ones((dim,))
    return RWParamsGauss(step_size=step_size, scale=scale)


def gaussian_transition_proposal(key, position, params: RWParamsGauss):
    flat_position, unflatten = ravel_pytree(position)
    eps = jax.random.normal(key, flat_position.shape)
    if params.scale is None:
        new_position = flat_position + params.step_size * eps
    elif len(params.scale.shape) == 1:
        new_position = flat_position + params.step_size * params.scale * eps
    elif len(params.scale.shape) == 2:
        new_position = flat_position + params.step_size * jnp.dot(params.scale, eps)
    else:
        raise ValueError("Invalid scale shape")

    return unflatten(new_position)


def build_adaptation(
    logdensity_fn: Callable,
) -> Callable:
    def fit_params(
        key: RngKey,
        state: PyTree,
        params: RWParamsGauss,
        num_steps: int = 100,
        method: str = "step_size",
        target_acceptance_rate: float = 0.235,
        is_diagonal_matrix: bool = True,
        t0: int = 10,
        gamma: float = 0.05,
        kappa: float = 0.75,
        **_,
    ) -> Tuple[RWState, RWInfo]:
        position = state.position if hasattr(state, "position") else state
        if method == "step_size":
            adaption_alg = step_size_adaption(
                gauss_rwmh,
                logdensity_fn,
                params,
                target=target_acceptance_rate,
                t0=t0,
                gamma=gamma,
                kappa=kappa,
            )
        elif method == "step_size_and_scale":
            adaption_alg = step_size_and_scale_adaption(
                gauss_rwmh,
                logdensity_fn,
                params,
                target_acceptance_rate=target_acceptance_rate,
                is_diagonal_matrix=is_diagonal_matrix,
                t0=t0,
                gamma=gamma,
                kappa=kappa,
            )
        else:
            raise ValueError("Invalid method")

        out, _ = adaption_alg.run(key, position, num_steps)
        return out.state, out.parameters

    return fit_params


gauss_rwmh = make_kernel_api(
    name="gauss_rwmh",
    init_fn=blackjax.rmh.init,
    init_params_fn=init_params_gaussian_rw,
    build_step_fn=partial(
        build_mh_step, transition_proposal_fn=gaussian_transition_proposal
    ),
    build_adaptation_fn=build_adaptation,
)
