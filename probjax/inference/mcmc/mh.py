from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.random_walk import RWInfo, RWState
from jax.flatten_util import ravel_pytree

from probjax.inference.adaptation import compose_adaptors
from probjax.inference.mcmc.adaptation import covariance_adaptor, step_size_adaptor
from probjax.inference.mcmc.base import MarkovKernelAPI
from probjax.utils.typing import Array, RngKey, Scalar


class RWParams(NamedTuple):
    pass


def build_mh_step(
    logdensity_fn: Callable,
    transition_proposal_fn: Callable,
    transition_proposal_logpdf: Callable | None = None,
) -> Callable:
    kernel = blackjax.rmh.build_kernel()

    def step(
        key: RngKey,
        state: RWState,
        params: RWParams,
    ) -> tuple[RWState, RWInfo]:
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


def init_params_mh(state: Any) -> RWParams:
    return RWParams()


class mh(MarkovKernelAPI):
    r"""Metropolis-Hastings with a user-supplied proposal.

    The kernel params are passed to the proposal as ``params=``; the default
    ``init_params`` returns the empty :class:`RWParams`.

    Args:
        logdensity_fn: Log density of the target.
        transition_proposal_fn: ``(key, position, params) -> new_position``.
        transition_proposal_logpdf: ``(new_state, state, params) -> log q``,
            evaluated on BlackJAX ``RWState``\ s; needed only for asymmetric
            proposals.
    """

    init = staticmethod(blackjax.rmh.init)
    init_params = staticmethod(init_params_mh)
    build_step = staticmethod(build_mh_step)


class RWParamsGauss(NamedTuple):
    step_size: Scalar
    scale: Array | None


def init_params_gaussian_rw(
    state: Any, step_size: float = 0.5, scale: Array | None = None
) -> RWParamsGauss:
    position = state.position if hasattr(state, "position") else state
    flat_position, _ = ravel_pytree(position)
    dim = flat_position.shape[0]
    if scale is None:
        scale = jnp.ones((dim,))
    return RWParamsGauss(step_size=step_size, scale=scale)


def gaussian_transition_proposal(
    key: RngKey, position: Any, params: RWParamsGauss
) -> Any:
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


def _gauss_rwmh_adaptation():
    return compose_adaptors(
        step=step_size_adaptor(target=0.234),
        scale=covariance_adaptor(field="scale", output="standard_deviation"),
    )


class gauss_rwmh(MarkovKernelAPI):
    """Gaussian random-walk Metropolis-Hastings.

    Proposes ``x + step_size * scale @ eps`` with standard normal ``eps``.
    Tunable parameters come from ``kernel.init_params(state, step_size=0.5,
    scale=None)`` (see :class:`RWParamsGauss`); ``scale`` is a diagonal
    (vector) or a dense matrix square root and defaults to the identity.
    Default adaptation: step size towards a 0.234 acceptance rate plus a
    covariance estimate for ``scale``.

    Args:
        logdensity_fn: Log density of the target.
    """

    init = staticmethod(blackjax.rmh.init)
    init_params = staticmethod(init_params_gaussian_rw)
    build_step = staticmethod(
        partial(build_mh_step, transition_proposal_fn=gaussian_transition_proposal)
    )
    default_adaptation_fn = staticmethod(_gauss_rwmh_adaptation)
