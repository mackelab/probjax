from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.random_walk import RWInfo, RWState
from chex import PRNGKey
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
from jaxtyping import Array, PyTree

from probjax.inference.kernels.adaptation import (
    step_size_adaption,
)
from probjax.inference.kernels.base import MarkovKernelAPI


class RWParams(NamedTuple):
    pass


def build_kernel_mh(
    logdensity_fn: Callable,
    transition_proposal_fn: Callable,
    transition_proposal_logpdf: Optional[Callable] = None,
) -> Callable:
    kernel = blackjax.rmh.build_kernel()

    def step(
        key: PRNGKey,
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


def init_params_mh(position: PyTree) -> RWParams:
    return RWParams()


class MH(MarkovKernelAPI):
    init = blackjax.rmh.init
    build_kernel = build_kernel_mh
    init_params = init_params_mh


class RWParamsGauss(NamedTuple):
    step_size: float
    scale: Optional[ArrayLike]


def init_params_gaussian_rw(
    position: PyTree, step_size: float = 0.5, scale: Optional[Array] = None
) -> RWParamsGauss:
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
    def adapt_parms(
        key: PRNGKey,
        position: PyTree,
        params: RWParamsGauss,
        num_steps: int = 100,
        method: str = "step_size",
        target_acceptance_rate: float = 0.235,
        t0: int = 10,
        gamma: float = 0.05,
        kappa: float = 0.75,
    ) -> Tuple[RWState, RWInfo]:
        if method == "step_size":
            adaption_alg = step_size_adaption(
                GaussRWMH,
                logdensity_fn,
                params,
                target_acceptance_rate=target_acceptance_rate,
                t0=t0,
                gamma=gamma,
                kappa=kappa,
            )

        out, _ = adaption_alg.run(key, position, num_steps)
        return out.state, out.parameters

    return adapt_parms


class GaussRWMH(MH):
    init = blackjax.rmh.init
    build_kernel = partial(
        build_kernel_mh, transition_proposal_fn=gaussian_transition_proposal
    )
    init_params = init_params_gaussian_rw
    build_adaptation = build_adaptation


# class GaussianMHKernel(MetropolisHastingKernel):
#     params: GaussRWParams

#     def __init__(
#         self,
#         logdensity_fn: Callable,
#         step_size: float = 2.38,
#         scale: Optional[Array] = None,
#         is_scale_diagonal: bool = True,
#     ) -> None:
#         self.scale = scale
#         if scale is not None:
#             if len(scale.shape) == 1:
#                 self.is_scale_diagonal = True
#             else:
#                 self.is_scale_diagonal = False
#         else:
#             self.is_scale_diagonal = is_scale_diagonal
#         self.step_size = step_size

#         self.params = GaussRWParams(step_size=step_size, scale=scale)
#         super().__init__(logdensity_fn, gaussian_transition_proposal, None)

#     def init_params(self, position: PyTree):
#         flat_position, _ = ravel_pytree(position)
#         dim = flat_position.shape[0]
#         if self.scale is None:
#             if self.is_scale_diagonal:
#                 scale = jnp.ones((dim,))
#             else:
#                 scale = jnp.eye(dim)
#         else:
#             scale = self.scale

#         step_size = self.step_size / jnp.sqrt(dim)

#         self.params = GaussRWParams(step_size=step_size, scale=scale)

#         return self.params

#     def adapt_params(
#         self,
#         key: PRNGKey,
#         position: PyTree,
#         num_steps: int = 100,
#         method="step_size_and_scale",
#         target_acceptance_rate=0.234,
#         **kwargs: Any,
#     ) -> Tuple[RWState, RWInfo]:
#         if method == "step_size":
#             adaption_alg = step_size_adaption(
#                 blackjax.rmh,
#                 self.logdensity_fn,
#                 self.params,
#                 gaussian_transition_proposal,
#                 target_acceptance_rate=target_acceptance_rate,
#                 **kwargs,
#             )

#             results, info = adaption_alg.run(key, position, num_steps)
#             step_size = results.parameters["step_size"]
#             self.params = GaussRWParams(step_size=step_size, scale=self.params.scale)
#         elif method == "step_size_and_scale":
#             adaption_alg = step_size_and_scale_adaption(
#                 blackjax.rmh,
#                 self.logdensity_fn,
#                 self.params,
#                 gaussian_transition_proposal,
#                 target_acceptance_rate=target_acceptance_rate,
#                 **kwargs,
#             )
#             results, info = adaption_alg.run(key, position, num_steps)
#             step_size = results.parameters["step_size"]
#             scale = results.parameters["scale"]
#             self.params = GaussRWParams(step_size=step_size, scale=scale)
#         else:
#             raise ValueError("Invalid method")

#         return results.state, info
