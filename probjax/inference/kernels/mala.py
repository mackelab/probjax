from typing import Callable, NamedTuple, Tuple

import blackjax
from blackjax.mcmc.mala import MALAInfo, MALAState
from chex import PRNGKey
from jaxtyping import PyTree

from probjax.inference.kernels.adaptation import step_size_adaption
from probjax.inference.kernels.base import MarkovKernelAPI


class MALAParams(NamedTuple):
    step_size: float


def build_step(logdensity_fn: Callable) -> Callable:
    kernel = blackjax.mala.build_kernel()

    def step(
        key: PRNGKey, state: MALAState, params: MALAParams
    ) -> Tuple[MALAState, MALAInfo]:
        return kernel(
            key, state, logdensity_fn=logdensity_fn, step_size=params.step_size
        )

    return step


def build_adaptation(
    logdensity_fn: Callable,
) -> Callable:
    def adapt_parms(
        key: PRNGKey,
        position: PyTree,
        params: MALAParams,
        num_steps: int = 100,
        target_acceptance_rate: float = 0.65,
        t0: int = 10,
        gamma: float = 0.05,
        kappa: float = 0.75,
    ) -> Tuple[MALAState, MALAInfo]:
        adaption_alg = step_size_adaption(
            MALA,
            logdensity_fn,
            params,
            target=target_acceptance_rate,
            t0=t0,
            gamma=gamma,
            kappa=kappa,
        )

        out, _ = adaption_alg.run(key, position, num_steps)
        return out.state, out.parameters

    return adapt_parms


def init_params(position: PyTree, step_size: float = 1e-2) -> MALAParams:
    return MALAParams(step_size=step_size)


class MALA(MarkovKernelAPI):
    init = blackjax.mala.init
    build_step = build_step
    init_params = init_params
    build_adaptation = build_adaptation
