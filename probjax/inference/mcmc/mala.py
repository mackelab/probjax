from typing import Callable, NamedTuple, Tuple

import blackjax
from blackjax.mcmc.mala import MALAInfo, MALAState
from probjax.utils.typing import PyTree, RngKey

from probjax.inference.mcmc.adaptation import step_size_adaption
from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel
from probjax.inference.mcmc.hmc import _scale_step_size_by_grad


class MALAParams(NamedTuple):
    step_size: float


def build_step(logdensity_fn: Callable) -> Callable:
    return make_step_from_kernel(
        logdensity_fn,
        blackjax.mala.build_kernel,
    )


def build_adaptation(logdensity_fn: Callable) -> Callable:
    def fit_params(
        key: RngKey,
        state: PyTree,
        params: MALAParams,
        num_steps: int = 100,
        target_acceptance_rate: float = 0.65,
        t0: int = 10,
        gamma: float = 0.05,
        kappa: float = 0.75,
        **_,
    ) -> Tuple[MALAState, MALAInfo]:
        position = state.position if hasattr(state, "position") else state
        adaption_alg = step_size_adaption(
            mala,
            logdensity_fn,
            params,
            target=target_acceptance_rate,
            t0=t0,
            gamma=gamma,
            kappa=kappa,
        )

        out, _ = adaption_alg.run(key, position, num_steps)
        return out.state, out.parameters

    return fit_params


def init_params(state: PyTree, step_size: float = 1e-2) -> MALAParams:
    step_size = _scale_step_size_by_grad(state, step_size)
    return MALAParams(step_size=step_size)


mala = make_kernel_api(
    name="mala",
    init_fn=blackjax.mala.init,
    init_params_fn=init_params,
    build_step_fn=build_step,
    build_adaptation_fn=build_adaptation,
)
