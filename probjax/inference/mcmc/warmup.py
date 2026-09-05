from functools import partial

import blackjax
import jax
from blackjax.adaptation.base import return_all_adapt_info
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

from probjax.inference.adaptation import get_param, replace_params
from probjax.inference.base import Warmup, WarmupResult


def pathfinder_warmup(
    algorithm,
    logdensity_fn,
    *,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.8,
    collect: bool = False,
    **extra_parameters,
) -> Warmup:
    """Create a BlackJAX Pathfinder warmup procedure."""
    info_fn = return_all_adapt_info if collect else lambda *_: None
    procedure = blackjax.pathfinder_adaptation(
        algorithm,
        logdensity_fn,
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_acceptance_rate,
        adaptation_info_fn=info_fn,
        **extra_parameters,
    )

    @partial(jax.jit, static_argnames=("num_steps",))
    def run(key, _kernel, state, params, num_steps):
        result, info = procedure.run(key, state.position, num_steps)
        params = replace_params(params, **result.parameters)
        return WarmupResult(result.state, params, info if collect else None)

    return Warmup(run)


def mclmc_warmup(
    mclmc_kernel,
    *,
    adjusted: bool = False,
    target_acceptance_rate: float = 0.8,
    collect: bool = False,
    **options,
) -> Warmup:
    """Create a specialized BlackJAX MCLMC warmup procedure.

    ``mclmc_kernel`` follows the corresponding BlackJAX adaptation kernel
    protocol. MCLMC warmup advances the chain while estimating ``L``, step size,
    and optional diagonal preconditioning.
    """
    finder = (
        blackjax.adjusted_mclmc_find_L_and_step_size
        if adjusted
        else blackjax.mclmc_find_L_and_step_size
    )

    @partial(jax.jit, static_argnames=("num_steps",))
    def run(key, _kernel, state, params, num_steps):
        adaptation_params = MCLMCAdaptationState(
            get_param(params, "L"),
            get_param(params, "step_size"),
            get_param(params, "inverse_mass_matrix"),
        )
        kwargs = dict(options)
        if adjusted:
            kwargs["target"] = target_acceptance_rate
        state, adaptation_params, work = finder(
            mclmc_kernel=mclmc_kernel,
            num_steps=num_steps,
            state=state,
            rng_key=key,
            params=adaptation_params,
            **kwargs,
        )
        params = replace_params(
            params,
            L=adaptation_params.L,
            step_size=adaptation_params.step_size,
            inverse_mass_matrix=adaptation_params.inverse_mass_matrix,
        )
        return WarmupResult(state, params, work if collect else None)

    return Warmup(run)
