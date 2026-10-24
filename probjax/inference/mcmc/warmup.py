from __future__ import annotations

from functools import partial
from typing import Any

import blackjax
import jax
from blackjax.adaptation.base import return_all_adapt_info
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

from probjax.inference.adaptation import get_param, replace_params
from probjax.inference.base import RunResult, Warmup


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
    # BlackJAX types the adaptation procedure's ``run`` with two arguments,
    # but the Pathfinder procedure also takes ``num_steps``.
    procedure: Any = blackjax.pathfinder_adaptation(
        algorithm,
        logdensity_fn,
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_acceptance_rate,
        adaptation_info_fn=info_fn,
        **extra_parameters,
    )

    @partial(jax.jit, static_argnames=("_kernel", "num_steps"))
    def run(rng, _kernel, state, params, num_steps):
        result, info = procedure.run(rng, state.position, num_steps)
        params = replace_params(params, **result.parameters)
        return RunResult(result.state, params, info if collect else None)

    return Warmup(run)


def mclmc_warmup(
    logdensity_fn,
    *,
    collect: bool = False,
    **options,
) -> Warmup:
    """Create BlackJAX's MCLMC warmup for an :class:`~probjax.inference.mclmc` kernel.

    Advances the chain while estimating ``L``, the step size and, optionally,
    a diagonal preconditioner. It tunes the kernel it is run with, so a custom
    integrator is respected. ``options`` go to
    ``blackjax.mclmc_find_L_and_step_size``.
    """

    @partial(jax.jit, static_argnames=("kernel", "num_steps"))
    def run(rng, kernel, state, params, num_steps):
        def tuned_kernel(
            rng_key, state, logdensity_fn, inverse_mass_matrix, L, step_size
        ):
            tuned = replace_params(
                params,
                L=L,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
            )
            return kernel.step(rng_key, state, tuned)

        adaptation_params = MCLMCAdaptationState(
            get_param(params, "L"),
            get_param(params, "step_size"),
            get_param(params, "inverse_mass_matrix"),
        )
        state, adaptation_params, work = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=tuned_kernel,
            num_steps=num_steps,
            state=state,
            rng_key=rng,
            logdensity_fn=logdensity_fn,
            params=adaptation_params,
            **options,
        )
        params = replace_params(
            params,
            L=adaptation_params.L,
            step_size=adaptation_params.step_size,
            inverse_mass_matrix=adaptation_params.inverse_mass_matrix,
        )
        return RunResult(state, params, work if collect else None)

    return Warmup(run)
