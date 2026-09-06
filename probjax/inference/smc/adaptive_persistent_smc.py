from typing import Callable, Dict, Optional

import blackjax
from blackjax.smc import adaptive_persistent_sampling as bj_adaptive_persistent
from blackjax.smc import persistent_sampling as bj_persistent

from probjax.inference.smc.base import (
    init_mcmc_params_from_logdensity,
    make_mcmc_adapter,
    make_smc_api,
)

PersistentSMCState = bj_persistent.PersistentSMCState


def build_step(
    logprior_fn: Optional[Callable],
    loglikelihood_fn: Optional[Callable],
    *,
    mcmc_kernel,
    num_mcmc_steps: int = 10,
    resampling_fn: Callable = blackjax.smc.resampling.systematic,
    target_ess: float = 0.5,
    update_strategy: Callable = blackjax.smc.base.update_and_take_last,
    root_solver: Callable = blackjax.smc.solver.dichotomy,
    **mcmc_kernel_kwargs,
):
    if mcmc_kernel is None:
        raise ValueError("mcmc_kernel must be provided for adaptive persistent SMC.")
    if logprior_fn is None or loglikelihood_fn is None:
        raise ValueError(
            "logprior_fn and loglikelihood_fn must be provided for adaptive "
            "persistent SMC."
        )

    mcmc_init_fn, mcmc_step_fn = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    delegate = bj_adaptive_persistent.build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        target_ess,
        update_strategy,
        root_solver,
    )

    def step(rng_key, state, mcmc_parameters):
        return delegate(rng_key, state, num_mcmc_steps, mcmc_parameters)

    return step


def init_params(
    particles,
    *,
    logprior_fn: Optional[Callable],
    loglikelihood_fn: Optional[Callable],
    mcmc_kernel,
    rng_key=None,
    lmbda: float = 0.0,
    mcmc_kernel_kwargs: Optional[Dict] = None,
    **mcmc_param_kwargs,
) -> Dict:
    def tempered_logposterior_fn(position):
        return logprior_fn(position) + lmbda * loglikelihood_fn(position)

    return init_mcmc_params_from_logdensity(
        particles,
        tempered_logposterior_fn,
        mcmc_kernel,
        rng_key=rng_key,
        mcmc_kernel_kwargs=mcmc_kernel_kwargs,
        **mcmc_param_kwargs,
    )


def init(
    particles,
    *,
    logprior_fn=None,
    loglikelihood_fn=None,
    max_iterations: int = 200,
):
    if loglikelihood_fn is None:
        raise ValueError(
            "loglikelihood_fn must be provided to adaptive persistent SMC init."
        )
    return bj_adaptive_persistent.init(particles, loglikelihood_fn, max_iterations)


adaptive_persistent_smc = make_smc_api(
    name="adaptive_persistent_smc",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=build_step,
)
