from typing import Callable, Dict, Optional

import blackjax
from blackjax.smc import adaptive_tempered as bj_adaptive_tempered

from probjax.inference.smc.base import (
    init_mcmc_params_from_logdensity,
    make_mcmc_adapter,
    make_smc_api,
)


def build_step(
    logprior_fn: Optional[Callable],
    loglikelihood_fn: Optional[Callable],
    *,
    path,
    mcmc_kernel,
    num_mcmc_steps: int = 10,
    resampling_fn: Callable = blackjax.smc.resampling.systematic,
    target_ess: float = 0.8,
    root_solver: Callable = blackjax.smc.solver.dichotomy,
    **mcmc_kernel_kwargs,
):
    if not hasattr(path, "logdensity_fn") or not hasattr(path, "initial_param"):
        raise TypeError("path must define logdensity_fn and initial_param")
    # Only geometric path is supported by BlackJAX adaptive_tempered.
    if not hasattr(path, "is_geometric") and path.__class__.__name__ != "GeometricPath":
        raise ValueError("adaptive SMC is only supported for the geometric path.")

    mcmc_init_fn, mcmc_step_fn = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    kernel = bj_adaptive_tempered.build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        target_ess,
        root_solver=root_solver,
    )

    def step(rng_key, state, mcmc_parameters):
        return kernel(rng_key, state, num_mcmc_steps, mcmc_parameters)

    return step


def init_params(
    particles,
    *,
    path,
    logprior_fn: Optional[Callable],
    loglikelihood_fn: Optional[Callable],
    mcmc_kernel,
    rng_key=None,
    mcmc_kernel_kwargs: Optional[Dict] = None,
    path_kwargs: Optional[Dict] = None,
    **mcmc_param_kwargs,
) -> Dict:
    if not hasattr(path, "is_geometric") and path.__class__.__name__ != "GeometricPath":
        raise ValueError("adaptive SMC is only supported for the geometric path.")
    path_kwargs = path_kwargs or {}
    logposterior_fn = path.logdensity_fn(
        path.initial_param(**path_kwargs),
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )
    return init_mcmc_params_from_logdensity(
        particles,
        logposterior_fn,
        mcmc_kernel,
        rng_key=rng_key,
        mcmc_kernel_kwargs=mcmc_kernel_kwargs,
        **mcmc_param_kwargs,
    )


def init(particles, **_):
    return bj_adaptive_tempered.init(particles)


adaptive_smc = make_smc_api(
    name="adaptive_smc",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=build_step,
)
