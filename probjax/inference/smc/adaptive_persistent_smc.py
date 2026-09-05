from typing import Callable, Dict, Optional

import blackjax
import jax
from blackjax.smc import adaptive_persistent_sampling as bj_adaptive_persistent
from blackjax.smc import persistent_sampling as bj_persistent

from probjax.inference.smc.base import (
    _ensure_param_batch,
    _filter_kwargs,
    _params_to_dict,
    make_mcmc_adapter,
    make_smc_api,
)

PersistentSMCState = bj_persistent.PersistentSMCState

_ll_fn_holder = [None]


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

    _ll_fn_holder[0] = loglikelihood_fn

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
    mcmc_kernel_kwargs = mcmc_kernel_kwargs or {}
    mcmc_init_fn, _ = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    particle0 = jax.tree_util.tree_map(lambda x: x[0], particles)

    def tempered_logposterior_fn(position):
        return logprior_fn(position) + lmbda * loglikelihood_fn(position)

    mcmc_state = mcmc_init_fn(particle0, tempered_logposterior_fn, rng_key=rng_key)
    init_param_kwargs = _filter_kwargs(
        mcmc_kernel.init_params, mcmc_param_kwargs, allow_kwargs=False
    )
    params = mcmc_kernel.init_params(mcmc_state, **init_param_kwargs)
    return _ensure_param_batch(_params_to_dict(params), shared=True)


def init(
    particles,
    *,
    logprior_fn=None,
    loglikelihood_fn=None,
    max_iterations: int = 200,
):
    if loglikelihood_fn is None:
        loglikelihood_fn = _ll_fn_holder[0]
    if loglikelihood_fn is None:
        raise ValueError(
            "loglikelihood_fn must be provided either directly or via build_step."
        )
    return bj_adaptive_persistent.init(particles, loglikelihood_fn, max_iterations)


adaptive_persistent_smc = make_smc_api(
    name="adaptive_persistent_smc",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=build_step,
)
