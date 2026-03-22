from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import blackjax

from blackjax.smc import persistent_sampling as bj_persistent

from probjax.inference.smc.base import (
    make_mcmc_adapter,
    make_smc_api,
    _params_to_dict,
    _ensure_param_batch,
    _filter_kwargs,
)
from probjax.inference.smc import tuning as smc_tuning

PersistentSMCState = bj_persistent.PersistentSMCState

# Module-level storage for loglikelihood_fn, set during build_step.
# Needed by init() to compute initial log-likelihoods for persistent state.
_ll_fn_holder = [None]


def build_step(
    logprior_fn: Optional[Callable],
    loglikelihood_fn: Optional[Callable],
    *,
    mcmc_kernel,
    num_mcmc_steps: int = 10,
    resampling_fn: Callable = blackjax.smc.resampling.systematic,
    update_strategy: Callable = blackjax.smc.base.update_and_take_last,
    **mcmc_kernel_kwargs,
):
    if mcmc_kernel is None:
        raise ValueError("mcmc_kernel must be provided for persistent SMC.")
    if logprior_fn is None or loglikelihood_fn is None:
        raise ValueError(
            "logprior_fn and loglikelihood_fn must be provided for persistent SMC."
        )

    _ll_fn_holder[0] = loglikelihood_fn

    mcmc_init_fn, mcmc_step_fn = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    delegate = bj_persistent.build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        update_strategy,
    )

    def step(rng_key, state, tempering_param, mcmc_parameters):
        return delegate(
            rng_key, state, num_mcmc_steps, tempering_param, mcmc_parameters
        )

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


def build_tuning(
    logprior_fn: Optional[Callable],
    loglikelihood_fn: Optional[Callable],
    *,
    method: str = "from_particles",
    **_,
):
    def tune_params(state, info, params, **kwargs):
        params = _params_to_dict(params)
        particles = state.particles
        if method == "from_particles":
            tuned = smc_tuning.tune_from_particles(params, particles, **kwargs)
            return _ensure_param_batch(tuned, shared=True)
        if method == "from_kernel_info":
            tuned = smc_tuning.tune_from_kernel_info(params, info.update_info, **kwargs)
            return _ensure_param_batch(tuned, shared=True)
        return _ensure_param_batch(params, shared=True)

    return tune_params


def init(particles, *, logprior_fn=None, loglikelihood_fn=None, n_schedule: int = 100):
    if loglikelihood_fn is None:
        loglikelihood_fn = _ll_fn_holder[0]
    if loglikelihood_fn is None:
        raise ValueError(
            "loglikelihood_fn must be provided either directly or via build_step."
        )
    return bj_persistent.init(particles, loglikelihood_fn, n_schedule)


persistent_smc = make_smc_api(
    name="persistent_smc",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=build_step,
    build_tuning_fn=build_tuning,
)
