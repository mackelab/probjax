from typing import Callable, Dict, NamedTuple, Optional

import blackjax
import jax
import jax.numpy as jnp
from blackjax.smc import from_mcmc as bj_from_mcmc

from probjax.inference.smc.base import (
    _filter_kwargs,
    init_mcmc_params_from_logdensity,
    make_mcmc_adapter,
    make_smc_api,
)


class PathSMCState(NamedTuple):
    particles: jnp.ndarray
    weights: jnp.ndarray
    tempering_param: jnp.ndarray


def _log_weights_fn(
    path, prev_param, next_param, *, logprior_fn, loglikelihood_fn, **path_kwargs
):
    logdensity_prev = path.logdensity_fn(
        prev_param,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        **path_kwargs,
    )
    logdensity_next = path.logdensity_fn(
        next_param,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        **path_kwargs,
    )

    def log_weights(x):
        return logdensity_next(x) - logdensity_prev(x)

    return log_weights, logdensity_next


def build_step(
    logprior_fn: Optional[Callable],
    loglikelihood_fn: Optional[Callable],
    *,
    path,
    mcmc_kernel,
    num_mcmc_steps: int = 10,
    resampling_fn: Callable = blackjax.smc.resampling.systematic,
    update_strategy: Callable = blackjax.smc.base.update_and_take_last,
    path_kwargs: Optional[Dict] = None,
    **mcmc_kernel_kwargs,
):
    if mcmc_kernel is None:
        raise ValueError("mcmc_kernel must be provided for SMC.")
    mcmc_init_fn, mcmc_step_fn = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    delegate = bj_from_mcmc.build_kernel(
        mcmc_step_fn, mcmc_init_fn, resampling_fn, update_strategy
    )

    if path_kwargs is None:
        path_kwargs = _filter_kwargs(
            path.logdensity_fn, mcmc_kernel_kwargs, allow_kwargs=False
        )
    else:
        path_kwargs = path_kwargs

    def step(rng_key, state, tempering_param, mcmc_parameters):
        log_weights_fn, logposterior_fn = _log_weights_fn(
            path,
            state.tempering_param,
            tempering_param,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            **path_kwargs,
        )
        smc_state, info = delegate(
            rng_key,
            state,
            num_mcmc_steps,
            mcmc_parameters,
            logposterior_fn,
            log_weights_fn,
        )
        return (
            PathSMCState(smc_state.particles, smc_state.weights, tempering_param),
            info,
        )

    return step


def init_params(
    particles,
    *,
    path,
    logprior_fn: Optional[Callable],
    loglikelihood_fn: Optional[Callable],
    mcmc_kernel,
    rng_key=None,
    initial_path_param=None,
    mcmc_kernel_kwargs: Optional[Dict] = None,
    path_kwargs: Optional[Dict] = None,
    **mcmc_param_kwargs,
) -> Dict:
    mcmc_kernel_kwargs = mcmc_kernel_kwargs or {}
    if path_kwargs is None:
        path_kwargs = _filter_kwargs(
            path.initial_param, mcmc_param_kwargs, allow_kwargs=False
        )
    else:
        path_kwargs = path_kwargs
    if initial_path_param is None:
        initial_path_param = path.initial_param(**path_kwargs)
    logdensity_kwargs = _filter_kwargs(
        path.logdensity_fn, mcmc_param_kwargs, allow_kwargs=False
    )
    logposterior_fn = path.logdensity_fn(
        initial_path_param,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        **{**path_kwargs, **logdensity_kwargs},
    )
    params_dict = init_mcmc_params_from_logdensity(
        particles,
        logposterior_fn,
        mcmc_kernel,
        rng_key=rng_key,
        mcmc_kernel_kwargs=mcmc_kernel_kwargs,
        allow_kwargs=True,
        **mcmc_param_kwargs,
    )
    if "inverse_mass_matrix" in params_dict:
        imm = jnp.asarray(params_dict["inverse_mass_matrix"])
        if imm.ndim == 1 and imm.shape[0] == 1:
            params_dict["inverse_mass_matrix"] = imm[None, ...]
    return params_dict


def init(particles, *, path, initial_path_param=None, **path_kwargs):
    if initial_path_param is None:
        initial_path_param = path.initial_param(**path_kwargs)
    num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    weights = jnp.ones(num_particles) / num_particles
    return PathSMCState(particles, weights, initial_path_param)


path_smc = make_smc_api(
    name="path_smc",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=build_step,
)
