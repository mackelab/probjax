from __future__ import annotations

from typing import Any, Callable, NamedTuple

import blackjax
import jax
import jax.numpy as jnp
from blackjax.smc import from_mcmc as bj_from_mcmc

from probjax.inference.smc.base import (
    SMCKernelAPI,
    filter_kwargs,
    init_mcmc_params_from_logdensity,
    make_mcmc_adapter,
)
from probjax.inference.smc.resampling import systematic
from probjax.utils.typing import Array


class PathSMCState(NamedTuple):
    particles: Array
    weights: Array
    tempering_param: Array


def _log_weights_fn(
    path: Any,
    prev_param: Any,
    next_param: Any,
    *,
    logprior_fn: Callable | None,
    loglikelihood_fn: Callable | None,
    **path_kwargs: Any,
) -> tuple[Callable, Callable]:
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

    def log_weights(x: Any) -> Array:
        return logdensity_next(x) - logdensity_prev(x)

    return log_weights, logdensity_next


def build_step(
    logprior_fn: Callable | None,
    loglikelihood_fn: Callable | None,
    *,
    path: Any,
    mcmc_kernel: Any,
    num_mcmc_steps: int = 10,
    resampling_fn: Callable = systematic,
    update_strategy: Callable = blackjax.smc.base.update_and_take_last,
    path_kwargs: dict[str, Any] | None = None,
    batch_size: int = 0,
    **mcmc_kernel_kwargs: Any,
) -> Callable:
    if mcmc_kernel is None:
        raise ValueError("mcmc_kernel must be provided for SMC.")
    mcmc_init_fn, mcmc_step_fn = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    delegate = bj_from_mcmc.build_kernel(
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        update_strategy,
        batch_size=batch_size,
    )

    if path_kwargs is None:
        path_kwargs = filter_kwargs(
            path.logdensity_fn, mcmc_kernel_kwargs, allow_kwargs=False
        )
    else:
        path_kwargs = path_kwargs

    def step(
        rng, state: Any, tempering_param: Any, mcmc_parameters: Any
    ) -> Any:
        log_weights_fn, logposterior_fn = _log_weights_fn(
            path,
            state.tempering_param,
            tempering_param,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            **path_kwargs,
        )
        smc_state, info = delegate(
            rng,
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
    particles: Any,
    *,
    path: Any,
    logprior_fn: Callable | None,
    loglikelihood_fn: Callable | None,
    mcmc_kernel: Any,
    rng = None,
    initial_path_param: Any = None,
    mcmc_kernel_kwargs: dict[str, Any] | None = None,
    path_kwargs: dict[str, Any] | None = None,
    **mcmc_param_kwargs: Any,
) -> dict[str, Array]:
    mcmc_kernel_kwargs = mcmc_kernel_kwargs or {}
    if path_kwargs is None:
        path_kwargs = filter_kwargs(
            path.initial_param, mcmc_param_kwargs, allow_kwargs=False
        )
    else:
        path_kwargs = path_kwargs
    if initial_path_param is None:
        initial_path_param = path.initial_param(**path_kwargs)
    logdensity_kwargs = filter_kwargs(
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
        rng=rng,
        mcmc_kernel_kwargs=mcmc_kernel_kwargs,
        allow_kwargs=True,
        **mcmc_param_kwargs,
    )
    if "inverse_mass_matrix" in params_dict:
        imm = jnp.asarray(params_dict["inverse_mass_matrix"])
        if imm.ndim == 1 and imm.shape[0] == 1:
            params_dict["inverse_mass_matrix"] = imm[None, ...]
    return params_dict


def init(
    particles: Any,
    *,
    path: Any,
    initial_path_param: Any = None,
    **path_kwargs: Any,
) -> PathSMCState:
    if initial_path_param is None:
        initial_path_param = path.initial_param(**path_kwargs)
    num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    weights = jnp.ones(num_particles) / num_particles
    return PathSMCState(particles, weights, initial_path_param)


class path_smc(SMCKernelAPI):
    """SMC along a user-defined path of intermediate targets.

    The runner supplies the path parameter for each step (e.g. a temperature
    schedule); ``path`` turns it into a log density and incremental weights.
    Usually built through :func:`smc`. Initial MCMC parameters come from
    ``kernel.init_params(particles, **mcmc_params)``.

    Args:
        logprior_fn: Log prior.
        loglikelihood_fn: Log likelihood.
        path: Path object, e.g. :class:`GeometricPath` or
            :class:`PartialPosteriorsPath`.
        mcmc_kernel: MCMC kernel class used to move the particles, e.g.
            :class:`hmc` (the class, not a built kernel).
        num_mcmc_steps: MCMC steps per SMC iteration.
        resampling_fn: BlackJAX resampling function.
        update_strategy: BlackJAX particle update strategy.
        path_kwargs: Extra arguments for ``path``; inferred from
            ``mcmc_kernel_kwargs`` when ``None``.
        batch_size: Particles per batch in the MCMC update; ``0`` updates all
            at once.
        **mcmc_kernel_kwargs: Options for ``mcmc_kernel``, e.g.
            ``num_integration_steps``.
    """

    init = staticmethod(init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_step)
