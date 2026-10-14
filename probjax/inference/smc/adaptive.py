from typing import Callable, Dict, Optional

import blackjax
from blackjax.smc import adaptive_tempered as bj_adaptive_tempered

from probjax.inference.smc.base import (
    SMCKernelAPI,
    init_mcmc_params_from_logdensity,
    make_mcmc_adapter,
)
from probjax.inference.smc.resampling import systematic


def build_step(
    logprior_fn: Optional[Callable],
    loglikelihood_fn: Optional[Callable],
    *,
    path,
    mcmc_kernel,
    num_mcmc_steps: int = 10,
    resampling_fn: Callable = systematic,
    target_ess: float = 0.8,
    root_solver: Callable = blackjax.smc.solver.dichotomy,
    update_strategy: Callable = blackjax.smc.base.update_and_take_last,
    batch_size: int = 0,
    **mcmc_kernel_kwargs,
):
    if not hasattr(path, "logdensity_fn") or not hasattr(path, "initial_param"):
        raise TypeError("path must define logdensity_fn and initial_param")
    # Only geometric path is supported by BlackJAX adaptive_tempered.
    if getattr(path, "is_geometric", False) is not True:
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
        update_strategy=update_strategy,
        batch_size=batch_size,
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
    if getattr(path, "is_geometric", False) is not True:
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


class adaptive_smc(SMCKernelAPI):
    """Tempered SMC that picks each temperature to hit a target ESS.

    Moves along the geometric path ``logprior + t * loglikelihood`` from
    ``t = 0`` to ``1``, choosing every increment so the effective sample size
    after reweighting is ``target_ess``. Initial MCMC parameters come from
    ``kernel.init_params(particles, **mcmc_params)``.

    Args:
        logprior_fn: Log prior.
        loglikelihood_fn: Log likelihood.
        path: Tempering path; must be :class:`GeometricPath`.
        mcmc_kernel: MCMC kernel class used to move the particles, e.g.
            :class:`hmc` (the class, not a built kernel).
        target_ess: Target effective sample size as a fraction of the
            particles.
        root_solver: Solver for the next temperature.
        num_mcmc_steps: MCMC steps per SMC iteration.
        resampling_fn: BlackJAX resampling function.
        update_strategy: BlackJAX particle update strategy.
        batch_size: Particles per batch in the MCMC update; ``0`` updates all
            at once.
        **mcmc_kernel_kwargs: Options for ``mcmc_kernel``, e.g.
            ``num_integration_steps``.
    """

    init = staticmethod(init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_step)
