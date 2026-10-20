from __future__ import annotations

from typing import Any, Callable

import blackjax
from blackjax.smc import adaptive_tempered as bj_adaptive_tempered
from blackjax.smc import solver as bj_solver

from probjax.inference.smc.base import (
    SMCKernelAPI,
    init_mcmc_params_from_logdensity,
    make_mcmc_adapter,
)
from probjax.inference.smc.resampling import systematic
from probjax.utils.typing import Array, RngKey


def build_step(
    logprior_fn: Callable | None,
    loglikelihood_fn: Callable | None,
    *,
    path: Any,
    mcmc_kernel: Any,
    num_mcmc_steps: int = 10,
    resampling_fn: Callable = systematic,
    target_ess: float = 0.8,
    root_solver: Callable = bj_solver.dichotomy,
    update_strategy: Callable = blackjax.smc.base.update_and_take_last,
    batch_size: int = 0,
    **mcmc_kernel_kwargs: Any,
) -> Callable:
    if not hasattr(path, "logdensity_fn") or not hasattr(path, "initial_param"):
        raise TypeError("path must define logdensity_fn and initial_param")
    # Only geometric path is supported by BlackJAX adaptive_tempered.
    if getattr(path, "is_geometric", False) is not True:
        raise ValueError("adaptive SMC is only supported for the geometric path.")

    if logprior_fn is None or loglikelihood_fn is None:
        raise ValueError("adaptive SMC requires both logprior_fn and loglikelihood_fn.")

    mcmc_init_fn, mcmc_step_fn = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    # BlackJAX forwards unknown keywords as a plain dict, so the update
    # strategy travels through ``extra_parameters``.
    extra_parameters: dict[str, Any] = {"update_strategy": update_strategy}
    kernel = bj_adaptive_tempered.build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        target_ess,
        root_solver=root_solver,
        batch_size=batch_size,
        **extra_parameters,
    )

    def step(rng_key: Any, state: Any, mcmc_parameters: Any) -> Any:
        return kernel(rng_key, state, num_mcmc_steps, mcmc_parameters)

    return step


def init_params(
    particles: Any,
    *,
    path: Any,
    logprior_fn: Callable | None,
    loglikelihood_fn: Callable | None,
    mcmc_kernel: Any,
    rng_key: RngKey | None = None,
    mcmc_kernel_kwargs: dict[str, Any] | None = None,
    path_kwargs: dict[str, Any] | None = None,
    **mcmc_param_kwargs: Any,
) -> dict[str, Array]:
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


def init(particles: Any, **_: Any) -> Any:
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
