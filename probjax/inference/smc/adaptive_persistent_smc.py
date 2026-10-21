from __future__ import annotations

from typing import Any, Callable

import blackjax
from blackjax.smc import adaptive_persistent_sampling as bj_adaptive_persistent
from blackjax.smc import solver as bj_solver

from probjax.inference.smc.base import (
    SMCKernelAPI,
    init_tempered_mcmc_params,
    make_mcmc_adapter,
)
from probjax.inference.smc.resampling import systematic
from probjax.utils.typing import Array


def build_step(
    logprior_fn: Callable | None,
    loglikelihood_fn: Callable | None,
    *,
    mcmc_kernel: Any,
    num_mcmc_steps: int = 10,
    resampling_fn: Callable = systematic,
    target_ess: float = 0.5,
    update_strategy: Callable = blackjax.smc.base.update_and_take_last,
    root_solver: Callable = bj_solver.dichotomy,
    batch_size: int = 0,
    **mcmc_kernel_kwargs: Any,
) -> Callable:
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
        batch_size=batch_size,
    )

    def step(rng, state: Any, mcmc_parameters: Any) -> Any:
        return delegate(rng, state, num_mcmc_steps, mcmc_parameters)

    return step


def init_params(
    particles: Any,
    *,
    logprior_fn: Callable | None,
    loglikelihood_fn: Callable | None,
    mcmc_kernel: Any,
    rng = None,
    lmbda: float = 0.0,
    mcmc_kernel_kwargs: dict[str, Any] | None = None,
    **mcmc_param_kwargs: Any,
) -> dict[str, Array]:
    return init_tempered_mcmc_params(
        particles,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=mcmc_kernel,
        rng=rng,
        lmbda=lmbda,
        mcmc_kernel_kwargs=mcmc_kernel_kwargs,
        kind="adaptive persistent SMC",
        **mcmc_param_kwargs,
    )


def init(
    particles: Any,
    *,
    logprior_fn: Callable | None = None,
    loglikelihood_fn: Callable | None = None,
    batch_size: int = 0,
    max_iterations: int = 200,
) -> Any:
    if loglikelihood_fn is None:
        raise ValueError(
            "loglikelihood_fn must be supplied explicitly or bound by the constructor."
        )
    return bj_adaptive_persistent.init(
        particles, loglikelihood_fn, max_iterations, batch_size=batch_size
    )


class adaptive_persistent_smc(SMCKernelAPI):
    """Persistent sampling with adaptively chosen temperatures.

    Like :class:`persistent_smc`, but every temperature is chosen so the ESS
    of the persistent population is ``target_ess``.

    Args:
        logprior_fn: Log prior.
        loglikelihood_fn: Log likelihood.
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
        max_iterations: Maximum number of tempering steps, forwarded to
            ``init``.
        **mcmc_kernel_kwargs: Options for ``mcmc_kernel``, e.g.
            ``num_integration_steps``.
    """

    init = staticmethod(init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_step)
