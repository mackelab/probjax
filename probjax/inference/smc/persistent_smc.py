from typing import Callable, Dict, Optional

import blackjax
from blackjax.smc import persistent_sampling as bj_persistent

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
    mcmc_kernel,
    num_mcmc_steps: int = 10,
    resampling_fn: Callable = systematic,
    update_strategy: Callable = blackjax.smc.base.update_and_take_last,
    batch_size: int = 0,
    **mcmc_kernel_kwargs,
):
    if mcmc_kernel is None:
        raise ValueError("mcmc_kernel must be provided for persistent SMC.")
    if logprior_fn is None or loglikelihood_fn is None:
        raise ValueError(
            "logprior_fn and loglikelihood_fn must be provided for persistent SMC."
        )

    mcmc_init_fn, mcmc_step_fn = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    delegate = bj_persistent.build_kernel(
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        update_strategy,
        batch_size=batch_size,
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
    n_schedule: int = 100,
    batch_size=0,
):
    if loglikelihood_fn is None:
        raise ValueError(
            "loglikelihood_fn must be supplied explicitly or bound by the constructor."
        )
    return bj_persistent.init(
        particles, loglikelihood_fn, n_schedule, batch_size=batch_size
    )


class persistent_smc(SMCKernelAPI):
    """Persistent sampling: tempered SMC that reuses all past particles.

    Particles from every earlier temperature stay in the population and are
    reweighted, which lowers the variance for a fixed budget. Follows a fixed
    schedule of ``n_schedule`` temperatures; initial MCMC parameters come from
    ``kernel.init_params(particles, **mcmc_params)``.

    Args:
        logprior_fn: Log prior.
        loglikelihood_fn: Log likelihood.
        mcmc_kernel: MCMC kernel class used to move the particles, e.g.
            :class:`hmc` (the class, not a built kernel).
        num_mcmc_steps: MCMC steps per SMC iteration.
        resampling_fn: BlackJAX resampling function.
        update_strategy: BlackJAX particle update strategy.
        batch_size: Particles per batch in the MCMC update; ``0`` updates all
            at once.
        n_schedule: Number of tempering steps, forwarded to ``init``.
        **mcmc_kernel_kwargs: Options for ``mcmc_kernel``, e.g.
            ``num_integration_steps``.
    """

    init = staticmethod(init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_step)
