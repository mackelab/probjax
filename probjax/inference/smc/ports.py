"""Small adapters exposing BlackJAX's SMC update strategies and tuning."""

from __future__ import annotations

from typing import Any, Callable

import blackjax
import jax
from blackjax.smc import from_mcmc, inner_kernel_tuning, pretuning
from blackjax.smc.base import extend_params
from blackjax.smc.waste_free import waste_free_smc as _waste_free_strategy

from probjax.inference.base import Kernel
from probjax.inference.smc.base import make_mcmc_adapter
from probjax.inference.smc.resampling import systematic
from probjax.utils.jaxutils import leading_dim
from probjax.utils.typing import RngKey


def tuned_smc(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    *,
    mcmc_kernel: Any,
    mcmc_parameters: dict[str, Any],
    parameter_update_fn: Callable,
    adaptive: bool = True,
    num_mcmc_steps: int = 10,
    target_ess: float = 0.8,
    batch_size: int = 0,
    resampling_fn: Callable = systematic,
    **mcmc_kernel_kwargs: Any,
) -> Kernel:
    """Wrap BlackJAX inner-kernel tuning with a probjax MCMC kernel.

    parameter_update_fn(key, new_smc_state, info) returns the next MCMC parameter
    dictionary, with BlackJAX's leading shared (1) or per-particle (N) axis.
    mcmc_parameters supplies initial *shared*, unbatched parameter values.
    adaptive=False accepts explicit temperatures through SMC.run; otherwise use
    SMC.run_adaptive. Parameters live in state.parameter_override.
    """
    init_mcmc, step_mcmc = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    # BlackJAX annotates ``extend_params`` with a single array, but it maps
    # over the parameter tree.
    shared_parameters: Any = mcmc_parameters
    parameters = extend_params(shared_parameters)
    algorithm = blackjax.adaptive_tempered_smc if adaptive else blackjax.tempered_smc
    kwargs: dict[str, Any] = {"batch_size": batch_size}
    if adaptive:
        kwargs["target_ess"] = target_ess
    # BlackJAX types the returned algorithm's step as a two-argument kernel,
    # but the tempered SMC step also takes the temperature.
    delegate: Any = inner_kernel_tuning.as_top_level_api(
        algorithm,
        logprior_fn,
        loglikelihood_fn,
        step_mcmc,
        init_mcmc,
        resampling_fn,
        parameter_update_fn,
        parameters,
        num_mcmc_steps=num_mcmc_steps,
        **kwargs,
    )

    def step(
        key: RngKey,
        state: Any,
        mcmc_parameters: Any = None,
        tempering_param: Any = None,
    ) -> Any:
        if adaptive:
            return delegate.step(key, state)
        return delegate.step(key, state, tempering_param=tempering_param)

    return Kernel(delegate.init, step, lambda *a, **kw: parameters)


def pretuned_smc(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    *,
    mcmc_kernel: Any,
    mcmc_parameters: dict[str, Any],
    num_particles: int,
    sigma_parameters: dict[str, Any],
    alpha: float = 0.5,
    adaptive: bool = True,
    num_mcmc_steps: int = 10,
    target_ess: float = 0.8,
    batch_size: int = 0,
    positive_parameters: Any = None,
    natural_parameters: Any = None,
    performance_of_chain_measure_factory: Callable = pretuning.default_measure_factory,
    resampling_fn: Callable = systematic,
    **mcmc_kernel_kwargs: Any,
) -> Kernel:
    """BlackJAX pilot-move pretuning, retaining a distribution of move parameters.

    sigma_parameters selects parameters to perturb and their noise scales.
    Initial mcmc_parameters use BlackJAX's explicit leading 1/N batch convention
    (unlike tuned_smc). The default ESJD measure requires a full shared inverse
    mass matrix of shape (1,D,D). Supply a custom measure factory for other moves.
    Pilot and production transitions use independent keys. Production moves
    respect batch_size; BlackJAX's pilot currently evaluates its population together.
    """
    init_mcmc, step_mcmc = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    pilot = pretuning.build_pretune(
        init_mcmc,
        step_mcmc,
        alpha,
        sigma_parameters,
        num_particles,
        performance_of_chain_measure_factory=performance_of_chain_measure_factory,
        positive_parameters=positive_parameters,
        natural_parameters=natural_parameters,
    )
    move = from_mcmc.build_kernel(
        step_mcmc, init_mcmc, resampling_fn, batch_size=batch_size
    )

    def update(
        key: RngKey,
        state: Any,
        num_steps: int,
        parameters: Any,
        logdensity: Callable,
        logweights: Callable,
    ) -> Any:
        pilot_key, move_key = jax.random.split(key)
        parameters = pilot(
            pilot_key,
            inner_kernel_tuning.StateWithParameterOverride(state, dict(parameters)),
            logdensity,
        )
        state, info = move(
            move_key, state, num_steps, parameters, logdensity, logweights
        )
        return state, pretuning.SMCInfoWithParameterDistribution(info, parameters)

    algorithm = blackjax.adaptive_tempered_smc if adaptive else blackjax.tempered_smc
    kwargs: dict[str, Any] = {"update_particles_fn": update, "batch_size": batch_size}
    if adaptive:
        kwargs["target_ess"] = target_ess

    def init(particles: Any) -> Any:
        if leading_dim(particles) != num_particles:
            raise ValueError("num_particles must match the initial population.")
        return inner_kernel_tuning.StateWithParameterOverride(
            blackjax.tempered_smc.init(particles), dict(mcmc_parameters)
        )

    def step(
        key: RngKey,
        state: Any,
        mcmc_parameters: Any = None,
        tempering_param: Any = None,
    ) -> Any:
        delegate: Any = algorithm(
            logprior_fn,
            loglikelihood_fn,
            step_mcmc,
            init_mcmc,
            state.parameter_override,
            resampling_fn,
            num_mcmc_steps=num_mcmc_steps,
            **kwargs,
        )
        step_kwargs = {} if adaptive else dict(tempering_param=tempering_param)
        sampler, info = delegate.step(key, state.sampler_state, **step_kwargs)
        return inner_kernel_tuning.StateWithParameterOverride(
            sampler, info.parameter_override
        ), info.smc_info

    return Kernel(init, step, lambda *a, **kw: dict(mcmc_parameters))


__all__ = ["waste_free_strategy", "waste_free_smc", "tuned_smc", "pretuned_smc"]


def waste_free_strategy(num_particles: int, p: int) -> Any:
    """BlackJAX waste-free update: retain p states per chain.

    num_particles must be divisible by p. When supplying this strategy manually,
    set num_mcmc_steps=None; p determines the chain length instead.
    """
    if p < 1 or num_particles < 1:
        raise ValueError("num_particles and p must be positive.")
    return _waste_free_strategy(num_particles, p)


def waste_free_smc(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    *,
    num_particles: int,
    p: int = 4,
    adaptive: bool = False,
    **kwargs: Any,
) -> Kernel:
    """Convenient waste-free preset for fixed or adaptive geometric/path SMC."""
    # Deferred: this module is re-exported by smc/__init__, so importing the
    # package-level factories at top level would be a self-parent cycle.
    from probjax.inference.smc import adaptive_smc_kernel, smc

    if "num_mcmc_steps" in kwargs or "update_strategy" in kwargs:
        raise ValueError("Use p to configure waste-free chain length.")
    constructor = adaptive_smc_kernel if adaptive else smc
    return constructor(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_mcmc_steps=None,
        update_strategy=waste_free_strategy(num_particles, p),
        **kwargs,
    )
