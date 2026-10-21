"""Stochastic Gradient Markov Chain Monte Carlo (SGMCMC) samplers.

Wraps BlackJAX's SGLD, SGHMC, and SGNHT into the probjax MarkovKernel API.

Extra positional arguments passed to ``step`` (or ``__call__``) are forwarded
to the ``grad_estimator``/``logdensity_fn``.  For SG-MCMC the first extra arg
is typically the minibatch.

Usage::

    from probjax.inference.mcmc.sgmcmc import grad_estimator, sgld

    ge = grad_estimator(logprior_fn, loglikelihood_fn, data_size)
    sampler = sgld(ge, temperature=0.5)
    state = sampler.init(position)
    params = sampler.init_params(state)

    # Manual loop
    for minibatch in data_loader:
        state, info = sampler.step(key, state, params, minibatch)

    # Or with the MCMC runner (stack batches along leading axis)
    batches = jnp.stack([next(data_loader) for _ in range(num_steps)])
    mcmc = MCMC(sampler)
    state = mcmc.run(key, state, num_steps, params=params, args=(batches,))
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple

import blackjax
from blackjax.sgmcmc.sgnht import SGNHTState
from blackjax.util import generate_gaussian_noise

from probjax.inference.mcmc.base import MarkovKernel
from probjax.utils.typing import ArrayTree, RngKey, Scalar

# ---------------------------------------------------------------------------
# Shared state / info types
# ---------------------------------------------------------------------------


class SGMCMCState(NamedTuple):
    """Wrapper that gives SGLD / SGHMC states a ``.position`` attribute."""

    position: ArrayTree


class SGMCMCInfo(NamedTuple):
    """Empty info — SG-MCMC has no acceptance step."""

    pass


# ---------------------------------------------------------------------------
# Gradient estimation utilities
# ---------------------------------------------------------------------------


def logdensity_estimator(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Build a minibatch estimator of the log-posterior density.

    Returns a function ``(position, minibatch) -> float``.
    """
    return blackjax.sgmcmc.logdensity_estimator(
        logprior_fn, loglikelihood_fn, data_size
    )


def grad_estimator(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Build a minibatch gradient estimator of the log-posterior.

    Returns a function ``(position, minibatch) -> pytree``.
    """
    return blackjax.sgmcmc.grad_estimator(logprior_fn, loglikelihood_fn, data_size)


# ---------------------------------------------------------------------------
# SGLD  –  Stochastic Gradient Langevin Dynamics
# ---------------------------------------------------------------------------


class SGMCMCParams(NamedTuple):
    """Shared step_size/temperature params for SGLD / SGHMC / SGNHT."""

    step_size: Scalar = 1e-3
    temperature: Scalar = 1.0


SGLDParams = SGMCMCParams
SGHMCParams = SGMCMCParams
SGNHTParams = SGMCMCParams


def _make_sgmcmc_kernel(
    grad_estimator_fn: Callable,
    build_kernel_fn: Callable,
    blackjax_init: Callable | None = None,
    state_init: Callable | None = None,
    extra_step_kwargs: dict[str, Any] | None = None,
) -> MarkovKernel:
    """Shared constructor for the SGLD/SGHMC/SGNHT MarkovKernels.

    ``state_init(rng, position)`` defaults to wrapping
    ``blackjax_init(position)`` in :class:`SGMCMCState`; SGNHT supplies its
    own (momentum-carrying) version. Raw (non-``SGMCMCState``) states pass
    through ``step`` unwrapped, preserving SGNHT behavior.
    """
    kernel = build_kernel_fn()
    extra_step_kwargs = extra_step_kwargs or {}

    if state_init is None:
        if blackjax_init is None:
            raise ValueError("Either blackjax_init or state_init is required.")

        def default_state_init(rng, position):
            return SGMCMCState(position=blackjax_init(position))

        state_init = default_state_init

    def init(rng: RngKey, position: Any):
        return state_init(rng, position)

    def step(rng, state, params, *args):
        position = state.position if isinstance(state, SGMCMCState) else state
        new_position = kernel(
            rng,
            position,
            grad_estimator_fn,
            minibatch=args[0] if args else None,
            step_size=params.step_size,
            temperature=params.temperature,
            **extra_step_kwargs,
        )
        if isinstance(state, SGMCMCState):
            return SGMCMCState(position=new_position), SGMCMCInfo()
        return new_position, SGMCMCInfo()

    return MarkovKernel(
        init,
        step,
        lambda state, step_size=1e-3, temperature=1.0: SGMCMCParams(
            step_size=step_size, temperature=temperature
        ),
    )


def sgld(grad_estimator: Callable, temperature: float = 1.0) -> MarkovKernel:
    """Stochastic Gradient Langevin Dynamics.

    Args:
        grad_estimator: Function ``(position, minibatch) -> pytree`` that
            estimates the gradient of the log-posterior.
        temperature: Temperature parameter (default 1.0).
    """
    kernel = _make_sgmcmc_kernel(
        grad_estimator,
        blackjax.sgld.build_kernel,
        blackjax_init=blackjax.sgld.init,
    )
    return kernel


# ---------------------------------------------------------------------------
# SGHMC  –  Stochastic Gradient Hamiltonian Monte Carlo
# ---------------------------------------------------------------------------


def sghmc(
    grad_estimator: Callable,
    num_integration_steps: int = 10,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
) -> MarkovKernel:
    """Stochastic Gradient Hamiltonian Monte Carlo.

    Args:
        grad_estimator: Function ``(position, minibatch) -> pytree`` that
            estimates the gradient of the log-posterior.
        num_integration_steps: Number of leapfrog steps per sample.
        alpha: Friction coefficient.
        beta: Noise scaling.
        temperature: Temperature parameter.
    """
    return _make_sgmcmc_kernel(
        grad_estimator,
        partial(blackjax.sghmc.build_kernel, alpha=alpha, beta=beta),
        blackjax_init=blackjax.sghmc.init,
        extra_step_kwargs={"num_integration_steps": num_integration_steps},
    )


# ---------------------------------------------------------------------------
# SGNHT  –  Stochastic Gradient Nosé-Hoover Thermostat
# ---------------------------------------------------------------------------


def _sgnht_init(position, rng, alpha: float = 0.01) -> SGNHTState:

    momentum = generate_gaussian_noise(rng, position)
    return SGNHTState(position, momentum, alpha)


def sgnht(
    grad_estimator: Callable,
    alpha: float = 0.01,
    beta: float = 0.0,
    temperature: float = 1.0,
) -> MarkovKernel:
    """Stochastic Gradient Nosé-Hoover Thermostat.

    SGNHT extends SGHMC with a thermostat variable that automatically
    adjusts the kinetic energy to maintain the target temperature.

    Args:
        grad_estimator: Function ``(position, minibatch) -> pytree`` that
            estimates the gradient of the log-posterior.
        alpha: Friction coefficient.
        beta: Noise scaling.
        temperature: Temperature parameter.

    Note:
        ``init`` requires the rng to initialise momentum::

            state = sampler.init(key, position)
    """
    kernel = blackjax.sgnht.build_kernel(alpha=alpha, beta=beta)

    def sgnht_state_init(rng: RngKey, position: Any):
        return _sgnht_init(position, rng=rng, alpha=alpha)

    return _make_sgmcmc_kernel(
        grad_estimator,
        lambda: kernel,
        state_init=sgnht_state_init,
    )
