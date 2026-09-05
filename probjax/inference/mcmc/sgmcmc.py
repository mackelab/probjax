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

from typing import Callable, NamedTuple

import blackjax
from blackjax.types import ArrayTree
from blackjax.util import generate_gaussian_noise

from probjax.inference.mcmc.base import MarkovKernel
from probjax.utils.typing import RngKey

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


class SGLDParams(NamedTuple):
    step_size: float = 1e-3
    temperature: float = 1.0


def sgld(grad_estimator: Callable, temperature: float = 1.0) -> MarkovKernel:
    """Stochastic Gradient Langevin Dynamics.

    Args:
        grad_estimator: Function ``(position, minibatch) -> pytree`` that
            estimates the gradient of the log-posterior.
        temperature: Temperature parameter (default 1.0).
    """
    kernel = blackjax.sgld.build_kernel()

    def init(key, position=None, rng_key=None):
        if position is None:
            position = key
        return SGMCMCState(position=blackjax.sgld.init(position))

    def step(key: RngKey, state: SGMCMCState, params, *args):
        new_position = kernel(
            key,
            state.position,
            grad_estimator,
            minibatch=args[0] if args else None,
            step_size=params.step_size,
            temperature=params.temperature,
        )
        return SGMCMCState(position=new_position), SGMCMCInfo()

    return MarkovKernel(
        init,
        step,
        lambda state, step_size=1e-3, temperature=1.0: SGLDParams(
            step_size=step_size, temperature=temperature
        ),
    )


# ---------------------------------------------------------------------------
# SGHMC  –  Stochastic Gradient Hamiltonian Monte Carlo
# ---------------------------------------------------------------------------


class SGHMCParams(NamedTuple):
    step_size: float = 1e-3
    temperature: float = 1.0


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
    kernel = blackjax.sghmc.build_kernel(alpha=alpha, beta=beta)

    def init(key, position=None, rng_key=None):
        if position is None:
            position = key
        return SGMCMCState(position=blackjax.sghmc.init(position))

    def step(key: RngKey, state: SGMCMCState, params, *args):
        new_position = kernel(
            key,
            state.position,
            grad_estimator,
            minibatch=args[0] if args else None,
            step_size=params.step_size,
            temperature=params.temperature,
            num_integration_steps=num_integration_steps,
        )
        return SGMCMCState(position=new_position), SGMCMCInfo()

    return MarkovKernel(
        init,
        step,
        lambda state, step_size=1e-3, temperature=1.0: SGHMCParams(
            step_size=step_size, temperature=temperature
        ),
    )


# ---------------------------------------------------------------------------
# SGNHT  –  Stochastic Gradient Nosé-Hoover Thermostat
# ---------------------------------------------------------------------------


class SGNHTParams(NamedTuple):
    step_size: float = 1e-3
    temperature: float = 1.0


def _sgnht_init(position, rng_key=None, alpha: float = 0.01):
    from blackjax.sgmcmc.sgnht import SGNHTState

    momentum = generate_gaussian_noise(rng_key, position)
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
        ``init`` requires ``rng_key`` to initialise momentum::

            state = sampler.init(position, rng_key=key)
    """
    kernel = blackjax.sgnht.build_kernel(alpha=alpha, beta=beta)

    def step(key: RngKey, state, params, *args):
        new_state = kernel(
            key,
            state,
            grad_estimator,
            minibatch=args[0] if args else None,
            step_size=params.step_size,
            temperature=params.temperature,
        )
        return new_state, SGMCMCInfo()

    def init(key, position=None, rng_key=None):
        if position is None:
            position, key = key, rng_key
        return _sgnht_init(position, rng_key=key, alpha=alpha)

    return MarkovKernel(
        init,
        step,
        lambda state, step_size=1e-3, temperature=1.0: SGNHTParams(
            step_size=step_size, temperature=temperature
        ),
    )
