"""Pseudo-Marginal MCMC.

Wraps any probjax MCMC kernel to work with a stochastic (unbiased)
log-density estimator ``stochastic_logdensity_fn(position, rng_key) -> float``.

At each step a fresh PRNG key is drawn and held fixed for the duration of
the inner kernel step, making the noisy log-density deterministic within
one transition.  Between steps the key changes.  This is the standard
pseudo-marginal construction (Andrieu & Roberts, 2009) and targets the
correct posterior.

For gradient-based inner kernels (HMC, MALA, NUTS) the noisy log-density
is fixed throughout the leapfrog trajectory so the Hamiltonian is
consistent.  The MH accept/reject at the end of the trajectory uses the
same noisy Hamiltonian for both endpoints, preserving detailed balance.

Variance reduction is available via ``num_samples``: multiple keys are
drawn, the stochastic log-density is evaluated in parallel via ``vmap``,
and the results are averaged.

Usage::

    from probjax.inference.mcmc import hmc
    from probjax.inference.mcmc.pmmcmc import pseudo_marginal

    def stochastic_logdensity(position, rng_key):
        # e.g. run a particle filter / importance sampler
        return unbiased_log_marginal_likelihood_estimate

    kernel = pseudo_marginal(hmc, stochastic_logdensity, num_integration_steps=10)
    state = kernel.init(position, rng_key=key)
    params = kernel.init_params(state)
    new_state, info = kernel.step(key, state, params)
"""

from typing import Callable

import jax
import jax.numpy as jnp

from probjax.inference.mcmc.base import MarkovKernel
from probjax.utils.typing import RngKey


def _make_fixed_logdensity(
    stochastic_logdensity_fn: Callable,
    key_logdensity: RngKey,
    num_samples: int,
) -> Callable:
    """Build a deterministic ``logdensity_fn(position) -> float`` by fixing
    the auxiliary randomness to *key_logdensity*.

    When *num_samples* > 1, ``num_samples`` sub-keys are drawn, the
    stochastic estimator is evaluated in parallel, and the results are
    averaged (variance reduction).
    """
    if num_samples == 1:

        def fixed_logdensity(position):
            return stochastic_logdensity_fn(position, key_logdensity)

    else:

        def fixed_logdensity(position):
            keys = jax.random.split(key_logdensity, num_samples)
            estimates = jax.vmap(stochastic_logdensity_fn, in_axes=(None, 0))(
                position, keys
            )
            return jnp.mean(estimates)

    return fixed_logdensity


def pseudo_marginal(
    inner_kernel_cls,
    stochastic_logdensity_fn: Callable,
    num_samples: int = 1,
    **inner_kernel_kwargs,
) -> MarkovKernel:
    """Wrap a probjax MCMC kernel for pseudo-marginal inference.

    Args:
        inner_kernel_cls: A probjax kernel class (e.g. ``hmc``, ``mala``,
            ``gauss_rwmh``) that exposes ``build_step``, ``init``, and
            ``init_params``.
        stochastic_logdensity_fn: A callable
            ``(position, rng_key) -> float`` that returns an unbiased
            estimate of the log-density (or log-marginal-likelihood)
            given auxiliary randomness from *rng_key*.
        num_samples: Number of independent keys to evaluate and average
            for variance reduction (default 1).
        **inner_kernel_kwargs: Forwarded to
            ``inner_kernel_cls.build_step`` (e.g. ``num_integration_steps``
            for HMC).

    Returns:
        A :class:`MarkovKernel` whose ``step`` splits the PRNG key,
        fixes one sub-key for the stochastic log-density, and delegates
        the transition to the inner kernel.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    # ------------------------------------------------------------------
    # init: evaluate the stochastic logdensity once to populate state
    # ------------------------------------------------------------------
    def init(key, position=None, rng_key=None):
        if position is None:
            position, key = key, rng_key
        if key is None:
            key = jax.random.PRNGKey(0)
        key_init, key_logdensity = jax.random.split(key)
        fixed_logdensity = _make_fixed_logdensity(
            stochastic_logdensity_fn, key_logdensity, num_samples
        )
        # The inner kernel's underlying init (e.g. blackjax.hmc.init)
        # takes (position, logdensity_fn) and evaluates at position.
        return inner_kernel_cls.init(
            position, logdensity_fn=fixed_logdensity, rng_key=key_init
        )

    # ------------------------------------------------------------------
    # step: fix randomness for this transition, delegate to inner kernel
    # ------------------------------------------------------------------
    def step(key: RngKey, state, params, *args):
        key_kernel, key_logdensity = jax.random.split(key)
        fixed_logdensity = _make_fixed_logdensity(
            stochastic_logdensity_fn, key_logdensity, num_samples
        )
        # build_step returns a raw step(key, state, params) closure
        inner_step = inner_kernel_cls.build_step(
            fixed_logdensity, **inner_kernel_kwargs
        )
        return inner_step(key_kernel, state, params, *args)

    def init_params(state, *args, **kwargs):
        return inner_kernel_cls.init_params(state, *args, **kwargs)

    return MarkovKernel(init, step, init_params)
