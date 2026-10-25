"""Tempered SMC: a particle population moved from the prior to the posterior.

The targets ``pi_t(x) ∝ prior(x) * exp(sum(t * loglikelihood(x)))`` bridge
the prior (``t = 0``) and the posterior (``t = 1``). Each step raises the
temperature, reweights the particles by the likelihood they gained, resamples
and moves them with MCMC steps that leave the new target invariant. The
log-likelihood may return one term per observation and the temperature may be
a vector of the same shape; a schedule that switches observations on one by
one then adds the data gradually (data tempering).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from probjax.inference.adaptation import compose_adaptors
from probjax.inference.base import Kernel
from probjax.inference.particles import nan_to_neg_inf, take_batch
from probjax.inference.smc.adaptation import acceptance_rate_adaptor, particle_adaptor
from probjax.inference.smc.ess import log_ess
from probjax.inference.smc.resampling import systematic
from probjax.utils.jaxutils import as_configured, leading_dim
from probjax.utils.typing import Array, PyTree

__all__ = ["SMCInfo", "SMCState", "smc"]


class SMCState(NamedTuple):
    """An equally weighted population at one temperature.

    Attributes:
        particles: pytree with a leading particle axis.
        temperature: the current temperature, a scalar or one per observation.
        log_evidence: running estimate of ``log Z`` of the current target.
    """

    particles: PyTree[Array]
    temperature: Array
    log_evidence: Array


class SMCInfo(NamedTuple):
    """Diagnostics of one SMC step.

    Attributes:
        ess: effective sample size of the reweighted population, as a fraction
            of the particles.
        log_evidence_increment: this step's factor of the evidence estimate.
        ancestors: index of each particle's parent.
        acceptance_rate: mean acceptance rate of the MCMC moves (NaN if the
            move reports none).
    """

    ess: Array
    log_evidence_increment: Array
    ancestors: Array
    acceptance_rate: Array


def _solve_temperature(loglik: Array, temperature: Array, target_ess: float) -> Array:
    """The largest temperature, at most 1, whose reweighting keeps ``target_ess``.

    The ESS falls monotonically as the temperature rises, so bisection on the
    increment converges.
    """

    def relative_ess(delta):
        log_weights = nan_to_neg_inf(delta * loglik)
        return jnp.exp(log_ess(log_weights)) / loglik.shape[0]

    def bisect(_, bounds):
        low, high = bounds
        middle = 0.5 * (low + high)
        enough = relative_ess(middle) >= target_ess
        return jnp.where(enough, middle, low), jnp.where(enough, high, middle)

    remaining = 1.0 - temperature
    low, _ = jax.lax.fori_loop(0, 50, bisect, (jnp.zeros_like(remaining), remaining))
    return jnp.where(relative_ess(remaining) >= target_ess, 1.0, temperature + low)


def smc(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    *,
    move: Any,
    num_mcmc_steps: int = 10,
    target_ess: float | None = None,
    waste_free: int | None = None,
    batch_size: int = 0,
) -> Kernel:
    """Tempered SMC with MCMC moves.

    With ``target_ess=None`` the schedule is fixed and each step takes the next
    temperature as its argument, so ``run`` is called with
    ``args=(schedule,)``. With a ``target_ess`` each step picks the largest
    temperature that keeps that fraction of effective particles; once the
    temperature reaches 1 further steps leave the state unchanged, so
    ``num_steps`` is a cap.

    Args:
        logprior_fn: Log prior of one particle.
        loglikelihood_fn: Log likelihood of one particle; a scalar, or one term
            per observation for data tempering.
        move: MCMC kernel moving the particles, an API class such as ``hmc``
            or a configured one such as ``hmc(num_integration_steps=5)``. The
            SMC kernel's ``params`` are this kernel's params, shared by all
            particles.
        num_mcmc_steps: MCMC steps per particle and SMC step.
        target_ess: Fraction of effective particles each adaptive step keeps.
        waste_free: Keep every state of the move chains instead of only the
            last (Dau and Chopin, 2022): each step resamples
            ``num_particles // waste_free`` particles and runs
            ``waste_free - 1`` moves from each. Replaces ``num_mcmc_steps``.
        batch_size: Move this many particles at a time to bound memory;
            ``0`` moves them all at once.

    Returns:
        A :class:`Kernel` with ``init(key, particles, temperature=0.0)``,
        ``init_params(state, **move_params)`` and a ``default_adaptation``
        that fits the move's geometry to the population.
    """
    move = as_configured(move)
    if waste_free is not None and waste_free < 2:
        raise ValueError("waste_free must keep at least two states per chain.")

    def tempered(temperature: Array) -> Callable:
        def logdensity(x: Any) -> Array:
            return logprior_fn(x) + jnp.sum(temperature * loglikelihood_fn(x))

        return logdensity

    def init(key: Any, particles: PyTree[Array], temperature: Any = 0.0) -> SMCState:
        del key  # the population is given; the signature matches every kernel
        return SMCState(particles, jnp.asarray(temperature, dtype=float), jnp.zeros(()))

    def init_params(state: SMCState, **move_params: Any) -> Any:
        move_kernel = move(tempered(state.temperature))
        first = jax.tree.map(lambda leaf: leaf[0], state.particles)
        move_state = move_kernel.init(jax.random.key(0), first)
        return move_kernel.init_params(move_state, **move_params)

    def chain(key, position, params, temperature, num_steps):
        """Run ``num_steps`` moves from one particle; return all positions."""
        move_kernel = move(tempered(temperature))
        init_key, key = jax.random.split(key)

        def one_move(move_state, step_key):
            move_state, info = move_kernel.step(step_key, move_state, params)
            rate = getattr(info, "acceptance_rate", jnp.nan)
            return move_state, (move_state.position, rate)

        move_state = move_kernel.init(init_key, position)
        keys = jax.random.split(key, num_steps)
        _, (positions, rates) = jax.lax.scan(one_move, move_state, keys)
        return positions, jnp.mean(rates)

    def move_all(key, seeds, params, temperature, num_steps):
        keys = jax.random.split(key, leading_dim(seeds))

        def one(key_and_seed):
            key, seed = key_and_seed
            return chain(key, seed, params, temperature, num_steps)

        if batch_size:
            return jax.lax.map(one, (keys, seeds), batch_size=batch_size)
        return jax.vmap(one)((keys, seeds))

    def advance(key, state, params, temperature, loglik):
        num_particles = loglik.shape[0]
        event_axes = tuple(range(1, loglik.ndim))
        delta = temperature - state.temperature
        log_increments = jnp.sum(delta * loglik, axis=event_axes)
        log_increments = nan_to_neg_inf(log_increments)
        # The population is equally weighted, so the evidence factor is the
        # mean incremental weight.
        log_norm = logsumexp(log_increments)
        log_evidence_increment = log_norm - jnp.log(num_particles)
        log_weights = log_increments - log_norm

        resample_key, move_key = jax.random.split(key)
        weights = jnp.exp(log_weights)
        if waste_free is None:
            ancestors = systematic(resample_key, weights, num_particles)
            seeds = take_batch(state.particles, ancestors)
            positions, rates = move_all(
                move_key, seeds, params, temperature, num_mcmc_steps
            )
            particles = jax.tree.map(lambda leaf: leaf[:, -1], positions)
        else:
            if num_particles % waste_free:
                raise ValueError(
                    f"waste_free={waste_free} must divide the {num_particles} "
                    "particles."
                )
            num_chains = num_particles // waste_free
            parents = systematic(resample_key, weights, num_chains)
            seeds = take_batch(state.particles, parents)
            positions, rates = move_all(
                move_key, seeds, params, temperature, waste_free - 1
            )

            def keep_all(seed, chain_positions):
                stacked = jnp.concatenate([seed[:, None], chain_positions], axis=1)
                return stacked.reshape((num_chains * waste_free,) + seed.shape[1:])

            particles = jax.tree.map(keep_all, seeds, positions)
            ancestors = jnp.repeat(parents, waste_free)

        state = SMCState(
            particles, temperature, state.log_evidence + log_evidence_increment
        )
        info = SMCInfo(
            jnp.exp(log_ess(log_weights)) / num_particles,
            log_evidence_increment,
            ancestors,
            jnp.mean(rates),
        )
        return state, info

    def step(key, state: SMCState, params: Any, *args: Any):
        loglik = jax.vmap(loglikelihood_fn)(state.particles)
        if target_ess is None:
            (temperature,) = args
            temperature = jnp.asarray(temperature, state.temperature.dtype)
            return advance(key, state, params, temperature, loglik)
        if state.temperature.ndim:
            raise ValueError("Adaptive tempering needs a scalar temperature.")
        ess_fraction = float(target_ess)

        def finished(_):
            info = SMCInfo(
                jnp.ones(()),
                jnp.zeros(()),
                jnp.arange(loglik.shape[0]),
                jnp.full((), jnp.nan),
            )
            return state, info

        def tempering(key):
            temperature = _solve_temperature(loglik, state.temperature, ess_fraction)
            return advance(key, state, params, temperature, loglik)

        return jax.lax.cond(state.temperature >= 1.0, finished, tempering, key)

    def default_adaptation():
        # Fit the fields the move has: HMC's mass matrix and step size, say.
        fields = inspect.signature(move.api.init_params).parameters
        adaptors = {}
        if "inverse_mass_matrix" in fields:
            adaptors["geometry"] = particle_adaptor()
        if "step_size" in fields:
            adaptors["step_size"] = acceptance_rate_adaptor()
        if not adaptors:
            raise ValueError(
                f"{move.api.__name__} has no step_size or inverse_mass_matrix "
                "to adapt; pass an adaptor explicitly."
            )
        return compose_adaptors(**adaptors)

    return Kernel(init, step, init_params, default_adaptation)
