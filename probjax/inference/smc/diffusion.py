"""Twisted SMC along a reverse diffusion.

Particles start from the diffusion's base distribution at ``times[0]`` and
follow its reverse SDE down to ``times[-1]`` with Euler--Maruyama steps

    x <- x - drift(t, x) h + diffusion(t, x) sqrt(h) eps.

A twist ``log psi_t(x)`` looks ahead to the target: the particles are
weighted by its increase, ``log psi_t'(x') - log psi_t(x)``, and resampled
when the ESS falls below a threshold. A guide may also steer the proposal up
the twist, adding ``diffusion^2 * guidance`` to the step; the weights then
carry the ratio of the reverse process's transition density to the steered
one. At the last time the twist is the exact target, so any twist and any
guidance give a consistent sampler; good ones only need fewer particles.

A :class:`Guide` supplies the reverse SDE and the twist. Two are provided:

- :func:`~probjax.inference.neural.diffusion_guide` steers a pretrained
  diffusion model towards ``prior(x) * exp(V(x))``, twisting and steering by
  ``V`` at the model's denoised estimate (twisted diffusion sampler, Wu et
  al. 2023);
- :func:`score_guide` needs no training: it estimates the score and the
  marginal of an unnormalised target by importance sampling (reverse diffusion
  SMC, Wu et al. 2025).
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from probjax.inference.base import Kernel
from probjax.inference.particles import take_batch
from probjax.inference.smc.ess import log_ess
from probjax.inference.smc.resampling import systematic
from probjax.utils.typing import Array, PyTree

__all__ = [
    "DiffusionSMCInfo",
    "DiffusionSMCState",
    "Guide",
    "diffusion_smc",
    "score_guide",
]

_LOG_2PI = jnp.log(2.0 * jnp.pi)


class Guide(NamedTuple):
    """The reverse diffusion and twist :func:`diffusion_smc` follows.

    All functions act on a batch of particles and take explicit ``params``.

    Attributes:
        sample_base: ``(key, t, num_particles, params) -> (particles,
            log_weights)``, the starting particles at time ``t`` and their log
            weights relative to the reverse process (zero when the base is the
            process's own start).
        evaluate: ``(key, t, particles, params) -> (log_twist, drift,
            diffusion, guidance)`` at time ``t``. ``guidance`` steers the
            proposal (usually the gradient of ``log_twist``); zeros follow
            the reverse process unchanged.
        log_target: ``(particles, params) -> log target``, the exact twist at
            the last time.
        log_transition: ``(t, particles, t_next, next_particles, params) ->
            log density`` of the reference transition the twist is relative
            to. ``None`` (the default) refers the twist to the reverse SDE
            itself, as for a pretrained model; a guide whose twist estimates
            the noised target's marginal passes the forward noising kernel
            from ``t_next`` back to ``t`` instead.
    """

    sample_base: Callable
    evaluate: Callable
    log_target: Callable
    log_transition: Callable | None = None


class DiffusionSMCState(NamedTuple):
    """A weighted population at one time of the reverse diffusion.

    Attributes:
        particles: pytree with a leading particle axis.
        log_weights: normalised log weights.
        log_twist: twist of each particle at the current time.
        drift, diffusion: the reverse SDE at each particle.
        guidance: the proposal's steer at each particle.
        time_index: index of the current time in ``times``.
        log_evidence: running estimate of ``log Z``.
    """

    particles: PyTree[Array]
    log_weights: Array
    log_twist: Array
    drift: PyTree[Array]
    diffusion: PyTree[Array]
    guidance: PyTree[Array]
    time_index: Array
    log_evidence: Array


class DiffusionSMCInfo(NamedTuple):
    """Diagnostics of one reverse step.

    Attributes:
        ess: effective sample size after reweighting, as a fraction.
        resampled: whether the population was resampled.
        ancestors: index of each particle's parent.
        log_evidence_increment: this step's factor of the evidence estimate.
    """

    ess: Array
    resampled: Array
    ancestors: Array
    log_evidence_increment: Array


def diffusion_smc(guide: Guide, times: Array, *, ess_threshold: float = 0.5) -> Kernel:
    """Twisted SMC that follows ``guide``'s reverse diffusion through ``times``.

    Run it for ``len(times) - 1`` steps; further steps leave the state
    unchanged.

    Args:
        guide: the reverse SDE and twist, e.g. from
            :func:`~probjax.inference.neural.diffusion_guide` or
            :func:`score_guide`.
        times: the time grid; sorted from the noisy end to the clean one.
        ess_threshold: resample when the ESS fraction falls below this.

    Returns:
        A :class:`Kernel` with ``init(key, num_particles, params=None)``; its
        ``params`` are the guide's.
    """
    if not 0.0 < ess_threshold <= 1.0:
        raise ValueError("ess_threshold must be in (0, 1].")
    times = jnp.sort(jnp.asarray(times))[::-1]
    if times.shape[0] < 2:
        raise ValueError("times needs at least two points.")
    last = times.shape[0] - 1

    def evaluate(key, t, particles, params):
        log_twist, *fields = guide.evaluate(key, t, particles, params)
        # Schedules may return a diffusion shared by all particles; give every
        # particle its own so resampling can gather it.
        drift, diffusion, guidance = (
            jax.tree.map(lambda v, x: jnp.broadcast_to(v, x.shape), value, particles)
            for value in fields
        )
        return log_twist, drift, diffusion, guidance

    def init(key: Any, num_particles: int, params: Any = None) -> DiffusionSMCState:
        base_key, evaluate_key = jax.random.split(key)
        particles, log_weights = guide.sample_base(
            base_key, times[0], num_particles, params
        )
        log_twist, drift, diffusion, guidance = evaluate(
            evaluate_key, times[0], particles, params
        )
        log_weights = log_weights + log_twist
        log_norm = logsumexp(log_weights)
        return DiffusionSMCState(
            particles,
            log_weights - log_norm,
            log_twist,
            drift,
            diffusion,
            guidance,
            jnp.zeros((), jnp.int32),
            log_norm - jnp.log(num_particles),
        )

    def advance(key, state: DiffusionSMCState, params: Any):
        noise_key, evaluate_key, resample_key = jax.random.split(key, 3)
        index = state.time_index
        step_size = times[index] - times[index + 1]
        leaves, treedef = jax.tree.flatten(state.particles)
        noise_keys = jax.random.split(noise_key, len(leaves))
        noise = treedef.unflatten([
            jax.random.normal(k, leaf.shape, leaf.dtype)
            for k, leaf in zip(noise_keys, leaves, strict=True)
        ])
        # The steered step shifts the mean by g^2 u h; the log ratio of the
        # unsteered to the steered Gaussian transition at the drawn point is
        # -(g sqrt(h) eps u + g^2 u^2 h / 2), summed over coordinates.
        particles = jax.tree.map(
            lambda x, drift, g, u, eps: (
                x
                - drift * step_size
                + g**2 * u * step_size
                + g * jnp.sqrt(step_size) * eps
            ),
            state.particles,
            state.drift,
            state.diffusion,
            state.guidance,
            noise,
        )
        if guide.log_transition is None:
            log_transition_ratio = _per_particle_sum(
                jax.tree.map(
                    lambda g, u, eps: (
                        -g * jnp.sqrt(step_size) * eps * u
                        - 0.5 * (g * u) ** 2 * step_size
                    ),
                    state.diffusion,
                    state.guidance,
                    noise,
                )
            )
        else:
            log_proposal = _per_particle_sum(
                jax.tree.map(
                    lambda g, eps: (
                        -0.5 * eps**2
                        - jnp.log(g * jnp.sqrt(step_size))
                        - 0.5 * _LOG_2PI
                    ),
                    state.diffusion,
                    noise,
                )
            )
            log_reference = guide.log_transition(
                times[index], state.particles, times[index + 1], particles, params
            )
            log_transition_ratio = log_reference - log_proposal

        def terminal(_):
            # The last twist is the target itself, which keeps the sampler exact.
            zeros = jax.tree.map(
                jnp.zeros_like, (state.drift, state.diffusion, state.guidance)
            )
            return (guide.log_target(particles, params), *zeros)

        def intermediate(key):
            return evaluate(key, times[index + 1], particles, params)

        log_twist, drift, diffusion, guidance = jax.lax.cond(
            index + 1 == last, terminal, intermediate, evaluate_key
        )

        log_increments = jnp.nan_to_num(
            log_twist - state.log_twist + log_transition_ratio, nan=-jnp.inf
        )
        increment = logsumexp(state.log_weights + log_increments)
        log_weights = state.log_weights + log_increments - increment
        num_particles = log_weights.shape[0]
        ess = jnp.exp(log_ess(log_weights)) / num_particles
        resampled = ess < ess_threshold
        ancestors = jnp.where(
            resampled,
            systematic(resample_key, jnp.exp(log_weights), num_particles),
            jnp.arange(num_particles),
        )
        particles, log_twist, drift, diffusion, guidance = take_batch(
            (particles, log_twist, drift, diffusion, guidance), ancestors
        )
        log_weights = jnp.where(resampled, -jnp.log(num_particles), log_weights)
        state = DiffusionSMCState(
            particles,
            log_weights,
            log_twist,
            drift,
            diffusion,
            guidance,
            index + 1,
            state.log_evidence + increment,
        )
        return state, DiffusionSMCInfo(ess, resampled, ancestors, increment)

    def step(key, state: DiffusionSMCState, params: Any, *args: Any):
        def finished(_):
            num_particles = state.log_weights.shape[0]
            info = DiffusionSMCInfo(
                jnp.exp(log_ess(state.log_weights)) / num_particles,
                jnp.zeros((), bool),
                jnp.arange(num_particles),
                jnp.zeros(()),
            )
            return state, info

        return jax.lax.cond(
            state.time_index >= last,
            finished,
            lambda key: advance(key, state, params),
            key,
        )

    return Kernel(init, step, lambda state, params=None: params)


def _per_particle_sum(tree: PyTree[Array]) -> Array:
    return sum(
        jnp.sum(leaf.reshape(leaf.shape[0], -1), axis=1)
        for leaf in jax.tree.leaves(tree)
    )


def _gaussian_logpdf(x: Array, mean: Array, std: Array) -> Array:
    return jnp.sum(-0.5 * ((x - mean) / std) ** 2 - jnp.log(std) - 0.5 * _LOG_2PI)


def _score_and_marginal(
    key: Any,
    x_t: Array,
    log_target_fn: Callable,
    alpha: Array,
    sigma: Array,
    num_inner: int,
    max_score_norm: float | None,
) -> tuple[Array, Array]:
    """Importance-sampling estimates of the score and the marginal at ``x_t``.

    With the forward noising ``x_t = alpha x_0 + sigma eps``, the score is the
    posterior mean ``E[(alpha x_0 - x_t) / sigma^2 | x_t]`` (denoising score
    identity). Proposing ``x_0 ~ N(x_t / alpha, (sigma / alpha)^2)`` estimates
    it, and the mean importance weight is an unbiased estimate of the marginal
    density of ``x_t`` up to the target's normaliser.
    """
    proposal_std = sigma / alpha
    proposal_mean = x_t / alpha
    inner = proposal_mean + proposal_std * jax.random.normal(
        key, (num_inner,) + x_t.shape, x_t.dtype
    )

    def log_weight(x_0):
        return (
            log_target_fn(x_0)
            + _gaussian_logpdf(x_t, alpha * x_0, sigma)
            - _gaussian_logpdf(x_0, proposal_mean, proposal_std)
        )

    log_weights = jax.vmap(log_weight)(inner)
    log_marginal = logsumexp(log_weights) - jnp.log(num_inner)
    weights = jax.nn.softmax(log_weights).reshape((num_inner,) + (1,) * x_t.ndim)
    score = jnp.sum(weights * (alpha * inner - x_t), axis=0) / sigma**2
    if max_score_norm is not None:
        norm = jnp.linalg.norm(score)
        score = score * max_score_norm / jnp.maximum(norm, max_score_norm)
    return score, log_marginal


def score_guide(
    log_target_fn: Callable[[Array], Array],
    schedule: Any,
    event_shape: tuple[int, ...],
    *,
    num_inner: int = 64,
    max_score_norm: float | None = 50.0,
) -> Guide:
    """A training-free guide for an unnormalised target (reverse diffusion SMC).

    At every particle it estimates the score with ``num_inner`` importance
    samples of the target, and uses the same samples' marginal estimate as the
    twist, relative to the forward noising kernel. That costs ``num_inner``
    target evaluations per particle and step.

    Args:
        log_target_fn: unnormalised log density of one point.
        schedule: the forward noising, with ``scale(t)``, ``std(t)``,
            ``drift(t, x)`` and ``diffusion(t, x)``, e.g.
            :class:`~probjax.nn.generative.diffusion.config.VPNoiseSchedule`.
        event_shape: shape of one point.
        num_inner: importance samples per score estimate.
        max_score_norm: clip the estimated score to this norm; noisy estimates
            at small times can otherwise throw particles far away.

    Returns:
        A :class:`Guide` without params; start from a standard normal at a
        time where the noised target is close to one.
    """

    def sample_base(key, t, num_particles, params):
        del t, params
        particles = jax.random.normal(key, (num_particles,) + tuple(event_shape))
        # The base is a proposal for the noised target, which the twist
        # carries in full, so the weights correct for the base density.
        log_weights = -jax.vmap(
            lambda x: _gaussian_logpdf(x, jnp.zeros_like(x), jnp.ones(()))
        )(particles)
        return particles, log_weights

    def evaluate(key, t, particles, params):
        del params
        alpha, sigma = schedule.scale(t), schedule.std(t)
        keys = jax.random.split(key, particles.shape[0])
        score, log_marginal = jax.vmap(
            lambda k, x: _score_and_marginal(
                k, x, log_target_fn, alpha, sigma, num_inner, max_score_norm
            )
        )(keys, particles)
        diffusion = schedule.diffusion(t, particles)
        drift = schedule.drift(t, particles) - diffusion**2 * score
        return log_marginal, drift, diffusion, jnp.zeros_like(particles)

    def log_target(particles, params):
        del params
        return jax.vmap(log_target_fn)(particles)

    def log_transition(t, particles, t_next, next_particles, params):
        # The forward noising from t_next to t: x_t = a x_next + s eps.
        del params
        a = schedule.scale(t) / schedule.scale(t_next)
        s = jnp.sqrt(schedule.std(t) ** 2 - (a * schedule.std(t_next)) ** 2)
        return jax.vmap(_gaussian_logpdf, in_axes=(0, 0, None))(
            particles, a * next_particles, s
        )

    return Guide(sample_base, evaluate, log_target, log_transition)
