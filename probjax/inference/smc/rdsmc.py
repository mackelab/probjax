"""Reverse-diffusion SMC (RDSMC): training-free sampling from unnormalized targets.

Implements the NeurIPS 2025 method of Wu et al. ("Reverse Diffusion
Sequential Monte Carlo Samplers"): a reverse diffusion process with
Monte-Carlo score estimates serves as the SMC proposal, while intermediate
targets built from marginal-estimate byproducts correct discretization and
score-estimation bias. The final target is exact by construction and the
evidence estimate is unbiased for any particle count.

Per reverse step ``t_cur -> t_next`` (``h > 0``) each particle:

1. proposes ``x_new = x - (f - g^2 s) h + g sqrt(h) eps`` with the DSI
   score ``s`` estimated at ``x`` (:mod:`score_estimation`);
2. re-estimates ``(s_new, log pi_hat_new)`` at ``x_new`` -- one estimator
   call per step, whose marginal byproduct is the twist target;
3. weights by the twist ratio ``pi_hat_new - pi_hat_old``, accumulates
   evidence, and resamples when relative ESS drops below threshold.

After the last transition an exact correction replaces the ``t_min``
marginal estimate with ``log_target_fn(x_0)``. Cost is
``N * T * M`` target evaluations for ``N`` particles, ``T`` steps and
``M`` inner importance samples.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from probjax.inference.base import SMCResult
from probjax.inference.particles import normalize_log_weights, take_batch
from probjax.inference.smc._utils import (
    log_weighted_mean_exp,
    random_like,
)
from probjax.inference.smc.ess import ess
from probjax.inference.smc.resampling import systematic
from probjax.nn.export import normalize_spec
from probjax.nn.generative.diffusion.config import (
    BaseSolverConfig,
    VPNoiseSchedule,
)
from probjax.nn.generative.diffusion.score_estimation import is_score_and_marginal
from probjax.nn.generative.sampling import sample_normal
from probjax.utils.typing import Array, PyTree, RngKey

_LOG_2PI = jnp.log(2.0 * jnp.pi)


class RDSMCState(NamedTuple):
    particles: PyTree[Array]
    log_weights: Array  # normalized, shape (num_particles,)


class RDSMCInfo(NamedTuple):
    ess: Array  # (num_transitions,) ESS before resampling each step
    resampled: Array  # (num_transitions,) bool
    log_z_increments: Array  # (num_transitions,) evidence increments
    inner_ess: Array  # (num_transitions + 1,) mean inner ESS per estimator call
    ancestors: Any  # (num_transitions, num_particles) resample indices
    final_log_z_increment: Array  # exact-correction evidence increment


def _standard_normal_logpdf(batch: PyTree[Array]) -> Array:
    """Summed log ``N(0, I)`` over event dims, result ``(num_particles,)``."""
    total: Array | None = None
    for leaf in jax.tree.leaves(batch):
        term = -0.5 * leaf**2 - 0.5 * _LOG_2PI
        reduced = (
            jnp.sum(term, axis=tuple(range(1, term.ndim))) if term.ndim > 1 else term
        )
        total = reduced if total is None else total + reduced
    if total is None:
        raise ValueError("A particle batch must have at least one leaf.")
    return total


def run_rdsmc(
    key: RngKey,
    *,
    log_target_fn: Callable[[PyTree[Array]], Array],
    event_spec,
    num_particles: int = 256,
    num_steps: int = 32,
    num_inner: int = 64,
    schedule=None,
    t_min: float = 1e-3,
    t_max: float = 1.0,
    max_score_norm: float | None = 50.0,
    ess_threshold: float = 0.5,
) -> SMCResult:
    """Sample an unnormalized target with reverse-diffusion SMC (no training).

    Args:
        key: RNG key.
        log_target_fn: single-particle unnormalized ``log pi(x_0)``.
        event_spec: single-event shape/spec; base noise is standard normal.
        num_particles: outer particle count ``N``.
        num_steps: reverse time steps ``T``.
        num_inner: inner importance samples ``M`` per score estimate.
        schedule: forward :class:`NoiseScheduleProtocol` (default VP).
            Only ``scale/std/drift/diffusion`` are used.
        t_min/t_max: reverse time range. ``t_min`` must stay slightly
            above zero so the denoising posterior is non-degenerate.
        max_score_norm: clip for MC score norms (paper B.3.2).
        ess_threshold: resample when ``ESS / N`` falls below this.

    Returns:
        :class:`SMCResult` with ``state`` an :class:`RDSMCState` of
        approximate target samples and ``log_evidence`` an unbiased
        estimate of ``log Z``.
    """
    if num_particles < 1:
        raise ValueError("num_particles must be positive.")
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if num_inner < 1:
        raise ValueError("num_inner must be positive.")
    if not 0.0 < ess_threshold <= 1.0:
        raise ValueError("ess_threshold must be in (0, 1].")
    if not 0.0 < t_min < t_max:
        raise ValueError("require 0 < t_min < t_max.")

    if schedule is None:
        schedule = VPNoiseSchedule(t_min=t_min, t_max=t_max)
    solver = BaseSolverConfig(schedule=schedule, num_steps=num_steps)
    ts = jnp.asarray(solver.solve_schedule(t_min=t_min, t_max=t_max))
    if ts.ndim != 1 or ts.shape[0] < 2:
        raise ValueError("solve_schedule must return at least 2 time points.")
    if ts[1] > ts[0]:
        ts = ts[::-1]

    spec = normalize_spec(event_spec)
    potential_batch = jax.vmap(log_target_fn)

    def estimate_batch(est_keys: Array, t: Array, batch: PyTree[Array]):
        def _single(k: Array, x: PyTree[Array]):
            return is_score_and_marginal(
                k,
                x,
                t,
                log_target_fn=log_target_fn,
                alpha=schedule.scale(t),
                sigma=schedule.std(t),
                num_inner=num_inner,
                max_score_norm=max_score_norm,
            )

        return jax.vmap(_single)(est_keys, batch)

    key, key_init, key_est = jax.random.split(key, 3)
    particles = sample_normal(key_init, (num_particles,), spec)
    est_keys = jax.random.split(key_est, num_particles)
    scores, log_psi, est_info = estimate_batch(est_keys, ts[0], particles)
    inner_ess_trace = [jnp.mean(est_info.ess)]

    # Initial weight pi_hat_T / q with standard-normal base q.
    log_unnorm = log_psi - _standard_normal_logpdf(particles)
    log_z = jax.scipy.special.logsumexp(log_unnorm) - jnp.log(num_particles)
    log_weights = normalize_log_weights(log_unnorm)

    ess_trace, resampled_trace, increment_trace, ancestor_trace = [], [], [], []
    n_transitions = ts.shape[0] - 1
    for i in range(n_transitions):
        t_cur, t_next = ts[i], ts[i + 1]
        h = float(t_cur - t_next)
        if h <= 0:
            raise ValueError("time grid must strictly decrease.")
        key, key_noise, key_est, key_resample = jax.random.split(key, 4)

        fwd_drift = schedule.drift(t_cur, particles)
        diffusion = schedule.diffusion(t_cur, particles)
        rev_drift = jax.tree.map(
            lambda f, g, s: f - g**2 * s, fwd_drift, diffusion, scores
        )
        noise = random_like(key_noise, particles)
        scale = jnp.sqrt(h)

        def _em_step(x, d, g, e, h=h, scale=scale):
            return x - d * h + g * scale * e

        particles_new = jax.tree.map(_em_step, particles, rev_drift, diffusion, noise)

        est_keys = jax.random.split(key_est, num_particles)
        scores_new, log_psi_new, est_info = estimate_batch(
            est_keys, t_next, particles_new
        )
        inner_ess_trace.append(jnp.mean(est_info.ess))

        log_unnorm = log_weights + (log_psi_new - log_psi)
        increment = log_weighted_mean_exp(log_weights, log_psi_new - log_psi)
        log_z = log_z + increment
        log_weights = normalize_log_weights(log_unnorm)

        current_ess = ess(log_weights)
        do_resample = bool(current_ess < ess_threshold * num_particles)
        if num_particles == 1:
            do_resample = False
        if do_resample:
            idx = systematic(key_resample, jnp.exp(log_weights), num_particles).astype(
                jnp.int32
            )
            particles = take_batch(particles_new, idx)
            scores = take_batch(scores_new, idx)
            log_psi = log_psi_new[idx]
            log_weights = jnp.full((num_particles,), -jnp.log(num_particles))
        else:
            particles = particles_new
            scores = scores_new
            log_psi = log_psi_new
            idx = jnp.arange(num_particles, dtype=jnp.int32)

        ess_trace.append(current_ess)
        resampled_trace.append(do_resample)
        increment_trace.append(increment)
        ancestor_trace.append(idx)

    # Exact terminal correction: replace the t_min estimate with pi_tilde(x_0).
    log_exact = jnp.asarray(potential_batch(particles), dtype=jnp.float32)
    final_increment = log_weighted_mean_exp(log_weights, log_exact - log_psi)
    log_z = log_z + final_increment
    log_weights = normalize_log_weights(log_weights + (log_exact - log_psi))

    state = RDSMCState(particles=particles, log_weights=log_weights)
    info = RDSMCInfo(
        ess=jnp.asarray(ess_trace),
        resampled=jnp.asarray(resampled_trace),
        log_z_increments=jnp.asarray(increment_trace),
        inner_ess=jnp.asarray(inner_ess_trace),
        ancestors=jnp.stack(ancestor_trace),
        final_log_z_increment=final_increment,
    )
    return SMCResult(
        state=state,
        params=None,
        info=info,
        log_evidence=log_z,
        final_info=None,
        num_steps=n_transitions,
        completed=True,
        key=key,
    )


__all__ = ["RDSMCState", "RDSMCInfo", "run_rdsmc"]
