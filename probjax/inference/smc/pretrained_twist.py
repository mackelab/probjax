"""Reverse-time twisted SMC with a frozen diffusion model as proposal.

Target: ``pi(x_0) propto p_phi(x_0) * exp(temperature * V(x_0))`` where
``p_phi`` is the prior encoded by a pretrained diffusion model (any
:class:`~probjax.nn.generative.diffusion.config.ScheduleAwareModelProtocol`
implementer, typically :class:`DiffusionDenoiser`) and ``V`` is a tractable
potential on clean data (reward, log-likelihood of observations, ...).

Method (twisted diffusion sampler style): simulate the model's reverse
kernels as proposals over a suffix path prior
``p(x_T) prod_t p(x_t | x_{t+1})`` and target
``gamma_t(x_{t:T}) = p(x_{t:T}) * psi_t(x_t)`` with a twist
``psi_t(x_t) ~= E[exp(V(x_0)) | x_t]``. The default twist evaluates ``V``
on the denoised estimate ``xhat_0(x_t)``. Because the proposal equals the
prior transition, the incremental weight telescopes to a twist ratio::

    log w_t = log w_{t+1} + log psi_t(x_t) - log psi_{t+1}(x_{t+1})

so only twist evaluations (plus one Euler-Maruyama reverse step) are needed
per particle per step -- no forward-kernel or marginal densities. The final
step evaluates ``V`` on clean ``x_0`` directly, keeping the terminal target
exact. Any positive twist preserves correctness; resampling is gated on
relative ESS.

The runner is intentionally a plain Python loop (not a compiled scan) in
v1 so it accepts any duck-typed model object without NNX tracing
concerns. Each model call still executes as XLA on accelerators.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from blackjax.smc.ess import ess
from blackjax.smc.resampling import systematic

from probjax.inference.base import SMCResult
from probjax.inference.particles import normalize_log_weights, take_batch
from probjax.inference.smc._utils import (
    log_weighted_mean_exp,
    random_like,
)
from probjax.nn.export import normalize_spec
from probjax.nn.generative.diffusion.twists import (
    DenoisedPotentialTwist,
    TwistFn,
)
from probjax.utils.typing import Array, PyTree, RngKey


class PretrainedTwistState(NamedTuple):
    particles: PyTree[Array]
    log_weights: Array  # normalized, shape (num_particles,)


class PretrainedTwistInfo(NamedTuple):
    ess: Array  # (num_transitions,) ESS before resampling each step
    resampled: Array  # (num_transitions,) bool
    log_z_increments: Array  # (num_transitions,) evidence increments
    ancestors: Any  # (num_transitions, num_particles) resample indices


def run_pretrained_diffusion_smc(
    key: RngKey,
    *,
    model,
    event_spec=None,
    log_potential_fn: Callable[[PyTree[Array]], Array],
    num_particles: int = 256,
    num_steps: Optional[int] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    temperature: float = 1.0,
    ess_threshold: float = 0.5,
    twist: Optional[TwistFn] = None,
    inputs: Optional[dict] = None,
) -> SMCResult:
    """Run twisted SMC steering a frozen diffusion model toward a potential.

    Args:
        key: RNG key.
        model: diffusion model exposing ``solver_cfg.solve_schedule``,
            ``_build_sde_drift_and_diffusion(inputs)``,
            ``denoise(t, x, **inputs)`` and
            ``_sample_base(rng, sample_shape, spec, t_max=...)`` (i.e. a
            :class:`DiffusionDenoiser` or a duck-typed equivalent).
        event_spec: single-event shape/spec for sampling. Defaults to the
            model's ``event_spec``.
        log_potential_fn: single-particle ``x_0 -> scalar`` potential ``V``.
            The target is ``p_phi(x_0) * exp(temperature * V(x_0))``.
        num_particles: particle count ``N``.
        num_steps: reverse steps; defaults to ``solver_cfg.num_steps``.
        t_min/t_max: time range; default to ``train_cfg`` bounds.
        temperature: scales ``V`` (``0.0`` recovers the prior).
        ess_threshold: resample when ``ESS / N`` falls below this.
        twist: custom :class:`TwistFn`. Defaults to
            :class:`DenoisedPotentialTwist` around ``model.denoise``.
        inputs: named model inputs (``context``, ...) forwarded to
            drift/diffusion/denoise.

    Returns:
        :class:`SMCResult` with ``state`` a :class:`PretrainedTwistState`
        of clean particles and normalized weights, and ``log_evidence``
        estimating ``log E_prior[exp(temperature * V)]``.
    """
    if num_particles < 1:
        raise ValueError("num_particles must be positive.")
    if not 0.0 < ess_threshold <= 1.0:
        raise ValueError("ess_threshold must be in (0, 1].")
    inputs = dict(inputs or {})

    cfg = model.solver_cfg
    if cfg is None:
        raise ValueError("model.solver_cfg is not set.")
    train_cfg = model.train_cfg
    if t_min is None:
        t_min = float(train_cfg.t_min)
    if t_max is None:
        t_max = float(train_cfg.t_max)
    steps = cfg.num_steps if num_steps is None else int(num_steps)
    if steps < 1:
        raise ValueError("num_steps must be positive.")

    ts = jnp.asarray(cfg.solve_schedule(t_min=t_min, t_max=t_max, num_steps=steps))
    if ts.ndim != 1 or ts.shape[0] < 2:
        raise ValueError("solve_schedule must return at least 2 time points.")
    if ts[1] > ts[0]:  # ascend -> flip so we integrate t_max -> t_min
        ts = ts[::-1]

    drift_fn, diffusion_fn = model._build_sde_drift_and_diffusion(inputs)

    def denoise_fn(t: Array, x: PyTree[Array]) -> PyTree[Array]:
        return model.denoise(t, x, **inputs)

    if twist is None:
        twist = DenoisedPotentialTwist(
            denoise_fn=denoise_fn,
            log_potential_fn=log_potential_fn,
            temperature=temperature,
        )
    potential_batch = jax.vmap(log_potential_fn)

    if event_spec is None:
        event_spec = model.event_spec
        if event_spec is None:
            raise TypeError(
                "event_spec is required: pass it explicitly or set a default "
                "on the model via set_event_spec()."
            )
    spec = normalize_spec(event_spec)

    key, key_init = jax.random.split(key)
    particles = model._sample_base(key_init, (num_particles,), spec, t_max=t_max)

    log_psi = jnp.asarray(twist.log_twist(ts[0], particles), dtype=jnp.float32)
    # Initial weight w_T = gamma_T / q = psi_T (proposal equals prior noise).
    log_z = jax.scipy.special.logsumexp(log_psi) - jnp.log(num_particles)
    log_weights = normalize_log_weights(log_psi)

    ess_trace, resampled_trace, increment_trace, ancestor_trace = [], [], [], []

    n_transitions = ts.shape[0] - 1
    for i in range(n_transitions):
        t_cur, t_next = ts[i], ts[i + 1]
        h = float(t_cur - t_next)
        if h <= 0:
            raise ValueError("time grid must strictly decrease.")
        key, key_noise, key_resample = jax.random.split(key, 3)

        drift = drift_fn(t_cur, particles)
        diffusion = diffusion_fn(t_cur, particles)
        noise = random_like(key_noise, particles)
        scale = jnp.sqrt(h)

        def _em_step(x, d, g, e, h=h, scale=scale):
            return x - d * h + g * scale * e

        particles_new = jax.tree.map(
            _em_step,
            particles,
            drift,
            diffusion,
            noise,
        )

        if i == n_transitions - 1:  # exact terminal potential on clean x_0
            log_psi_new = temperature * jnp.asarray(
                potential_batch(particles_new), dtype=jnp.float32
            )
        else:
            log_psi_new = jnp.asarray(
                twist.log_twist(t_next, particles_new), dtype=jnp.float32
            )

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
            log_weights = jnp.full((num_particles,), -jnp.log(num_particles))
            log_psi = log_psi_new[idx]
        else:
            particles = particles_new
            idx = jnp.arange(num_particles, dtype=jnp.int32)
            log_psi = log_psi_new

        ess_trace.append(current_ess)
        resampled_trace.append(do_resample)
        increment_trace.append(increment)
        ancestor_trace.append(idx)

    state = PretrainedTwistState(particles=particles, log_weights=log_weights)
    info = PretrainedTwistInfo(
        ess=jnp.asarray(ess_trace),
        resampled=jnp.asarray(resampled_trace),
        log_z_increments=jnp.asarray(increment_trace),
        ancestors=jnp.stack(ancestor_trace),
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


__all__ = [
    "PretrainedTwistState",
    "PretrainedTwistInfo",
    "run_pretrained_diffusion_smc",
]
