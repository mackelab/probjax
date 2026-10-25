"""Smoothing, joint path sampling and particle Gibbs from a filter's output.

Run a filter with ``collect=("state", "info")`` to keep its states and infos;
these functions take them together with the initial state, so their results
cover the initial time and every filtered time.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from probjax.inference.filtering.observations import validate_observations
from probjax.inference.filtering.state_space import LinearGaussianModel
from probjax.inference.particles import normalize_log_weights
from probjax.utils.jaxutils import prepend_to_trace
from probjax.utils.linalg import symmetrize_matrix
from probjax.utils.linear_operator import as_array
from probjax.utils.typing import Array

__all__ = ["particle_gibbs", "sample_paths", "smooth"]


def _covariance(container: Any, suffix: str = "") -> Array:
    """The covariances of a Gaussian filter's states (or its predictions)."""
    if hasattr(container, f"cov{suffix}"):
        return getattr(container, f"cov{suffix}")
    if hasattr(container, f"std{suffix}"):
        std = getattr(container, f"std{suffix}")
        return std @ jnp.swapaxes(std, -1, -2)
    factor = getattr(container, f"cov_factor{suffix}")
    core = getattr(container, f"cov_core{suffix}")
    return factor @ core @ jnp.swapaxes(factor, -1, -2)


def _gaussian_backward_data(model, theta, initial, states, infos):
    """Filtered moments at every time, predictions, and the RTS gains."""
    states = prepend_to_trace(initial, states)
    means, covs = states.mean, _covariance(states)
    pred_covs = _covariance(infos, "_pred")

    def transition_matrix(mean, t_previous, t):
        if isinstance(model, LinearGaussianModel):
            return as_array(model.transition(theta, t_previous, t)[0])
        # A nonlinear model is linearised at the filtered mean, as the EKF does.
        return jax.jacfwd(lambda x: model.transition(theta, x, t_previous, t)[0])(mean)

    phis = jax.vmap(transition_matrix)(means[:-1], states.t[:-1], states.t[1:])
    gains = jax.vmap(lambda cov, phi, pred: jnp.linalg.solve(pred, phi @ cov).T)(
        covs[:-1], phis, pred_covs
    )
    return means, covs, infos.mean_pred, pred_covs, gains


def smooth(
    model: Any, theta: Any, initial: Any, states: Any, infos: Any
) -> tuple[Array, Array]:
    """Rauch--Tung--Striebel smoothed means and covariances of a Gaussian filter.

    Exact for a :class:`LinearGaussianModel` filtered by a Kalman filter; for
    a :class:`GaussianModel` it linearises the transition at the filtered
    means (the extended RTS smoother). Both outputs include the initial time,
    so they have ``len(states.t) + 1`` entries.
    """
    means, covs, pred_means, pred_covs, gains = _gaussian_backward_data(
        model, theta, initial, states, infos
    )

    def backward(carry: Any, data: Any) -> Any:
        next_mean, next_cov = carry
        mean, cov, pred_mean, pred_cov, gain = data
        mean = mean + gain @ (next_mean - pred_mean)
        cov = symmetrize_matrix(cov + gain @ (next_cov - pred_cov) @ gain.T)
        return (mean, cov), (mean, cov)

    _, (smoothed_means, smoothed_covs) = jax.lax.scan(
        backward,
        (means[-1], covs[-1]),
        (means[:-1], covs[:-1], pred_means, pred_covs, gains),
        reverse=True,
    )
    return (
        jnp.concatenate((smoothed_means, means[-1:])),
        jnp.concatenate((smoothed_covs, covs[-1:])),
    )


def _normal(rng, mean: Array, cov: Array, num_samples: int) -> Array:
    # Backward conditional covariance can be semidefinite. Clip roundoff at zero.
    values, vectors = jnp.linalg.eigh(symmetrize_matrix(cov))
    factor = vectors * jnp.sqrt(jnp.maximum(values, 0))
    return mean + jax.random.normal(rng, (num_samples, cov.shape[-1])) @ factor.T


def _sample_gaussian_paths(rng, model, theta, initial, states, infos, num_samples):
    # Backward conditionals keep the cross-time dependence of the joint law.
    means, covs, pred_means, pred_covs, gains = _gaussian_backward_data(
        model, theta, initial, states, infos
    )
    key, final_key = jax.random.split(rng)
    final = _normal(final_key, means[-1], covs[-1], num_samples)

    def backward(carry: Any, data: Any) -> Any:
        key, next_x = carry
        key, draw_key = jax.random.split(key)
        mean, cov, pred_mean, pred_cov, gain = data
        conditional_mean = mean + (next_x - pred_mean) @ gain.T
        conditional_cov = cov - gain @ pred_cov @ gain.T
        x = _normal(draw_key, conditional_mean, conditional_cov, num_samples)
        return (key, x), x

    _, paths = jax.lax.scan(
        backward,
        (key, final),
        (means[:-1], covs[:-1], pred_means, pred_covs, gains),
        reverse=True,
    )
    return jnp.concatenate((paths, final[None]))


def _genealogies(
    rng, particles: Array, log_weights: Array, ancestors: Array, num_samples: int
) -> Array:
    final = jax.random.categorical(rng, log_weights, shape=(num_samples,))

    def backward(indices: Array, data: Any) -> Any:
        population, parents = data
        indices = parents[indices]
        return indices, population[indices]

    _, paths = jax.lax.scan(backward, final, (particles[:-1], ancestors), reverse=True)
    return jnp.concatenate((paths, particles[-1, final][None]))


def _backward_simulation(rng, model, theta, states, num_samples):
    """Forward-filter backward-simulation (FFBSi), ``O(T N M)``.

    Each backward step reweights the filter particles at ``t`` by the
    transition density to the path's state at ``t + 1``.
    """
    particles, log_weights, times = states.particles, states.log_weights, states.t
    key, final_key = jax.random.split(rng)
    final = jax.random.categorical(final_key, log_weights[-1], shape=(num_samples,))

    def backward(carry: Any, data: Any) -> Any:
        key, next_x = carry
        population, weights, t_previous, t = data
        key, draw_key = jax.random.split(key)

        def ancestor(key, x_next):
            log_transition = jax.vmap(
                lambda x: model.transition_logdensity(theta, x_next, x, t_previous, t)
            )(population)
            backward_weights = normalize_log_weights(weights + log_transition)
            return jax.random.categorical(key, backward_weights)

        keys = jax.random.split(draw_key, num_samples)
        x = population[jax.vmap(ancestor)(keys, next_x)]
        return (key, x), x

    _, paths = jax.lax.scan(
        backward,
        (key, particles[-1, final]),
        (particles[:-1], log_weights[:-1], times[:-1], times[1:]),
        reverse=True,
    )
    return jnp.concatenate((paths, particles[-1, final][None]))


def sample_paths(
    rng,
    model: Any,
    theta: Any,
    initial: Any,
    states: Any,
    infos: Any,
    *,
    num_samples: int = 1,
    method: str | None = None,
) -> Array:
    """Draw joint latent paths from a filter's output at fixed ``theta``.

    Paths have shape ``(len(states.t) + 1, num_samples, D)``, including the
    initial state. A Gaussian filter's output is sampled backward from the
    RTS conditionals: exactly for a :class:`LinearGaussianModel`, linearised
    for a :class:`GaussianModel`. A particle filter's output uses
    ``method='backward'`` (FFBSi, cost ``O(T N M)``, needs
    ``model.transition_logdensity``) or ``'ancestry'`` (the stored parents,
    ``O(T M)``, prone to path collapse). By default backward sampling is used
    whenever the model has a transition density.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if not hasattr(states, "particles"):
        return _sample_gaussian_paths(
            rng, model, theta, initial, states, infos, num_samples
        )
    if method is None:
        method = "ancestry" if model.transition_logdensity is None else "backward"
    states = prepend_to_trace(initial, states)
    if method == "ancestry":
        return _genealogies(
            rng,
            states.particles,
            states.log_weights[-1],
            infos.ancestors,
            num_samples,
        )
    if method != "backward" or model.transition_logdensity is None:
        raise ValueError("Choose 'ancestry', or 'backward' with a transition density.")
    return _backward_simulation(rng, model, theta, states, num_samples)


def particle_gibbs(
    rng,
    reference_path: Array,
    model: Any,
    theta: Any,
    ts: Any,
    observations: Any,
    *,
    t0: Any = 0.0,
    num_particles: int = 32,
    observed: Any = None,
    ancestor_sampling: bool = True,
) -> Array:
    """One conditional-SMC update (PGAS by default), returning a new joint path.

    ``reference_path`` has shape ``(len(ts) + 1, D)``, including ``x(t0)``, and
    ``model`` is a :class:`StateSpaceModel` with array states. Ancestor
    sampling needs ``model.transition_logdensity``. Multinomial resampling at
    every step is intentional: pinning one index in systematic resampling is
    not a valid conditional resampler.

    The update preserves the smoothing law at fixed ``theta``. Repeated calls
    form an MCMC chain; one call is not an independent posterior draw.
    """
    ts, observations, observed = validate_observations(ts, observations, observed)
    if num_particles < 2:
        raise ValueError("Conditional SMC requires at least two particles.")
    if reference_path.ndim != 2 or reference_path.shape[0] != ts.shape[0] + 1:
        raise ValueError("reference_path must have shape (len(ts)+1, D).")
    if ancestor_sampling and model.transition_logdensity is None:
        raise ValueError("Ancestor sampling needs model.transition_logdensity.")
    key, initial_key = jax.random.split(rng)
    particles = jax.vmap(model.init, (0, None))(
        jax.random.split(initial_key, num_particles), theta
    )
    particles = particles.at[0].set(reference_path[0])
    initial_particles = particles
    weights = jnp.full((num_particles,), -jnp.log(num_particles))
    transition = jax.vmap(model.step, (0, None, 0, None, None))
    likelihood = jax.vmap(model.log_likelihood, (None, 0, None, None))

    def step(carry: Any, data: Any) -> Any:
        key, particles, weights, previous = carry
        key, resample_key, ancestor_key, predict_key = jax.random.split(key, 4)
        t, y, mask, reference = data
        parents = jax.random.categorical(resample_key, weights, shape=(num_particles,))
        if ancestor_sampling:
            ancestor_weights = weights + jax.vmap(
                lambda x: model.transition_logdensity(theta, reference, x, previous, t)
            )(particles)
            reference_parent = jax.random.categorical(ancestor_key, ancestor_weights)
        else:
            reference_parent = 0
        parents = parents.at[0].set(reference_parent)
        keys = jax.random.split(predict_key, num_particles)
        particles = transition(keys, theta, particles[parents], previous, t)
        particles = particles.at[0].set(reference)
        weights = jax.lax.cond(
            mask,
            lambda: likelihood(theta, particles, y, t),
            lambda: jnp.zeros_like(weights),
        )
        weights = normalize_log_weights(weights)
        return (key, particles, weights, t), (particles, parents)

    (key, _, weights, _), (populations, ancestors) = jax.lax.scan(
        step,
        (key, particles, weights, jnp.asarray(t0, dtype=ts.dtype)),
        (ts, observations, observed, reference_path[1:]),
    )
    populations = jnp.concatenate((initial_particles[None], populations))
    return _genealogies(key, populations, weights, ancestors, 1)[:, 0]
