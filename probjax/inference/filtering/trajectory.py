"""Joint trajectory sampling and particle Gibbs with ancestor sampling."""

import jax
import jax.numpy as jnp

from probjax.inference.filtering.smoothing import particle_smoother
from probjax.inference.filtering.temporal import _observations


def _prepend(initial, history):
    return jax.tree.map(lambda x, xs: jnp.concatenate((x[None], xs)), initial, history)


def _genealogies(key, particles, log_weights, ancestors, num_samples):
    final = jax.random.categorical(key, log_weights, shape=(num_samples,))

    def backward(indices, data):
        population, parents = data
        indices = parents[indices]
        return indices, population[indices]

    _, paths = jax.lax.scan(backward, final, (particles[:-1], ancestors), reverse=True)
    return jnp.concatenate((paths, particles[-1, final][None]))


def sample_particle_paths(
    key, trace, *, num_samples=1, method="backward", transition_logdensity_fn=None
):
    """Return joint paths (time INCLUDING initial state, samples, dimension).

    'ancestry' traces stored discrete parents in O(T*M); 'backward' uses FFBSi
    in O(T*N*M), requiring log p(x_next|x_previous, t_previous, t_next).
    Traces must come from discrete resampling, as in particle_backend.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    states = _prepend(trace.initial_state, trace.states)
    if method == "ancestry":
        return _genealogies(
            key,
            states.particles,
            states.log_weights[-1],
            trace.infos.ancestors,
            num_samples,
        )
    if method != "backward" or transition_logdensity_fn is None:
        raise ValueError("Choose 'ancestry', or 'backward' with a transition density.")
    times = jnp.concatenate((jnp.asarray(trace.initial_state.t)[None], trace.ts))
    paths, _ = particle_smoother(
        key,
        times,
        states.particles,
        states.log_weights,
        transition_logdensity_fn,
        num_samples=num_samples,
    )
    return paths


def _gaussian_backward_data(trace, transition_fn):
    states = _prepend(trace.initial_state, trace.states)
    times = jnp.concatenate((jnp.asarray(trace.initial_state.t)[None], trace.ts))
    phis = jax.vmap(transition_fn)(times[:-1], times[1:])
    gains = jax.vmap(lambda p, a, pred: jnp.linalg.solve(pred, a @ p).T)(
        states.cov[:-1], phis, trace.infos.cov_pred
    )
    return states, gains


def smooth_gaussian_path(trace, transition_fn):
    """RTS means/covariances including the initial state; transition_fn(t0,t1)->Phi."""
    states, gains = _gaussian_backward_data(trace, transition_fn)

    def backward(carry, data):
        next_mean, next_cov = carry
        mean, cov, pred_mean, pred_cov, gain = data
        mean = mean + gain @ (next_mean - pred_mean)
        cov = cov + gain @ (next_cov - pred_cov) @ gain.T
        cov = (cov + cov.T) / 2
        return (mean, cov), (mean, cov)

    _, (means, covs) = jax.lax.scan(
        backward,
        (states.mean[-1], states.cov[-1]),
        (
            states.mean[:-1],
            states.cov[:-1],
            trace.infos.mean_pred,
            trace.infos.cov_pred,
            gains,
        ),
        reverse=True,
    )
    return (
        jnp.concatenate((means, states.mean[-1:])),
        jnp.concatenate((covs, states.cov[-1:])),
    )


def _normal(key, mean, cov, num_samples):
    # Backward conditional covariance can be semidefinite. Clip roundoff at zero.
    values, vectors = jnp.linalg.eigh((cov + cov.T) / 2)
    factor = vectors * jnp.sqrt(jnp.maximum(values, 0))
    return mean + jax.random.normal(key, (num_samples, cov.shape[-1])) @ factor.T


def sample_gaussian_paths(key, trace, transition_fn, *, num_samples=1):
    """Sample joint Gaussian trajectories, preserving cross-time dependence."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    states, gains = _gaussian_backward_data(trace, transition_fn)
    key, final_key = jax.random.split(key)
    final = _normal(final_key, states.mean[-1], states.cov[-1], num_samples)

    def backward(carry, data):
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
        (
            states.mean[:-1],
            states.cov[:-1],
            trace.infos.mean_pred,
            trace.infos.cov_pred,
            gains,
        ),
        reverse=True,
    )
    return jnp.concatenate((paths, final[None]))


def particle_gibbs(
    key,
    reference_path,
    theta,
    t0,
    ts,
    observations,
    *,
    initial_fn,
    transition_fn,
    transition_logdensity_fn,
    log_likelihood_fn,
    num_particles=32,
    observed=None,
    ancestor_sampling=True,
):
    """One bootstrap conditional-SMC/PGAS update, returning a new joint path.

    reference_path has shape (len(ts)+1, D), including x(t0). Callback signatures
    match particle_backend; the transition density takes scalar states and returns
    a scalar: (theta, x_next, x_previous, t_previous, t_next). initial_fn samples
    exactly num_particles independent draws from the initial law. Observation
    likelihoods are batched. Multinomial resampling at every step is intentional:
    conditioning ordinary systematic resampling by pinning one index is invalid.

    The kernel preserves the full smoothing target for a fixed theta. Repeated
    calls form an MCMC chain; one call is not an independent posterior draw.
    """
    ts, observations, observed = _observations(ts, observations, observed)
    if num_particles < 2:
        raise ValueError("Conditional SMC requires at least two particles.")
    if reference_path.ndim != 2 or reference_path.shape[0] != ts.shape[0] + 1:
        raise ValueError("reference_path must have shape (len(ts)+1, D).")
    key, initial_key = jax.random.split(key)
    particles = initial_fn(initial_key, theta, t0)
    if particles.shape != (num_particles, reference_path.shape[1]):
        raise ValueError("initial_fn must return (num_particles, path_dimension).")
    particles = particles.at[0].set(reference_path[0])
    initial_particles = particles
    weights = jnp.full((num_particles,), -jnp.log(num_particles))

    def step(carry, data):
        key, particles, weights, previous = carry
        key, resample_key, ancestor_key, predict_key = jax.random.split(key, 4)
        t, y, mask, reference = data
        parents = jax.random.categorical(resample_key, weights, shape=(num_particles,))
        if ancestor_sampling:
            ancestor_weights = weights + jax.vmap(
                lambda x: transition_logdensity_fn(theta, reference, x, previous, t)
            )(particles)
            reference_parent = jax.random.categorical(ancestor_key, ancestor_weights)
        else:
            reference_parent = 0
        parents = parents.at[0].set(reference_parent)
        particles = transition_fn(predict_key, theta, particles[parents], previous, t)
        particles = particles.at[0].set(reference)
        weights = jax.lax.cond(
            mask,
            lambda: log_likelihood_fn(theta, particles, y, t),
            lambda: jnp.zeros_like(weights),
        )
        weights = weights - jax.scipy.special.logsumexp(weights)
        return (key, particles, weights, t), (particles, parents)

    (key, _, weights, _), (populations, ancestors) = jax.lax.scan(
        step,
        (key, particles, weights, jnp.asarray(t0, dtype=ts.dtype)),
        (ts, observations, observed, reference_path[1:]),
    )
    populations = jnp.concatenate((initial_particles[None], populations))
    return _genealogies(key, populations, weights, ancestors, 1)[:, 0]
