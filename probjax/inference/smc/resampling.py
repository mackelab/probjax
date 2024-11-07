import jax
import jax.numpy as jnp
from blackjax.smc.resampling import multinomial, residual, systematic
from ott.geometry import pointcloud
from ott.initializers.linear.initializers import GaussianInitializer
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn


def resample_systematic(key, log_weights, particles):
    weights = jnp.exp(log_weights)
    idx = systematic(key, weights, log_weights.shape[0])
    new_log_weights = jnp.zeros_like(log_weights) - jnp.log(log_weights.shape[0])
    return particles[idx], new_log_weights, idx


def resample_multinomial(key, log_weights, particles):
    weights = jnp.exp(log_weights)
    idx = multinomial(key, weights, log_weights.shape[0])
    new_log_weights = jnp.zeros_like(log_weights) - jnp.log(log_weights.shape[0])
    return particles[idx], new_log_weights, idx


def resample_residual(key, log_weights, particles):
    weights = jnp.exp(log_weights)
    idx = residual(key, weights, log_weights.shape[0])
    new_log_weights = jnp.zeros_like(log_weights) - jnp.log(log_weights.shape[0])
    return particles[idx], new_log_weights, idx


def resample_ot(
    key,
    log_weights,
    particles,
    eps=0.5,
    tol=1e-2,
    min_iter=2,
    max_iter=2,
    inner_iterations=2,
    implicit_diff=None,
):
    N = log_weights.shape[0]
    weights = jnp.exp(log_weights)
    org_shape = particles.shape
    particles = particles.reshape(N, -1)

    geom = pointcloud.PointCloud(particles, particles, epsilon=eps)
    initializer = GaussianInitializer()
    weights = jnp.clip(weights, 1e-10, 1.0)
    ot_prob = linear_problem.LinearProblem(geom, weights, jnp.ones_like(weights) / N)
    solver = sinkhorn.Sinkhorn(
        threshold=tol,
        min_iterations=min_iter,
        max_iterations=max_iter,
        inner_iterations=max(inner_iterations, min_iter),
        initializer=initializer,
        implicit_diff=implicit_diff,
    )

    ot = solver(ot_prob, rng=key)
    T_matrix = N * ot.matrix.T
    idx = T_matrix.argmax(axis=1)
    new_particles = T_matrix @ particles
    new_log_weights = jnp.zeros_like(log_weights) - jnp.log(log_weights.shape[0])
    return new_particles.reshape(org_shape), new_log_weights, idx


def resample_smoothed(
    key, log_weights, particles, resample_fn=resample_systematic, alpha=0.8
):
    # Also return weights
    weights_adapted = (
        alpha * jnp.exp(log_weights)
        + (1 - alpha) * jnp.ones_like(log_weights) / log_weights.shape[0]
    )
    weights_adapted = weights_adapted / jnp.sum(weights_adapted)
    log_weights_adapted = jnp.log(weights_adapted)
    new_particles, new_log_weights, idx = resample_fn(
        key, log_weights_adapted, particles
    )
    new_log_weights = log_weights[idx] - log_weights_adapted[idx]
    new_log_weights = new_log_weights - jax.scipy.special.logsumexp(new_log_weights)
    return new_particles, new_log_weights, idx
