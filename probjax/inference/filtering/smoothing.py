from typing import Callable, Float, Tuple

import jax.numpy as jnp
from jax import Array


def rauch_tung_stribel_smoother(
    transition_matrix_fn: Callable,
    t0: Float,
    t1: Float,
    mu0_s: Array,
    cov0_s: Array,
    mu0: Array,
    cov0: Array,
    mu0_: Array,
    cov0_: Array,
) -> Tuple[Array, Array]:
    """Discrete time Rauch-Tung-Striebel smoothing.

    Args:
        t0 (Float): Time of start
        t1 (Float): Time of end
        mu0_s (Array): Smoothed mean at t0
        cov0_s (Array): Smoothed covariance at t0
        mu0 (Array): Unsmoothed mean at t0
        cov0 (Array): Unsommthed covariance at t0
        mu0_ (Array): Prediction mean at t0
        cov0_ (Array): Prediction covariance at t0
        drift_matrix (Callable): Drift matrix

    Returns:
        Tuple[Array, Array]: Updated mean and covariance.
    """
    Phi = transition_matrix_fn(t0, t1)
    G = jnp.dot(cov0, jnp.linalg.solve(cov0_, Phi).T)
    mu1 = mu0 + jnp.dot(G, mu0_s - jnp.dot(Phi, mu0_))
    cov1 = cov0 + jnp.dot(G, jnp.dot(cov0_s - cov0_, G.T))
    return mu1, cov1


def particle_smoother(
    transition_fn: Callable,
    observation_fn: Callable,
    resample_fn: Callable,
    num_particles: int,
    t0: Float,
    t1: Float,
    particles: Array,
    weights: Array,
    rng_key: Array,
    *args,
    **kwargs,
) -> Tuple[Array, Array]:
    pass
