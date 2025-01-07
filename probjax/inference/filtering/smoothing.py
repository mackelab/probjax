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


def smooth(
    ts: Array, mus: Array, covs: Array, mus_: Array, covs_: Array, smooth: Callable
) -> Tuple[Array, Array]:
    """Smooths the state given a Kalman filter output.

    Args:
        ts (Array): Time grid
        mus (Array): Means
        covs (Array): Covs
        mus_ (Array): Predicted means
        covs_ (Array): Predicted covs
        smooth (Callable): Smoothing function

    Returns:
        Tuple[Array, Array]: _description_
    """

    idx_last = jnp.where((mus != mus_).all(-1))[-1][-1]
    mus_needed_ = jnp.flip(mus_[1 : idx_last + 1])
    covs_needed_ = jnp.flip(covs_[1 : idx_last + 1])
    mus_needed = jnp.flip(mus[:idx_last])
    covs_needed = jnp.flip(covs[:idx_last])
    ts_needed = jnp.flip(ts[:idx_last])

    def scan_fun(carry, data):
        (mu0_s, cov0_s, t1) = carry
        t0, mu0, cov0, mu0_, cov0_ = data
        mu1, cov1 = smooth(t0, t1, mu0_s, cov0_s, mu0, cov0, mu0_, cov0_)
        return (mu1, cov1, t0), (mu1, cov1)

    init_carry = (mus[idx_last], covs[idx_last], ts[idx_last])
    _, (mus_s, covs_s) = lax.scan(
        scan_fun,
        init_carry,
        (ts_needed, mus_needed, covs_needed, mus_needed_, covs_needed_),
    )

    mus = jnp.concatenate([mus_s[::-1], mus[idx_last:]])
    covs = jnp.concatenate([covs_s[::-1], covs[idx_last:]])

    return mus, covs
