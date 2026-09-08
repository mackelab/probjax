from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array


def rauch_tung_stribel_smoother(
    transition_matrix_fn: Callable,
    t0: float,
    t1: float,
    mu0_s: Array,
    cov0_s: Array,
    mu0: Array,
    cov0: Array,
    mu0_: Array,
    cov0_: Array,
) -> Tuple[Array, Array]:
    """Discrete time Rauch-Tung-Striebel smoothing.

    Args:
        transition_matrix_fn (Callable): Transition matrix function
        t0 (float): Time of start
        t1 (float): Time of end
        mu0_s (Array): Smoothed mean at t0
        cov0_s (Array): Smoothed covariance at t0
        mu0 (Array): Unsmoothed mean at t0
        cov0 (Array): Unsmoothed covariance at t0
        mu0_ (Array): Prediction mean at t0
        cov0_ (Array): Prediction covariance at t0

    Returns:
        Tuple[Array, Array]: Updated mean and covariance.
    """
    Phi = transition_matrix_fn(t0, t1)
    G = jnp.dot(cov0, jnp.linalg.solve(cov0_, Phi).T)
    mu1 = mu0 + jnp.dot(G, mu0_s - jnp.dot(Phi, mu0_))
    cov1 = cov0 + jnp.dot(G, jnp.dot(cov0_s - cov0_, G.T))
    return mu1, cov1


def _ffbsi_backward_step(
    key: Array,
    smoothed_indices: Array,
    filter_particles_t: Array,
    filter_log_weights_t: Array,
    filter_particles_tp1: Array,
    transition_logdensity_fn: Callable,
    t: float,
    tp1: float,
) -> Array:
    """Single backward step of the FFBSi algorithm.

    For each smoothed particle at t+1, compute backward smoothing weights
    over all filter particles at t, then sample an ancestor.

    Args:
        key: Random key.
        smoothed_indices: Indices into filter_particles_tp1 for the current
            smoothed particles at t+1.
        filter_particles_t: Filter particles at time t, shape (N, D).
        filter_log_weights_t: Filter log-weights at time t, shape (N,).
        filter_particles_tp1: Filter particles at time t+1, shape (N, D).
        transition_logdensity_fn: Log-density of transition p(x_{t+1} | x_t, t, t+1).
        t: Time at step t.
        tp1: Time at step t+1.

    Returns:
        New smoothed indices at time t, shape (N,).
    """
    N = smoothed_indices.shape[0]
    # The smoothed particles at t+1
    x_tp1 = filter_particles_tp1[smoothed_indices]

    # For each smoothed particle x_{t+1}^j, compute weights over all
    # filter particles x_t^i: w_i|j = w_t^i * p(x_{t+1}^j | x_t^i)
    def compute_ancestor_for_one(key_j, x_tp1_j):
        # log p(x_{t+1}^j | x_t^i) for all i
        log_transition = jax.vmap(
            lambda x_t_i: transition_logdensity_fn(x_tp1_j, x_t_i, t, tp1)
        )(filter_particles_t)
        # Backward smoothing weights (unnormalized): log w_t^i + log p(x_{t+1}^j | x_t^i)
        log_backward_weights = filter_log_weights_t + log_transition
        # Normalize
        log_backward_weights = log_backward_weights - jax.scipy.special.logsumexp(
            log_backward_weights
        )
        # Sample ancestor index
        ancestor = jax.random.categorical(key_j, log_backward_weights)
        return ancestor

    keys = jax.random.split(key, N)
    new_indices = jax.vmap(compute_ancestor_for_one)(keys, x_tp1)
    return new_indices


def particle_smoother(
    key: Array,
    ts: Array,
    filter_particles: Array,
    filter_log_weights: Array,
    transition_logdensity_fn: Callable,
    ancestors: Optional[Array] = None,
    *,
    num_samples: Optional[int] = None,
) -> Tuple[Array, Array]:
    """Forward Filter-Backward Simulator (FFBSi) particle smoother.

    Takes the output of a forward particle filter pass and runs a backward
    simulation pass to approximate the smoothing distribution p(x_{0:T} | y_{1:T}).

    The algorithm is O(T * N^2) where T is the number of time steps and N is
    the number of particles, due to the pairwise transition density evaluation
    at each backward step.

    Args:
        key (Array): Random key.
        ts (Array): Time grid, shape (T,).
        filter_particles (Array): Particles from the filter, shape (T, N, D).
        filter_log_weights (Array): Log-weights from the filter, shape (T, N).
        transition_logdensity_fn (Callable): Log-density of the transition model.
            Signature: (x_tp1, x_t, t, tp1) -> scalar log-density.
        ancestors (Optional[Array]): Ancestor indices from the filter,
            shape (T, N). Not used in FFBSi but accepted for API compatibility.
        num_samples: Number M of joint trajectories to sample. Defaults to N.

    Returns:
        Tuple[Array, Array]:
            - smoothed_particles: shape (T, M, D)
            - smoothed_log_weights: shape (T, M), uniform weights (1/M)
    """
    T, N, D = filter_particles.shape
    num_samples = N if num_samples is None else num_samples
    if T < 1 or num_samples < 1:
        raise ValueError(
            "A smoother requires a nonempty trace and positive num_samples."
        )

    # Initialize: at the last time step, smoothed = filter
    final_log_weights = filter_log_weights[-1]
    # Sample initial smoothed indices from the final filter distribution
    key, subkey = jax.random.split(key)
    smoothed_indices_T = jax.random.categorical(
        subkey, final_log_weights, shape=(num_samples,)
    )

    # Backward scan from T-1 down to 0
    def backward_step(carry, data):
        key, smoothed_indices_tp1 = carry
        filter_particles_t, filter_log_weights_t, filter_particles_tp1, t, tp1 = data
        key, subkey = jax.random.split(key)

        new_indices = _ffbsi_backward_step(
            subkey,
            smoothed_indices_tp1,
            filter_particles_t,
            filter_log_weights_t,
            filter_particles_tp1,
            transition_logdensity_fn,
            t,
            tp1,
        )
        return (key, new_indices), new_indices

    # Prepare backward scan data: from t=T-2 down to t=0
    # At each step we need: filter_particles[t], filter_log_weights[t],
    #   filter_particles[t+1], ts[t], ts[t+1]
    scan_data = (
        jnp.flip(filter_particles[:-1], axis=0),  # filter particles at t
        jnp.flip(filter_log_weights[:-1], axis=0),  # filter log-weights at t
        jnp.flip(filter_particles[1:], axis=0),  # filter particles at t+1
        jnp.flip(ts[:-1]),  # t
        jnp.flip(ts[1:]),  # t+1
    )

    init_carry = (key, smoothed_indices_T)
    _, all_smoothed_indices = jax.lax.scan(backward_step, init_carry, scan_data)

    # all_smoothed_indices is shape (T-1, N) in reversed time order
    # Flip back to forward time order and append the final indices
    all_smoothed_indices = jnp.flip(all_smoothed_indices, axis=0)
    all_smoothed_indices = jnp.concatenate(
        [all_smoothed_indices, smoothed_indices_T[None, :]], axis=0
    )

    # Gather smoothed particles
    smoothed_particles = jax.vmap(lambda particles, idx: particles[idx])(
        filter_particles, all_smoothed_indices
    )

    # Smoothed weights are uniform
    smoothed_log_weights = jnp.full((T, num_samples), fill_value=-jnp.log(num_samples))

    return smoothed_particles, smoothed_log_weights


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
    _, (mus_s, covs_s) = jax.lax.scan(
        scan_fun,
        init_carry,
        (ts_needed, mus_needed, covs_needed, mus_needed_, covs_needed_),
    )

    mus = jnp.concatenate([mus_s[::-1], mus[idx_last:]])
    covs = jnp.concatenate([covs_s[::-1], covs[idx_last:]])

    return mus, covs
