"""Filters on the kernel contract, checked against a linear-Gaussian model.

Every Gaussian filter is exact on a linear-Gaussian model, so its likelihood,
filtered moments and smoothed moments must match the Kalman filter's; the
particle filter's likelihood must be unbiased.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import multivariate_normal

from probjax.inference import (
    GaussianModel,
    LinearGaussianModel,
    StateSpaceModel,
    extended_kalman_filter,
    kalman_filter,
    particle_filter,
    rank_reduced_kalman_filter,
    run,
    sample_paths,
    smooth,
    square_root_kalman_filter,
    unscented_kalman_filter,
)
from probjax.inference.filtering import (
    julier_uhlmann_sigma_points,
    merwe_sigma_point,
    spherical_simplex_sigma_points,
)

A = jnp.array([[0.9, 0.1], [0.0, 0.8]])
C = jnp.array([[1.0, 0.5]])
R = jnp.array([[0.3]])
TS = jnp.arange(1.0, 9.0)
OBSERVED = jnp.array([True, True, False, True, True, True, False, True])
KEY = jax.random.key(0)


def process_noise(theta):
    return jnp.diag(jnp.exp(theta))


THETA = jnp.log(jnp.array([0.1, 0.2]))
LINEAR = LinearGaussianModel(
    lambda theta: (jnp.zeros(2), jnp.eye(2)),
    lambda theta, t0, t1: (A, process_noise(theta)),
    lambda theta, t: (C, R),
)
NONLINEAR = GaussianModel(
    LINEAR.initial,
    lambda theta, x, t0, t1: (A @ x, process_noise(theta)),
    lambda theta, x, t: (C @ x, R),
)


def particle_model(theta=None):
    def init(key, theta):
        return jax.random.normal(key, (2,))

    def step(key, theta, x, t0, t1):
        noise = jnp.sqrt(jnp.exp(theta)) * jax.random.normal(key, (2,))
        return A @ x + noise

    def log_likelihood(theta, x, y, t):
        return multivariate_normal.logpdf(y, C @ x, R)

    def transition_logdensity(theta, new, old, t0, t1):
        return multivariate_normal.logpdf(new, A @ old, process_noise(theta))

    return StateSpaceModel(init, step, log_likelihood, transition_logdensity)


def _data():
    ys = jax.random.normal(jax.random.key(1), (8, 1))
    return ys.at[2].set(jnp.nan)  # an unobserved step may hold anything


YS = _data()


def exact_log_likelihood(theta):
    """The joint Gaussian law of the observed ys, built directly."""
    Q = process_noise(theta)
    covs, cov = [], jnp.eye(2)
    for _ in TS:
        cov = A @ cov @ A.T + Q
        covs.append(cov)
    # Cov(x_i, x_j) = A^(j-i) Cov(x_i) for j >= i.
    n = len(TS)
    joint = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cross = np.linalg.matrix_power(np.asarray(A), j - i) @ np.asarray(covs[i])
            joint[i, j] = joint[j, i] = (C @ cross @ C.T)[0, 0]
    idx = np.flatnonzero(np.asarray(OBSERVED))
    obs_cov = joint[np.ix_(idx, idx)] + 0.3 * np.eye(len(idx))
    return multivariate_normal.logpdf(
        jnp.asarray(YS)[idx, 0], jnp.zeros(len(idx)), jnp.asarray(obs_cov)
    )


def filter_run(kernel, theta=THETA, key=KEY, collect=("state", "info")):
    init_key, run_key = jax.random.split(key)
    state = kernel.init(init_key, theta)
    result = run(
        run_key, kernel, state, theta, len(TS), args=(TS, YS, OBSERVED), collect=collect
    )
    return state, result


def _covariance(state):
    if hasattr(state, "cov"):
        return state.cov
    if hasattr(state, "std"):
        return state.std @ state.std.T
    return state.cov_factor @ state.cov_core @ state.cov_factor.T


GAUSSIAN_FILTERS = {
    "kalman": kalman_filter(LINEAR),
    "extended": extended_kalman_filter(NONLINEAR),
    "extended-matrix-free": extended_kalman_filter(NONLINEAR, matrix_free=True),
    "unscented-merwe": unscented_kalman_filter(NONLINEAR),
    "unscented-julier": unscented_kalman_filter(
        NONLINEAR, sigma_points=julier_uhlmann_sigma_points
    ),
    "unscented-simplex": unscented_kalman_filter(
        NONLINEAR, sigma_points=spherical_simplex_sigma_points
    ),
    "square-root": square_root_kalman_filter(LINEAR),
    "rank-reduced-full": rank_reduced_kalman_filter(LINEAR, 2),
}


@pytest.mark.parametrize("name", GAUSSIAN_FILTERS)
def test_gaussian_filters_are_exact_on_a_linear_model(name):
    _, reference = filter_run(GAUSSIAN_FILTERS["kalman"])
    _, result = filter_run(GAUSSIAN_FILTERS[name])
    np.testing.assert_allclose(
        result.state.log_likelihood, exact_log_likelihood(THETA), rtol=1e-4
    )
    np.testing.assert_allclose(result.state.mean, reference.state.mean, atol=1e-4)
    np.testing.assert_allclose(
        _covariance(result.state), reference.state.cov, atol=1e-4
    )
    # Unobserved steps only predict.
    increments = result.trace["info"].log_likelihood_increment
    assert jnp.all(increments[~OBSERVED] == 0.0)


def test_likelihood_gradient_matches_finite_differences():
    kernel = kalman_filter(LINEAR)

    def log_likelihood(theta):
        return filter_run(kernel, theta, collect=())[1].state.log_likelihood

    gradient = jax.grad(log_likelihood)(THETA)
    step = 1e-3
    for i in range(2):
        shift = jnp.zeros(2).at[i].set(step)
        numeric = (
            exact_log_likelihood(THETA + shift) - exact_log_likelihood(THETA - shift)
        ) / (2 * step)
        np.testing.assert_allclose(gradient[i], numeric, rtol=2e-2)


def test_nan_in_unobserved_steps_keeps_vmapped_gradients_finite():
    kernel = particle_filter(particle_model(), 64, unbiased_gradients=True)

    def log_likelihood(theta, key):
        return filter_run(kernel, theta, key, collect=())[1].state.log_likelihood

    keys = jax.random.split(KEY, 3)
    gradients = jax.vmap(jax.grad(log_likelihood), (None, 0))(THETA, keys)
    assert jnp.all(jnp.isfinite(gradients))


@pytest.mark.parametrize("sigma_points", [merwe_sigma_point])
def test_unscented_filter_handles_a_nonlinear_observation(sigma_points):
    model = NONLINEAR._replace(observation=lambda theta, x, t: (jnp.tanh(C @ x), R))
    _, result = filter_run(unscented_kalman_filter(model, sigma_points=sigma_points))
    assert jnp.isfinite(result.state.log_likelihood)
    assert jnp.all(jnp.linalg.eigvalsh(result.state.cov) > 0)


def test_rank_reduced_low_rank_noise_matches_dense():
    Uq, Sq = jnp.array([[1.0], [0.3]]), jnp.array([[0.04]])
    dense = LINEAR._replace(transition=lambda theta, t0, t1: (A, Uq @ Sq @ Uq.T))
    factored = LINEAR._replace(transition=lambda theta, t0, t1: (A, (Uq, Sq)))
    _, a = filter_run(rank_reduced_kalman_filter(dense, 2))
    _, b = filter_run(rank_reduced_kalman_filter(factored, 2))
    np.testing.assert_allclose(b.state.mean, a.state.mean, atol=1e-6)
    np.testing.assert_allclose(_covariance(b.state), _covariance(a.state), atol=1e-6)


def test_rank_reduced_energy_threshold_drops_directions():
    model = LinearGaussianModel(
        lambda theta: (jnp.zeros(3), jnp.eye(3)),
        lambda theta, t0, t1: (
            jnp.diag(jnp.array([0.98, 0.92, 0.85])),
            jnp.diag(jnp.array([0.08, 0.01, 1e-4])),
        ),
        lambda theta, t: (jnp.array([[1.0, 0.2, 0.0]]), jnp.array([[0.1]])),
    )
    kernel = rank_reduced_kalman_filter(model, 3, energy_threshold=0.9)
    state = kernel.init(KEY, None)
    result = run(KEY, kernel, state, None, 4, args=(TS[:4], jnp.full((4, 1), 0.05)))
    effective_rank = int(jnp.sum(jnp.diag(result.state.cov_core) > 1e-6))
    assert 1 <= effective_rank < 3


def _pf_log_likelihoods(kernel, num_runs=200):
    return jax.vmap(
        lambda key: filter_run(kernel, THETA, key, collect=())[1].state.log_likelihood
    )(jax.random.split(KEY, num_runs))


def test_particle_filter_likelihood_is_unbiased():
    estimates = _pf_log_likelihoods(particle_filter(particle_model(), 256))
    mean_likelihood = jax.scipy.special.logsumexp(estimates) - jnp.log(len(estimates))
    np.testing.assert_allclose(mean_likelihood, exact_log_likelihood(THETA), atol=0.05)


def test_guided_proposal_is_unbiased_and_less_noisy():
    model = particle_model()

    def locally_optimal(key, theta, x, y, t0, t1):
        # p(x' | x, y) for the linear-Gaussian model: the exact guided proposal.
        Q = process_noise(theta)
        gain = Q @ C.T @ jnp.linalg.inv(C @ Q @ C.T + R)
        mean = A @ x + gain @ (y - C @ A @ x)
        cov = Q - gain @ C @ Q
        draw = jax.random.multivariate_normal(key, mean, cov)
        return draw, multivariate_normal.logpdf(draw, mean, cov)

    bootstrap = _pf_log_likelihoods(particle_filter(model, 64))
    guided = _pf_log_likelihoods(particle_filter(model, 64, proposal=locally_optimal))
    exact = exact_log_likelihood(THETA)
    mean_guided = jax.scipy.special.logsumexp(guided) - jnp.log(len(guided))
    np.testing.assert_allclose(mean_guided, exact, atol=0.05)
    assert guided.std() < 0.85 * bootstrap.std()


def test_smoothing_is_the_same_for_every_exact_filter():
    initial, reference = filter_run(GAUSSIAN_FILTERS["kalman"])
    means, covs = smooth(
        LINEAR, THETA, initial, reference.trace["state"], reference.trace["info"]
    )
    assert means.shape == (len(TS) + 1, 2)
    for name, model in [
        ("extended", NONLINEAR),
        ("square-root", LINEAR),
        ("rank-reduced-full", LINEAR),
    ]:
        initial, result = filter_run(GAUSSIAN_FILTERS[name])
        other_means, other_covs = smooth(
            model, THETA, initial, result.trace["state"], result.trace["info"]
        )
        np.testing.assert_allclose(other_means, means, atol=1e-4)
        np.testing.assert_allclose(other_covs, covs, atol=1e-4)


def test_particle_paths_match_the_smoothed_means():
    initial, reference = filter_run(GAUSSIAN_FILTERS["kalman"])
    means, _ = smooth(
        LINEAR, THETA, initial, reference.trace["state"], reference.trace["info"]
    )
    model = particle_model()
    initial, result = filter_run(particle_filter(model, 512))
    paths = sample_paths(
        KEY,
        model,
        THETA,
        initial,
        result.trace["state"],
        result.trace["info"],
        num_samples=1000,
    )
    assert paths.shape == (len(TS) + 1, 1000, 2)
    np.testing.assert_allclose(paths.mean(1), means, atol=0.15)
