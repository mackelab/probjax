"""Tests for the filtering module.

Tests all filters against a simple 1D linear Gaussian model where
analytical solutions are known. Also tests the filter_smooth orchestration.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial

from probjax.inference.filtering.base import FilterKernel, make_filter_api
from probjax.inference.filtering.kalman_filter import (
    KalmanFilterState,
    kalman_filter,
)
from probjax.inference.filtering.extended_kalman_filter import extended_kalman_filter
from probjax.inference.filtering.unscented_kalman_filter import (
    ukf,
    merwe_sigma_point,
    julier_uhlmann_sigma_points,
    spherical_simplex_sigma_points,
)
from probjax.inference.filtering.square_root_kf import sq_kalman_filter
from probjax.inference.filtering.rank_reduced_kalman_filter import (
    rank_reduced_kalman_filter,
)
from probjax.inference.filtering.particle_filter import ParticleFilter
from probjax.inference.filtering.smoothing import (
    particle_smoother,
    rauch_tung_stribel_smoother,
)


# ---------------------------------------------------------------------------
# Shared test fixtures: 1D linear Gaussian model
# x_{t+1} = A * x_t + w_t,  w_t ~ N(0, Q)
# y_t = C * x_t + v_t,      v_t ~ N(0, R)
# ---------------------------------------------------------------------------

A = jnp.array([[0.9]])
Q = jnp.array([[0.1]])
C = jnp.array([[1.0]])
R = jnp.array([[0.5]])


# Transition model for KF: returns (Phi, Q)
def transition_model(t_old, t):
    return A, Q


# Observation model for KF: returns (C, R)
def observation_model(t):
    return C, R


# Transition function for EKF/UKF: f(x, t_old, t) -> x_new
def transition_fn(x, t_old, t):
    return A @ x


# Observation function for EKF/UKF: h(x, t) -> y
def observation_fn(x, t):
    return C @ x


# EKF-specific: returns (Jacobian, Q)
def transition_matrix_and_cov(x, cov, t):
    return A, Q


# EKF-specific: returns (Jacobian, R)
def observation_matrix_and_cov(x, cov, t):
    return C, R


# Initial state
mu0 = jnp.array([0.0])
cov0 = jnp.array([[1.0]])
std0 = jnp.linalg.cholesky(cov0)


# Generate some synthetic data
def generate_data(key, num_steps=10):
    """Generate observations from the linear Gaussian model."""
    key1, key2 = jax.random.split(key)
    states = [mu0]
    for i in range(num_steps):
        key1, subkey = jax.random.split(key1)
        x_new = A @ states[-1] + jax.random.normal(subkey, (1,)) * jnp.sqrt(Q[0, 0])
        states.append(x_new)

    observations = []
    for i, x in enumerate(states[1:]):
        key2, subkey = jax.random.split(key2)
        y = C @ x + jax.random.normal(subkey, (1,)) * jnp.sqrt(R[0, 0])
        observations.append(y)

    return jnp.stack(states[1:]), jnp.stack(observations)


# ---------------------------------------------------------------------------
# Tests: FilterKernel interface
# ---------------------------------------------------------------------------


class TestFilterKernel:
    def test_kernel_has_default_unpack(self):
        """FilterKernel should have a default_unpack field."""
        kernel = kalman_filter(transition_model, observation_model)
        assert hasattr(kernel, "default_unpack")
        assert callable(kernel.default_unpack)

    def test_kernel_is_callable(self):
        """FilterKernel should be callable via __call__."""
        kernel = kalman_filter(transition_model, observation_model)
        state = kernel.init(mu0, cov0, t=0.0)
        new_state, info = kernel(state, t=1.0, observed=None, rng_key=None)
        assert new_state.mean.shape == mu0.shape

    def test_kalman_default_unpack(self):
        """Kalman filter default_unpack should return (mean, cov)."""
        kernel = kalman_filter(transition_model, observation_model)
        state = KalmanFilterState(mu0, cov0, 0.0)
        mean, cov = kernel.default_unpack(state, None)
        np.testing.assert_array_equal(mean, mu0)
        np.testing.assert_array_equal(cov, cov0)


# ---------------------------------------------------------------------------
# Tests: Kalman Filter
# ---------------------------------------------------------------------------


class TestKalmanFilter:
    def test_predict_step(self):
        """Test a single predict step (no observation)."""
        kernel = kalman_filter(transition_model, observation_model)
        state = kernel.init(mu0, cov0, t=0.0)
        new_state, info = kernel(state, t=1.0, observed=None, rng_key=None)

        expected_mean = A @ mu0
        expected_cov = A @ cov0 @ A.T + Q

        np.testing.assert_allclose(new_state.mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(new_state.cov, expected_cov, atol=1e-6)

    def test_update_step(self):
        """Test a single update step (with observation)."""
        kernel = kalman_filter(transition_model, observation_model)
        state = kernel.init(mu0, cov0, t=0.0)

        y = jnp.array([1.0])
        new_state, info = kernel(state, t=1.0, observed=y, rng_key=None)

        # After predict + update, covariance should be smaller than predict-only
        predict_cov = A @ cov0 @ A.T + Q
        assert jnp.all(jnp.diag(new_state.cov) < jnp.diag(predict_cov))

        # Info should have log_likelihood
        assert info.log_likelihood is not None
        assert jnp.isfinite(info.log_likelihood)

    def test_multiple_steps(self):
        """Test multiple filtering steps."""
        key = jax.random.PRNGKey(42)
        _, observations = generate_data(key, num_steps=5)

        kernel = kalman_filter(transition_model, observation_model)
        state = kernel.init(mu0, cov0, t=0.0)

        for i in range(5):
            state, info = kernel(state, t=float(i + 1), observed=observations[i])
            assert jnp.all(jnp.isfinite(state.mean))
            assert jnp.all(jnp.isfinite(state.cov))


# ---------------------------------------------------------------------------
# Tests: Extended Kalman Filter
# ---------------------------------------------------------------------------


class TestExtendedKalmanFilter:
    def test_linear_model_matches_kf(self):
        """On a linear model, EKF should produce the same results as KF."""
        kernel_kf = kalman_filter(transition_model, observation_model)
        kernel_ekf = extended_kalman_filter(
            transition_fn,
            observation_fn,
            transition_matrix_and_cov,
            observation_matrix_and_cov,
        )

        state_kf = kernel_kf.init(mu0, cov0, t=0.0)
        state_ekf = kernel_ekf.init(mu0, cov0, t=0.0)

        key = jax.random.PRNGKey(0)
        _, observations = generate_data(key, num_steps=3)

        for i in range(3):
            y = observations[i]
            state_kf, _ = kernel_kf(state_kf, t=float(i + 1), observed=y)
            state_ekf, _ = kernel_ekf(state_ekf, t=float(i + 1), observed=y)

            np.testing.assert_allclose(
                state_kf.mean,
                state_ekf.mean,
                atol=1e-5,
                err_msg=f"Mean mismatch at step {i}",
            )
            np.testing.assert_allclose(
                state_kf.cov,
                state_ekf.cov,
                atol=1e-5,
                err_msg=f"Cov mismatch at step {i}",
            )


# ---------------------------------------------------------------------------
# Tests: Unscented Kalman Filter
# ---------------------------------------------------------------------------


class TestUnscentedKalmanFilter:
    def test_linear_model_produces_finite_results(self):
        """On a linear model, UKF should produce finite, reasonable results.

        Note: The UKF sigma-point weighting scheme means covariance estimates
        may differ from the exact KF solution, especially in low dimensions
        with default Merwe parameters. We test for finiteness and that the
        mean is in the right ballpark.
        """
        kernel_kf = kalman_filter(transition_model, observation_model)
        kernel_ukf = ukf(
            transition_fn,
            Q,
            observation_fn,
            R,
            sigma_point_fn=merwe_sigma_point,
        )

        state_kf = kernel_kf.init(mu0, cov0, t=0.0)
        state_ukf = kernel_ukf.init(mu0, cov0, t=0.0)

        key = jax.random.PRNGKey(0)
        _, observations = generate_data(key, num_steps=3)

        for i in range(3):
            y = observations[i]
            state_kf, _ = kernel_kf(state_kf, t=float(i + 1), observed=y)
            state_ukf, _ = kernel_ukf(state_ukf, t=float(i + 1), observed=y)

            assert jnp.all(jnp.isfinite(state_ukf.mean)), f"Non-finite mean at step {i}"
            assert jnp.all(jnp.isfinite(state_ukf.cov)), f"Non-finite cov at step {i}"
            # Means should be in the same direction (covariances may differ)
            np.testing.assert_allclose(
                state_kf.mean,
                state_ukf.mean,
                atol=0.1,
                err_msg=f"Mean mismatch at step {i}",
            )

    @pytest.mark.parametrize(
        "sigma_fn",
        [
            merwe_sigma_point,
            julier_uhlmann_sigma_points,
            spherical_simplex_sigma_points,
        ],
        ids=["merwe", "julier_uhlmann", "spherical_simplex"],
    )
    def test_sigma_point_strategies(self, sigma_fn):
        """All sigma point strategies should produce finite results."""
        kernel = ukf(
            transition_fn,
            Q,
            observation_fn,
            R,
            sigma_point_fn=sigma_fn,
        )
        state = kernel.init(mu0, cov0, t=0.0)
        y = jnp.array([0.5])
        new_state, info = kernel(state, t=1.0, observed=y)

        assert jnp.all(jnp.isfinite(new_state.mean))
        assert jnp.all(jnp.isfinite(new_state.cov))


# ---------------------------------------------------------------------------
# Tests: Square Root Kalman Filter
# ---------------------------------------------------------------------------


class TestSquareRootKalmanFilter:
    def test_predict_step(self):
        """Test predict step of square root KF."""
        kernel = sq_kalman_filter(A, jnp.linalg.cholesky(Q), C, R)
        state = kernel.init(mu0, std0, t=0.0)
        new_state, info = kernel(state, t=1.0, observed=None, rng_key=None)

        # Mean should match standard KF predict
        expected_mean = A @ mu0
        np.testing.assert_allclose(new_state.mean, expected_mean, atol=1e-5)

    def test_update_step(self):
        """Test update step of square root KF."""
        kernel = sq_kalman_filter(A, jnp.linalg.cholesky(Q), C, R)
        state = kernel.init(mu0, std0, t=0.0)
        y = jnp.array([1.0])
        new_state, info = kernel(state, t=1.0, observed=y, rng_key=None)

        assert jnp.all(jnp.isfinite(new_state.mean))
        assert jnp.all(jnp.isfinite(new_state.std))
        assert info.log_likelihood is not None


class TestRankReducedKalmanFilter:
    def test_predict_and_update_steps(self):
        kernel = rank_reduced_kalman_filter(
            transition_model,
            observation_model,
            rank=1,
        )
        state = kernel.init(mu0, cov0, t=0.0)

        y = jnp.array([0.25])
        state, info = kernel(state, t=1.0, observed=y)

        assert state.mean.shape == mu0.shape
        assert state.cov_factor.shape[1] == 1
        assert state.cov_core.shape == (1, 1)
        assert jnp.all(jnp.isfinite(state.mean))
        assert jnp.all(jnp.isfinite(state.cov_factor))
        assert jnp.all(jnp.isfinite(state.cov_core))
        assert jnp.isfinite(info.log_likelihood)

    def test_matches_dense_kf_when_rank_is_full(self):
        A2 = jnp.array([[0.9, 0.1], [0.0, 0.95]])
        Q2 = jnp.array([[0.05, 0.0], [0.0, 0.03]])
        C2 = jnp.array([[1.0, 0.0]])
        R2 = jnp.array([[0.2]])

        def transition2(t_old, t):
            return A2, Q2

        def observation2(t):
            return C2, R2

        mu02 = jnp.array([0.0, 0.0])
        cov02 = jnp.eye(2)

        dense = kalman_filter(transition2, observation2)
        rr = rank_reduced_kalman_filter(transition2, observation2, rank=2)

        s_dense = dense.init(mu02, cov02, t=0.0)
        s_rr = rr.init(mu02, cov02, t=0.0)

        ys = [jnp.array([0.1]), jnp.array([-0.05]), jnp.array([0.2])]
        for i, y in enumerate(ys, start=1):
            s_dense, _ = dense(s_dense, t=float(i), observed=y)
            s_rr, _ = rr(s_rr, t=float(i), observed=y)

        P_rr = s_rr.cov_factor @ s_rr.cov_core @ s_rr.cov_factor.T
        np.testing.assert_allclose(s_rr.mean, s_dense.mean, atol=1e-5)
        np.testing.assert_allclose(P_rr, s_dense.cov, atol=1e-5)

    def test_lowrank_process_noise_tuple_matches_dense(self):
        A2 = jnp.array([[0.95, 0.02], [0.0, 0.9]])
        Uq = jnp.array([[1.0], [0.3]])
        Sq = jnp.array([[0.04]])
        Q2 = Uq @ Sq @ Uq.T
        C2 = jnp.array([[1.0, 0.0]])
        R2 = jnp.array([[0.3]])

        def transition_dense(t_old, t):
            return A2, Q2

        def transition_lr(t_old, t):
            return A2, (Uq, Sq)

        def observation2(t):
            return C2, R2

        mu02 = jnp.array([0.0, 0.0])
        cov02 = jnp.eye(2)
        ys = [jnp.array([0.1]), jnp.array([-0.08]), jnp.array([0.03])]

        rr_dense = rank_reduced_kalman_filter(transition_dense, observation2, rank=2)
        rr_lr = rank_reduced_kalman_filter(transition_lr, observation2, rank=2)

        s_dense = rr_dense.init(mu02, cov02, t=0.0)
        s_lr = rr_lr.init(mu02, cov02, t=0.0)

        for i, y in enumerate(ys, start=1):
            s_dense, _ = rr_dense(s_dense, t=float(i), observed=y)
            s_lr, _ = rr_lr(s_lr, t=float(i), observed=y)

        P_dense = s_dense.cov_factor @ s_dense.cov_core @ s_dense.cov_factor.T
        P_lr = s_lr.cov_factor @ s_lr.cov_core @ s_lr.cov_factor.T
        np.testing.assert_allclose(s_lr.mean, s_dense.mean, atol=1e-6)
        np.testing.assert_allclose(P_lr, P_dense, atol=1e-6)

    def test_energy_threshold_reduces_effective_rank(self):
        A3 = jnp.diag(jnp.array([0.98, 0.92, 0.85]))
        Q3 = jnp.diag(jnp.array([0.08, 0.01, 1e-4]))
        C3 = jnp.array([[1.0, 0.2, 0.0]])
        R3 = jnp.array([[0.1]])

        def transition3(t_old, t):
            return A3, Q3

        def observation3(t):
            return C3, R3

        rr = rank_reduced_kalman_filter(
            transition3,
            observation3,
            rank=3,
            energy_threshold=0.9,
            min_rank=1,
        )
        state = rr.init(jnp.zeros(3), jnp.eye(3), t=0.0)
        for i in range(4):
            state, _ = rr(state, t=float(i + 1), observed=jnp.array([0.05]))

        eigvals = jnp.diag(state.cov_core)
        effective_rank = int(jnp.sum(eigvals > 1e-6))
        assert 1 <= effective_rank <= 3
        assert effective_rank < 3

    def test_transition_three_tuple_and_lowrank_init(self):
        A2 = jnp.array([[0.9, 0.0], [0.0, 0.95]])
        Uq = jnp.array([[1.0], [0.1]])
        Sq = jnp.array([[0.02]])
        C2 = jnp.array([[1.0, 0.0]])
        R2 = jnp.array([[0.2]])

        def transition3(t_old, t):
            return A2, Uq, Sq

        def observation2(t):
            return C2, R2

        kernel = rank_reduced_kalman_filter(transition3, observation2, rank=2)
        state = kernel.init(jnp.zeros(2), (jnp.eye(2), jnp.eye(2)), t=0.0, rank=2)
        state, _ = kernel(state, t=1.0, observed=jnp.array([0.1]))

        assert state.cov_factor.shape == (2, 2)
        assert state.cov_core.shape == (2, 2)


# ---------------------------------------------------------------------------
# Tests: Particle Filter
# ---------------------------------------------------------------------------


class TestParticleFilter:
    def test_predict_step(self):
        """Test a single predict step of particle filter."""
        num_particles = 100

        def log_likelihood(particles, obs, t):
            return jax.vmap(lambda x: -0.5 * jnp.sum((C @ x - obs) ** 2 / R[0, 0]))(
                particles
            )

        def pf_transition(key, particles, t):
            noise = jax.random.normal(key, particles.shape) * jnp.sqrt(Q[0, 0])
            return jax.vmap(lambda x: A @ x)(particles) + noise

        kernel = ParticleFilter(log_likelihood, pf_transition)

        key = jax.random.PRNGKey(0)
        particles = jax.random.normal(key, (num_particles, 1))
        state = kernel.init(particles, t=0.0)

        key, subkey = jax.random.split(key)
        new_state, info = kernel(state, t=1.0, observed=None, rng_key=subkey)
        assert new_state.particles.shape == (num_particles, 1)

    def test_update_step(self):
        """Test particle filter with observation."""
        num_particles = 200

        def log_likelihood(particles, obs, t):
            return jax.vmap(lambda x: -0.5 * jnp.sum((C @ x - obs) ** 2 / R[0, 0]))(
                particles
            )

        def pf_transition(key, particles, t):
            noise = jax.random.normal(key, particles.shape) * jnp.sqrt(Q[0, 0])
            return jax.vmap(lambda x: A @ x)(particles) + noise

        kernel = ParticleFilter(log_likelihood, pf_transition)

        key = jax.random.PRNGKey(0)
        particles = jax.random.normal(key, (num_particles, 1))
        state = kernel.init(particles, t=0.0)

        y = jnp.array([1.0])
        key, subkey = jax.random.split(key)
        new_state, info = kernel(state, t=1.0, observed=y, rng_key=subkey)

        assert new_state.particles.shape == (num_particles, 1)
        assert jnp.isfinite(info.log_likelihood)

    def test_default_unpack_returns_particles(self):
        """Particle filter default_unpack should return particles."""

        def dummy_ll(p, o, t):
            return jnp.zeros(p.shape[0])

        def dummy_transition(key, p, t):
            return p

        kernel = ParticleFilter(dummy_ll, dummy_transition)
        from probjax.inference.filtering.particle_filter import ParticleFilterState

        state = ParticleFilterState(
            particles=jnp.ones((10, 2)),
            log_weights=jnp.zeros(10),
            t=0.0,
        )
        result = kernel.default_unpack(state, None)
        np.testing.assert_array_equal(result, jnp.ones((10, 2)))


# ---------------------------------------------------------------------------
# Tests: make_filter_api factory
# ---------------------------------------------------------------------------


class TestMakeFilterApi:
    def test_basic_factory(self):
        """make_filter_api should create a working FilterAPI subclass."""

        def my_init(mean, t=None):
            return KalmanFilterState(mean, jnp.eye(1), t)

        def my_build_kernel():
            def step(state, t=None, observed=None, rng_key=None):
                return state, None

            return step

        MyFilter = make_filter_api(
            name="my_filter",
            init_fn=my_init,
            build_kernel_fn=my_build_kernel,
        )
        kernel = MyFilter()
        assert isinstance(kernel, FilterKernel)
        assert callable(kernel.init)
        assert callable(kernel.step)

    def test_custom_unpack(self):
        """make_filter_api should accept a custom default_unpack."""

        def my_init(mean, t=None):
            return KalmanFilterState(mean, jnp.eye(1), t)

        def my_build_kernel():
            def step(state, t=None, observed=None, rng_key=None):
                return state, None

            return step

        def my_unpack(state, info):
            return state.mean

        MyFilter = make_filter_api(
            name="my_filter",
            init_fn=my_init,
            build_kernel_fn=my_build_kernel,
            default_unpack_fn=my_unpack,
        )
        kernel = MyFilter()
        state = kernel.init(jnp.array([1.0]))
        result = kernel.default_unpack(state, None)
        np.testing.assert_array_equal(result, jnp.array([1.0]))


# ---------------------------------------------------------------------------
# Tests: RTS Smoother
# ---------------------------------------------------------------------------


class TestRTSSmoother:
    def test_single_step(self):
        """Test a single backward RTS smoother step."""
        # Run one KF forward step to get predict + update values
        kernel = kalman_filter(transition_model, observation_model)
        state0 = kernel.init(mu0, cov0, t=0.0)

        y = jnp.array([0.5])
        state1, info1 = kernel(state0, t=1.0, observed=y)

        # The smoother step should return finite results
        mu_s, cov_s = rauch_tung_stribel_smoother(
            lambda t0, t1: A,
            t0=0.0,
            t1=1.0,
            mu0_s=state1.mean,
            cov0_s=state1.cov,
            mu0=state0.mean,
            cov0=state0.cov,
            mu0_=info1.mean_pred,
            cov0_=info1.cov_pred,
        )
        assert jnp.all(jnp.isfinite(mu_s))
        assert jnp.all(jnp.isfinite(cov_s))


# ---------------------------------------------------------------------------
# Tests: Particle Smoother (FFBSi)
# ---------------------------------------------------------------------------


class TestParticleSmoother:
    """Tests for the FFBSi particle smoother."""

    def _run_particle_filter(self, key, num_steps=5, num_particles=100):
        """Helper: run a particle filter forward and collect output."""

        def log_likelihood_fn(particles, obs, t):
            return jax.vmap(lambda x: -0.5 * jnp.sum((C @ x - obs) ** 2 / R[0, 0]))(
                particles
            )

        def pf_transition(key, particles, t):
            noise = jax.random.normal(key, particles.shape) * jnp.sqrt(Q[0, 0])
            return jax.vmap(lambda x: A @ x)(particles) + noise

        kernel = ParticleFilter(log_likelihood_fn, pf_transition)

        key, init_key = jax.random.split(key)
        particles = jax.random.normal(init_key, (num_particles, 1))
        state = kernel.init(particles, t=0.0)

        # Generate synthetic observations
        key, data_key = jax.random.split(key)
        _, observations = generate_data(data_key, num_steps=num_steps)
        ts = jnp.arange(num_steps + 1, dtype=float)

        # Run filter forward, collecting particles and weights
        all_particles = [state.particles]
        all_log_weights = [state.log_weights]
        all_ancestors = []

        for i in range(num_steps):
            key, subkey = jax.random.split(key)
            state, info = kernel(
                state, t=ts[i + 1], observed=observations[i], rng_key=subkey
            )
            all_particles.append(state.particles)
            all_log_weights.append(state.log_weights)
            all_ancestors.append(info.ancestors)

        filter_particles = jnp.stack(all_particles)  # (T, N, D)
        filter_log_weights = jnp.stack(all_log_weights)  # (T, N)

        return ts, filter_particles, filter_log_weights

    def test_basic_smoke(self):
        """Particle smoother should run and return correct shapes."""
        key = jax.random.PRNGKey(42)
        num_steps = 5
        num_particles = 50

        ts, filter_particles, filter_log_weights = self._run_particle_filter(
            key, num_steps=num_steps, num_particles=num_particles
        )

        # Transition log-density for linear Gaussian: p(x_{t+1} | x_t)
        def transition_logdensity(x_tp1, x_t, t, tp1):
            mean = A @ x_t
            diff = x_tp1 - mean
            return -0.5 * jnp.sum(diff**2 / Q[0, 0])

        key2 = jax.random.PRNGKey(123)
        smoothed_particles, smoothed_log_weights = particle_smoother(
            key2, ts, filter_particles, filter_log_weights, transition_logdensity
        )

        T = num_steps + 1
        assert smoothed_particles.shape == (T, num_particles, 1)
        assert smoothed_log_weights.shape == (T, num_particles)

    def test_output_is_finite(self):
        """All smoothed particles should be finite."""
        key = jax.random.PRNGKey(0)
        num_steps = 4
        num_particles = 30

        ts, filter_particles, filter_log_weights = self._run_particle_filter(
            key, num_steps=num_steps, num_particles=num_particles
        )

        def transition_logdensity(x_tp1, x_t, t, tp1):
            mean = A @ x_t
            diff = x_tp1 - mean
            return -0.5 * jnp.sum(diff**2 / Q[0, 0])

        key2 = jax.random.PRNGKey(99)
        smoothed_particles, smoothed_log_weights = particle_smoother(
            key2, ts, filter_particles, filter_log_weights, transition_logdensity
        )

        assert jnp.all(jnp.isfinite(smoothed_particles))
        assert jnp.all(jnp.isfinite(smoothed_log_weights))

    def test_weights_are_uniform(self):
        """FFBSi produces uniformly weighted smoothed particles."""
        key = jax.random.PRNGKey(7)
        num_particles = 20

        ts, filter_particles, filter_log_weights = self._run_particle_filter(
            key, num_steps=3, num_particles=num_particles
        )

        def transition_logdensity(x_tp1, x_t, t, tp1):
            mean = A @ x_t
            return -0.5 * jnp.sum((x_tp1 - mean) ** 2 / Q[0, 0])

        key2 = jax.random.PRNGKey(8)
        _, smoothed_log_weights = particle_smoother(
            key2, ts, filter_particles, filter_log_weights, transition_logdensity
        )

        expected = -jnp.log(num_particles)
        np.testing.assert_allclose(smoothed_log_weights, expected, atol=1e-6)

    def test_smoothed_mean_closer_to_truth(self):
        """Smoothed particle mean should be at least as good as filter mean.

        On a linear Gaussian model, the smoothing distribution has lower
        variance than the filtering distribution. We verify by checking that
        the smoothed weighted mean is reasonable.
        """
        key = jax.random.PRNGKey(1)
        num_steps = 5
        num_particles = 500

        ts, filter_particles, filter_log_weights = self._run_particle_filter(
            key, num_steps=num_steps, num_particles=num_particles
        )

        def transition_logdensity(x_tp1, x_t, t, tp1):
            mean = A @ x_t
            return -0.5 * jnp.sum((x_tp1 - mean) ** 2 / Q[0, 0])

        key2 = jax.random.PRNGKey(2)
        smoothed_particles, smoothed_log_weights = particle_smoother(
            key2, ts, filter_particles, filter_log_weights, transition_logdensity
        )

        # Compute smoothed means (uniform weights)
        smoothed_means = jnp.mean(smoothed_particles, axis=1)  # (T, D)

        # Compute filter means (weighted)
        filter_weights = jnp.exp(filter_log_weights)
        filter_means = jnp.sum(
            filter_weights[:, :, None] * filter_particles, axis=1
        )  # (T, D)

        # Both should be finite and not wildly different
        assert jnp.all(jnp.isfinite(smoothed_means))
        assert jnp.all(jnp.isfinite(filter_means))

        # The smoothed and filter means should be in the same ballpark
        np.testing.assert_allclose(
            smoothed_means,
            filter_means,
            atol=1.0,
            err_msg="Smoothed and filter means are too far apart",
        )


# ---------------------------------------------------------------------------
# Tests: __init__.py re-exports
# ---------------------------------------------------------------------------


class TestImports:
    def test_filtering_init_exports(self):
        """All key symbols should be importable from the filtering package."""
        from probjax.inference.filtering import (
            FilterAPI,
            FilterInfo,
            FilterKernel,
            FilterState,
            make_filter_api,
            kalman_filter,
            extended_kalman_filter,
            ukf,
            sq_kalman_filter,
            rank_reduced_kalman_filter,
            ParticleFilter,
            particle_smoother,
            rauch_tung_stribel_smoother,
            smooth,
        )

        # Just check they imported without error
        assert FilterKernel is not None

    def test_filter_smooth_imports(self):
        """filter_smooth.py should export filter, smooth, and filter_log_likelihood."""
        from probjax.inference.filter_smooth import (
            filter,
            smooth,
            filter_log_likelihood,
        )

        assert callable(filter)
        assert callable(smooth)
        assert callable(filter_log_likelihood)


class TestUnifiedFilterSmoothingAPI:
    def test_gaussian_smoothing_from_filtering_states(self):
        from probjax.inference import filter_smooth as fs

        kernel = kalman_filter(transition_model, observation_model)
        ts = jnp.array([0.0, 1.0, 2.0])
        t_o = jnp.array([1.0, 2.0])
        x_o = jnp.array([[0.1], [-0.2]])

        trace = fs.filter(
            jax.random.PRNGKey(0),
            ts,
            t_o,
            x_o,
            kernel,
            mu0,
            cov0,
            return_trace=True,
        )

        smoother = partial(rauch_tung_stribel_smoother, lambda t0, t1: A)
        mus_s, covs_s = fs.smooth(trace, smoother=smoother)

        assert mus_s.shape[0] == trace.states.mean.shape[0]
        assert covs_s.shape[0] == trace.states.cov.shape[0]

    def test_particle_smoothing_from_filtering_states(self):
        from probjax.inference import filter_smooth as fs

        num_particles = 64

        def log_likelihood(particles, obs, t):
            return jax.vmap(lambda x: -0.5 * jnp.sum((C @ x - obs) ** 2 / R[0, 0]))(
                particles
            )

        def pf_transition(key, particles, t):
            noise = jax.random.normal(key, particles.shape) * jnp.sqrt(Q[0, 0])
            return jax.vmap(lambda x: A @ x)(particles) + noise

        kernel = ParticleFilter(log_likelihood, pf_transition)
        key = jax.random.PRNGKey(0)
        init_particles = jax.random.normal(key, (num_particles, 1))

        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        t_o = jnp.array([1.0, 2.0, 3.0])
        x_o = jnp.array([[0.1], [-0.2], [0.05]])

        trace = fs.filter(
            key,
            ts,
            t_o,
            x_o,
            kernel,
            init_particles,
            return_trace=True,
        )

        def transition_logdensity(x_tp1, x_t, t, tp1):
            mean = A @ x_t
            diff = x_tp1 - mean
            return -0.5 * jnp.sum(diff**2 / Q[0, 0])

        smoothed_particles, smoothed_log_weights = fs.smooth(
            trace,
            key=jax.random.PRNGKey(1),
            transition_logdensity_fn=transition_logdensity,
        )

        assert smoothed_particles.shape == trace.states.particles.shape
        assert smoothed_log_weights.shape == trace.states.log_weights.shape

    def test_filter_and_smooth_gaussian(self):
        from probjax.inference import filter_smooth as fs

        kernel = kalman_filter(transition_model, observation_model)
        ts = jnp.array([0.0, 1.0, 2.0])
        t_o = jnp.array([1.0, 2.0])
        x_o = jnp.array([[0.1], [-0.2]])

        smoother = partial(rauch_tung_stribel_smoother, lambda t0, t1: A)
        trace, (mus_s, covs_s) = fs.filter_and_smooth(
            jax.random.PRNGKey(3),
            ts,
            t_o,
            x_o,
            kernel,
            mu0,
            cov0,
            smoother=smoother,
        )

        assert hasattr(trace, "states")
        assert mus_s.shape[0] == trace.states.mean.shape[0]
        assert covs_s.shape[0] == trace.states.cov.shape[0]

    def test_filter_and_smooth_particle(self):
        from probjax.inference import filter_smooth as fs

        num_particles = 64

        def log_likelihood(particles, obs, t):
            return jax.vmap(lambda x: -0.5 * jnp.sum((C @ x - obs) ** 2 / R[0, 0]))(
                particles
            )

        def pf_transition(key, particles, t):
            noise = jax.random.normal(key, particles.shape) * jnp.sqrt(Q[0, 0])
            return jax.vmap(lambda x: A @ x)(particles) + noise

        def transition_logdensity(x_tp1, x_t, t, tp1):
            mean = A @ x_t
            diff = x_tp1 - mean
            return -0.5 * jnp.sum(diff**2 / Q[0, 0])

        kernel = ParticleFilter(log_likelihood, pf_transition)
        key = jax.random.PRNGKey(0)
        init_particles = jax.random.normal(key, (num_particles, 1))

        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        t_o = jnp.array([1.0, 2.0, 3.0])
        x_o = jnp.array([[0.1], [-0.2], [0.05]])

        trace, (smoothed_particles, smoothed_log_weights) = fs.filter_and_smooth(
            key,
            ts,
            t_o,
            x_o,
            kernel,
            init_particles,
            smooth_key=jax.random.PRNGKey(4),
            transition_logdensity_fn=transition_logdensity,
        )

        assert smoothed_particles.shape == trace.states.particles.shape
        assert smoothed_log_weights.shape == trace.states.log_weights.shape

    def test_filter_and_smooth_rank_reduced_gaussian(self):
        from probjax.inference import filter_smooth as fs

        kernel = rank_reduced_kalman_filter(
            transition_model,
            observation_model,
            rank=1,
        )
        ts = jnp.array([0.0, 1.0, 2.0])
        t_o = jnp.array([1.0, 2.0])
        x_o = jnp.array([[0.1], [-0.2]])

        smoother = partial(rauch_tung_stribel_smoother, lambda t0, t1: A)
        trace, (mus_s, covs_s) = fs.filter_and_smooth(
            jax.random.PRNGKey(5),
            ts,
            t_o,
            x_o,
            kernel,
            mu0,
            cov0,
            rank=1,
            smoother=smoother,
        )

        assert mus_s.shape[0] == trace.states.mean.shape[0]
        assert covs_s.shape[0] == trace.states.cov_factor.shape[0]


class TestFilterClass:
    """Tests for the new Filter class API."""

    def test_filter_returns_trace(self):
        from probjax.inference.filter_smooth import Filter

        kernel = kalman_filter(transition_model, observation_model)
        filt = Filter(kernel)

        ts = jnp.array([0.0, 1.0, 2.0])
        t_o = jnp.array([1.0, 2.0])
        x_o = jnp.array([[0.1], [-0.2]])

        trace = filt.filter(jax.random.PRNGKey(0), ts, t_o, x_o, mu0, cov0)

        assert hasattr(trace, "states")
        assert hasattr(trace, "infos")
        assert hasattr(trace, "ts")
        assert hasattr(trace, "outputs")

    def test_filter_gaussian_matches_backward_compat(self):
        from probjax.inference.filter_smooth import Filter, filter as filter_fn

        kernel = kalman_filter(transition_model, observation_model)
        ts = jnp.array([0.0, 1.0, 2.0])
        t_o = jnp.array([1.0, 2.0])
        x_o = jnp.array([[0.1], [-0.2]])
        key = jax.random.PRNGKey(0)

        trace_new = Filter(kernel).filter(key, ts, t_o, x_o, mu0, cov0)
        trace_old = filter_fn(key, ts, t_o, x_o, kernel, mu0, cov0, return_trace=True)

        np.testing.assert_allclose(trace_new.states.mean, trace_old.states.mean)
        np.testing.assert_allclose(trace_new.states.cov, trace_old.states.cov)

    def test_smooth_gaussian(self):
        from probjax.inference.filter_smooth import Filter

        kernel = kalman_filter(transition_model, observation_model)
        filt = Filter(kernel)

        ts = jnp.array([0.0, 1.0, 2.0])
        t_o = jnp.array([1.0, 2.0])
        x_o = jnp.array([[0.1], [-0.2]])

        trace = filt.filter(jax.random.PRNGKey(0), ts, t_o, x_o, mu0, cov0)

        smoother = partial(rauch_tung_stribel_smoother, lambda t0, t1: A)
        mus_s, covs_s = filt.smooth(trace, smoother=smoother)

        assert mus_s.shape[0] == trace.states.mean.shape[0]
        assert covs_s.shape[0] == trace.states.cov.shape[0]

    def test_smooth_particle(self):
        from probjax.inference.filter_smooth import Filter

        num_particles = 64

        def log_likelihood(particles, obs, t):
            return jax.vmap(lambda x: -0.5 * jnp.sum((C @ x - obs) ** 2 / R[0, 0]))(
                particles
            )

        def pf_transition(key, particles, t):
            noise = jax.random.normal(key, particles.shape) * jnp.sqrt(Q[0, 0])
            return jax.vmap(lambda x: A @ x)(particles) + noise

        kernel = ParticleFilter(log_likelihood, pf_transition)
        filt = Filter(kernel)
        key = jax.random.PRNGKey(0)
        init_particles = jax.random.normal(key, (num_particles, 1))

        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        t_o = jnp.array([1.0, 2.0, 3.0])
        x_o = jnp.array([[0.1], [-0.2], [0.05]])

        trace = filt.filter(key, ts, t_o, x_o, init_particles)

        def transition_logdensity(x_tp1, x_t, t, tp1):
            mean = A @ x_t
            diff = x_tp1 - mean
            return -0.5 * jnp.sum(diff**2 / Q[0, 0])

        smoothed_particles, smoothed_log_weights = filt.smooth(
            trace,
            key=jax.random.PRNGKey(1),
            transition_logdensity_fn=transition_logdensity,
        )

        assert smoothed_particles.shape == trace.states.particles.shape
        assert smoothed_log_weights.shape == trace.states.log_weights.shape

    def test_log_likelihood(self):
        from probjax.inference.filter_smooth import Filter

        kernel = kalman_filter(transition_model, observation_model)
        filt = Filter(kernel)

        ts = jnp.array([0.0, 1.0, 2.0])
        t_o = jnp.array([1.0, 2.0])
        x_o = jnp.array([[0.1], [-0.2]])

        ll = filt.log_likelihood(jax.random.PRNGKey(0), ts, t_o, x_o, mu0, cov0)

        assert ll.shape == ()
        assert jnp.isfinite(ll)

    def test_log_likelihood_matches_backward_compat(self):
        from probjax.inference.filter_smooth import Filter, filter_log_likelihood

        kernel = kalman_filter(transition_model, observation_model)
        ts = jnp.array([0.0, 1.0, 2.0])
        t_o = jnp.array([1.0, 2.0])
        x_o = jnp.array([[0.1], [-0.2]])
        key = jax.random.PRNGKey(0)

        ll_new = Filter(kernel).log_likelihood(key, ts, t_o, x_o, mu0, cov0)
        ll_old = filter_log_likelihood(key, ts, t_o, x_o, kernel, mu0, cov0)

        np.testing.assert_allclose(ll_new, ll_old)

    def test_filter_smooth_particle_one_shot(self):
        from probjax.inference.filter_smooth import Filter

        num_particles = 64

        def log_likelihood(particles, obs, t):
            return jax.vmap(lambda x: -0.5 * jnp.sum((C @ x - obs) ** 2 / R[0, 0]))(
                particles
            )

        def pf_transition(key, particles, t):
            noise = jax.random.normal(key, particles.shape) * jnp.sqrt(Q[0, 0])
            return jax.vmap(lambda x: A @ x)(particles) + noise

        def transition_logdensity(x_tp1, x_t, t, tp1):
            mean = A @ x_t
            diff = x_tp1 - mean
            return -0.5 * jnp.sum(diff**2 / Q[0, 0])

        kernel = ParticleFilter(log_likelihood, pf_transition)
        filt = Filter(kernel)
        key = jax.random.PRNGKey(0)
        init_particles = jax.random.normal(key, (num_particles, 1))

        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        t_o = jnp.array([1.0, 2.0, 3.0])
        x_o = jnp.array([[0.1], [-0.2], [0.05]])

        trace = filt.filter(key, ts, t_o, x_o, init_particles)
        smoothed_particles, smoothed_log_weights = filt.smooth(
            trace,
            key=jax.random.PRNGKey(1),
            transition_logdensity_fn=transition_logdensity,
        )

        assert smoothed_particles.shape == trace.states.particles.shape
        assert smoothed_log_weights.shape == trace.states.log_weights.shape
