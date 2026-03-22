import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.inference.mcmc import (
    adjusted_mclmc,
    adjusted_mclmc_dynamic,
    dynamic_hmc,
    elliptical_slice,
    gauss_rwmh,
    gaussian_imh,
    hmc,
    latent_slice,
    mala,
    mclmc,
    nuts,
    sgld,
    sghmc,
    sgnht,
    slice,
)
from probjax.inference.mcmc.pmmcmc import pseudo_marginal
from probjax.inference.mcmc.sgmcmc import grad_estimator
from probjax.inference.mcmc_runner import MCMC
from probjax.inference.smc import (
    smc,
    persistent_smc_kernel,
    adaptive_persistent_smc_kernel,
)
from probjax.inference.smc.path import GeometricPath, PartialPosteriorsPath
from probjax.inference.smc_runner import SMC

KERNELS = [
    hmc,
    nuts,
    mclmc,
    adjusted_mclmc,
    adjusted_mclmc_dynamic,
    gaussian_imh,
    gauss_rwmh,
    mala,
    dynamic_hmc,
    slice,
    latent_slice,
    elliptical_slice,
]
SHAPES = [(1,), (2,), (2, 3), (4, 5, 6)]


@pytest.mark.parametrize(
    "kernel_type,in_shape", [(kernel, s) for kernel in KERNELS for s in SHAPES]
)
def test_markov_kernel_vector_input(kernel_type, in_shape):
    if kernel_type is mclmc and in_shape == (1,):
        return
    i = np.random.randint(0, 2**16)
    xs = np.random.randn(*in_shape)

    def logdensity(x):
        return -jnp.sum(x**2)

    if kernel_type is elliptical_slice:
        kernel = kernel_type(
            logdensity, cov_matrix=jnp.eye(xs.size), mean=jnp.zeros(xs.size)
        )
    else:
        kernel = kernel_type(logdensity)
    state = kernel.init(xs, rng_key=jax.random.PRNGKey(i))
    params = kernel.init_params(state)

    assert hasattr(state, "position"), "State must have a position attribute"
    assert state.position.shape == in_shape, "Position shape must match input shape"

    next_state, next_info = kernel(jax.random.PRNGKey(i), state, params)

    assert hasattr(next_state, "position"), "State must have a position attribute"
    assert next_state.position.shape == in_shape, (
        "Position shape must match input shape after transition"
    )


@pytest.mark.parametrize("kernel_type", [hmc, nuts, mala, gauss_rwmh])
def test_markov_kernel_fit_params_sanity(kernel_type):
    key = jax.random.PRNGKey(0)
    x0 = jnp.array([0.1, -0.2])

    def logdensity(x):
        return -0.5 * jnp.sum(x**2)

    if kernel_type is hmc:
        kernel = kernel_type(logdensity, num_integration_steps=5)
    elif kernel_type is nuts:
        kernel = kernel_type(logdensity, max_num_doublings=5)
    else:
        kernel = kernel_type(logdensity)
    state = kernel.init(x0)
    params = kernel.init_params(state)

    new_state, new_params = kernel.fit_params(key, state, params, num_steps=10)

    assert hasattr(new_state, "position")
    assert new_state.position.shape == x0.shape
    assert isinstance(new_params, type(params))


@pytest.mark.parametrize(
    "kernel_type,in_shape",
    [(kernel, s) for kernel in KERNELS for s in [(1,), (2,), (2, 2)]],
)
def test_markov_kernel_invariance(kernel_type, in_shape):
    if kernel_type is mclmc:
        return

    i = np.random.randint(0, 2**16)
    key = jax.random.PRNGKey(i)

    N = 5000

    positions = np.random.randn(N, *in_shape)
    if kernel_type is elliptical_slice:
        # Elliptical slice uses a Gaussian prior; use a flat likelihood so
        # the target is the prior N(0, I).
        def logdensity(x):
            return jnp.array(0.0)

        kernel = kernel_type(
            logdensity,
            cov_matrix=jnp.eye(positions[0].size),
            mean=jnp.zeros(positions[0].size),
        )
    else:
        # Gaussian target, hence must leave the particle distribution invariant
        def logdensity(x):
            return jax.scipy.stats.norm.logpdf(x).sum()

        kernel = kernel_type(logdensity)
    states = jax.vmap(lambda x: kernel.init(x, rng_key=key))(positions)
    state0 = jax.tree_util.tree_map(lambda x: x[0], states)
    if kernel_type is gaussian_imh:
        flat_position, _ = jax.flatten_util.ravel_pytree(state0.position)
        dim = flat_position.shape[0]
        params = kernel.init_params(
            state0,
            mean=jnp.zeros_like(flat_position),
            cov=jnp.eye(dim, dtype=flat_position.dtype),
        )
    else:
        params = kernel.init_params(state0)

    keys = jax.random.split(key, (N,))
    next_states, _ = jax.vmap(kernel, in_axes=(0, 0, None))(keys, states, params)

    new_positions = next_states.position

    assert new_positions.shape == positions.shape, "Output shape must match input shape"

    # Must be invariant to input
    mean_before = jnp.mean(positions, axis=0)
    mean_after = jnp.mean(new_positions, axis=0)

    assert jnp.allclose(mean_before, mean_after, atol=1e-1, rtol=1e-1), (
        "Mean must be invariant"
    )

    var_before = jnp.var(positions, axis=0)
    var_after = jnp.var(new_positions, axis=0)

    assert jnp.allclose(var_before, var_after, atol=1e-1, rtol=1e-1), (
        "Variance must be invariant"
    )


def test_smc_geometric_path_step():
    key = jax.random.PRNGKey(0)
    particles = jax.random.normal(key, (32, 2))

    def logprior_fn(x):
        return -0.5 * jnp.sum(x**2)

    def loglikelihood_fn(x):
        return -0.5 * jnp.sum((x - 1.0) ** 2)

    kernel = smc(
        path=GeometricPath(),
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_mcmc_steps=3,
        num_integration_steps=3,
    )

    state = kernel.init(particles, path=GeometricPath())
    params = kernel.init_params(
        particles,
        path=GeometricPath(),
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_integration_steps=3,
        step_size=0.1,
    )

    next_state, info = kernel.step(
        key, state, tempering_param=0.2, mcmc_parameters=params
    )

    assert next_state.particles.shape == particles.shape
    assert next_state.weights.shape[0] == particles.shape[0]
    assert np.isfinite(np.asarray(next_state.weights)).all()
    assert info is not None


def test_smc_partial_posteriors_path_step():
    key = jax.random.PRNGKey(1)
    particles = jax.random.normal(key, (16, 1))
    num_datapoints = 5
    data = jnp.linspace(-1.0, 1.0, num_datapoints)

    def partial_logposterior_factory(mask):
        mask = mask.astype(jnp.float32)

        def logposterior(x):
            x = x.reshape(())
            logprior = -0.5 * x**2
            loglik = -0.5 * jnp.sum(mask * (x - data) ** 2)
            return logprior + loglik

        return logposterior

    kernel = smc(
        path=PartialPosteriorsPath(),
        partial_logposterior_factory=partial_logposterior_factory,
        num_datapoints=num_datapoints,
        mcmc_kernel=hmc,
        num_mcmc_steps=2,
        num_integration_steps=2,
    )

    state = kernel.init(
        particles,
        path=PartialPosteriorsPath(),
        num_datapoints=num_datapoints,
    )
    params = kernel.init_params(
        particles,
        path=PartialPosteriorsPath(),
        partial_logposterior_factory=partial_logposterior_factory,
        num_datapoints=num_datapoints,
        mcmc_kernel=hmc,
        num_integration_steps=2,
        step_size=0.1,
    )

    mask = jnp.array([1, 0, 1, 0, 1], dtype=jnp.float32)
    next_state, info = kernel.step(
        key, state, tempering_param=mask, mcmc_parameters=params
    )

    assert next_state.particles.shape == particles.shape
    assert next_state.weights.shape[0] == particles.shape[0]
    assert np.isfinite(np.asarray(next_state.weights)).all()
    assert info is not None


def test_smc_runner_run():
    key = jax.random.PRNGKey(0)
    particles = jax.random.normal(key, (16, 2))

    def logprior_fn(x):
        return -0.5 * jnp.sum(x**2)

    def loglikelihood_fn(x):
        return -0.5 * jnp.sum((x - 1.0) ** 2)

    kernel = smc(
        path=GeometricPath(),
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_mcmc_steps=2,
        num_integration_steps=2,
    )

    state = kernel.init(particles, path=GeometricPath())
    params = kernel.init_params(
        particles,
        path=GeometricPath(),
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_integration_steps=2,
        step_size=0.1,
    )

    runner = SMC(kernel)
    temps = jnp.array([0.2, 0.5])
    final_state, final_params = runner.run(key, state, temps, params)

    assert final_state.particles.shape == particles.shape
    assert final_state.weights.shape[0] == particles.shape[0]
    assert final_state.tempering_param == temps[-1]


def test_smc_runner_sample_with_tuning():
    key = jax.random.PRNGKey(1)
    particles = jax.random.normal(key, (8, 1))

    def logprior_fn(x):
        return -0.5 * jnp.sum(x**2)

    def loglikelihood_fn(x):
        return -0.5 * jnp.sum((x - 0.5) ** 2)

    kernel = smc(
        path=GeometricPath(),
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_mcmc_steps=1,
        num_integration_steps=2,
    )

    state = kernel.init(particles, path=GeometricPath())
    params = kernel.init_params(
        particles,
        path=GeometricPath(),
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_integration_steps=2,
        step_size=0.1,
    )

    runner = SMC(kernel)
    temps = jnp.array([0.1, 0.3, 0.6])
    particles_hist, weights_hist, final_state, final_params = runner.sample(
        key, state, temps, params, tune_params=True
    )

    assert particles_hist.shape[0] == temps.shape[0]
    assert weights_hist.shape[0] == temps.shape[0]
    assert final_state.tempering_param == temps[-1]


def test_persistent_smc_step():
    key = jax.random.PRNGKey(0)
    num_particles = 16
    particles = jax.random.normal(key, (num_particles, 2))
    n_schedule = 3

    def logprior_fn(x):
        return -0.5 * jnp.sum(x**2)

    def loglikelihood_fn(x):
        return -0.5 * jnp.sum((x - 1.0) ** 2)

    kernel = persistent_smc_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_mcmc_steps=2,
        num_integration_steps=2,
    )

    state = kernel.init(particles, n_schedule=n_schedule)
    params = kernel.init_params(
        particles,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_integration_steps=2,
        step_size=0.1,
    )

    next_state, info = kernel.step(
        key, state, tempering_param=0.3, mcmc_parameters=params
    )

    assert next_state.particles.shape == particles.shape
    assert next_state.tempering_param == 0.3
    assert info is not None


def test_persistent_smc_multiple_steps():
    key = jax.random.PRNGKey(1)
    num_particles = 8
    particles = jax.random.normal(key, (num_particles, 1))
    n_schedule = 5

    def logprior_fn(x):
        return -0.5 * jnp.sum(x**2)

    def loglikelihood_fn(x):
        return -0.5 * jnp.sum((x - 0.5) ** 2)

    kernel = persistent_smc_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_mcmc_steps=1,
        num_integration_steps=2,
    )

    state = kernel.init(particles, n_schedule=n_schedule)
    params = kernel.init_params(
        particles,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_integration_steps=2,
        step_size=0.1,
    )

    temps = [0.2, 0.5, 0.8]
    for t in temps:
        key, subkey = jax.random.split(key)
        state, info = kernel.step(
            subkey, state, tempering_param=t, mcmc_parameters=params
        )

    assert state.particles.shape == particles.shape
    assert state.tempering_param == temps[-1]
    assert state.iteration == len(temps)


def test_adaptive_persistent_smc_step():
    key = jax.random.PRNGKey(0)
    num_particles = 16
    particles = jax.random.normal(key, (num_particles, 2))
    max_iterations = 10

    def logprior_fn(x):
        return -0.5 * jnp.sum(x**2)

    def loglikelihood_fn(x):
        return -0.5 * jnp.sum((x - 1.0) ** 2)

    kernel = adaptive_persistent_smc_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_mcmc_steps=2,
        num_integration_steps=2,
        target_ess=0.5,
    )

    state = kernel.init(particles, max_iterations=max_iterations)
    params = kernel.init_params(
        particles,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_integration_steps=2,
        step_size=0.1,
    )

    next_state, info = kernel.step(key, state, mcmc_parameters=params)

    assert next_state.particles.shape == particles.shape
    assert next_state.tempering_param > 0.0
    assert info is not None


def test_adaptive_persistent_smc_multiple_steps():
    key = jax.random.PRNGKey(1)
    num_particles = 8
    particles = jax.random.normal(key, (num_particles, 1))
    max_iterations = 10

    def logprior_fn(x):
        return -0.5 * jnp.sum(x**2)

    def loglikelihood_fn(x):
        return -0.5 * jnp.sum((x - 0.5) ** 2)

    kernel = adaptive_persistent_smc_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_mcmc_steps=1,
        num_integration_steps=2,
        target_ess=0.5,
    )

    state = kernel.init(particles, max_iterations=max_iterations)
    params = kernel.init_params(
        particles,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_kernel=hmc,
        num_integration_steps=2,
        step_size=0.1,
    )

    for _ in range(5):
        key, subkey = jax.random.split(key)
        state, info = kernel.step(subkey, state, mcmc_parameters=params)

    assert state.particles.shape == particles.shape
    assert state.tempering_param <= 1.0
    assert state.iteration == 5


# ---------------------------------------------------------------------------
# SG-MCMC tests — minibatch via *args
# ---------------------------------------------------------------------------


def _make_sgmcmc_problem():
    """Simple Bayesian linear regression for SG-MCMC tests."""
    data_size = 100
    batch_size = 10

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (data_size, 2))
    y = X @ jnp.array([1.0, -0.5]) + 0.1 * jax.random.normal(k2, (data_size,))

    def logprior_fn(position):
        return -0.5 * jnp.sum(position**2)

    def loglikelihood_fn(position, minibatch):
        X_batch, y_batch = minibatch
        preds = X_batch @ position
        return -0.5 * jnp.sum((preds - y_batch) ** 2)

    ge = grad_estimator(logprior_fn, loglikelihood_fn, data_size)

    # Pre-generate batches as stacked arrays
    indices = jnp.arange(data_size)
    batch_indices = indices.reshape(-1, batch_size)  # (10, 10)
    X_batches = X[batch_indices]  # (10, 10, 2)
    y_batches = y[batch_indices]  # (10, 10)
    # Wrap in a 1-tuple so scan slices correctly and *args unpacks to one
    # positional argument (the minibatch tuple) for the kernel.
    batches = ((X_batches, y_batches),)

    return ge, batches, data_size


def _get_position(state):
    """Extract position from state. All states now have .position."""
    return state.position


@pytest.mark.parametrize("kernel_fn", [sgld, sghmc, sgnht])
def test_sgmcmc_step_with_args(kernel_fn):
    """SG-MCMC step accepts minibatch via *args (not params)."""
    ge, batches, _ = _make_sgmcmc_problem()

    key = jax.random.PRNGKey(0)
    position = jnp.zeros(2)

    if kernel_fn is sgnht:
        sampler = kernel_fn(ge)
        state = sampler.init(position, rng_key=key)
    elif kernel_fn is sghmc:
        sampler = kernel_fn(ge, num_integration_steps=2)
        state = sampler.init(position)
    else:
        sampler = kernel_fn(ge)
        state = sampler.init(position)

    params = sampler.init_params(state)

    # One step with a single minibatch: unwrap the outer 1-tuple, take first slice
    single_batch = jax.tree_util.tree_map(lambda x: x[0], batches[0])
    new_state, info = sampler.step(key, state, params, single_batch)

    new_pos = _get_position(new_state)
    old_pos = _get_position(state)
    assert new_pos.shape == position.shape
    assert not jnp.allclose(new_pos, old_pos), "Position should change after a step"


@pytest.mark.parametrize("kernel_fn", [sgld, sghmc, sgnht])
def test_sgmcmc_mcmc_runner_run(kernel_fn):
    """MCMC.run works with SG-MCMC kernels when args= is provided."""
    ge, batches, _ = _make_sgmcmc_problem()

    key = jax.random.PRNGKey(1)
    position = jnp.zeros(2)

    if kernel_fn is sgnht:
        sampler = kernel_fn(ge)
        state = sampler.init(position, rng_key=key)
    elif kernel_fn is sghmc:
        sampler = kernel_fn(ge, num_integration_steps=2)
        state = sampler.init(position)
    else:
        sampler = kernel_fn(ge)
        state = sampler.init(position)

    params = sampler.init_params(state)
    num_steps = batches[0][0].shape[0]  # 10

    runner = MCMC(sampler)
    final_state = runner.run(key, state, num_steps, params=params, args=batches)

    final_pos = _get_position(final_state)
    assert final_pos.shape == position.shape
    assert not jnp.allclose(final_pos, position), "Position should change after running"


def test_mcmc_runner_run_without_args_unchanged():
    """MCMC.run still works for standard kernels without args."""
    key = jax.random.PRNGKey(0)
    position = jnp.array([0.1, -0.2])

    def logdensity(x):
        return -0.5 * jnp.sum(x**2)

    kernel = mala(logdensity)
    state = kernel.init(position)
    params = kernel.init_params(state)

    runner = MCMC(kernel)
    final_state = runner.run(key, state, 10, params=params)

    assert final_state.position.shape == position.shape


@pytest.mark.parametrize("kernel_fn", [sgld, sgnht])
def test_sgmcmc_mcmc_runner_sample(kernel_fn):
    """MCMC.sample works with SG-MCMC kernels when args= is provided."""
    ge, batches, _ = _make_sgmcmc_problem()

    key = jax.random.PRNGKey(2)
    position = jnp.zeros(2)

    if kernel_fn is sgnht:
        sampler = kernel_fn(ge)
        state = sampler.init(position, rng_key=key)
    else:
        sampler = kernel_fn(ge)
        state = sampler.init(position)
    params = sampler.init_params(state)

    num_samples = 5
    thin = 2
    # Need num_samples * thin = 10 batches — slice the inner arrays
    total_batches = jax.tree_util.tree_map(lambda x: x[: num_samples * thin], batches)

    runner = MCMC(sampler)
    samples, final_state = runner.sample(
        key, state, num_samples, params=params, thin=thin, args=total_batches
    )

    assert samples.shape == (num_samples, 2)
    assert final_state.position.shape == position.shape


# ---------------------------------------------------------------------------
# Pseudo-marginal MCMC tests
# ---------------------------------------------------------------------------


def _stochastic_logdensity(position, rng_key):
    """True Gaussian log-density + small additive noise."""
    true_logd = -0.5 * jnp.sum(position**2)
    noise = 0.1 * jax.random.normal(rng_key)
    return true_logd + noise


@pytest.mark.parametrize("inner_cls", [gauss_rwmh, mala])
def test_pseudo_marginal_step(inner_cls):
    """Pseudo-marginal step produces a new state with correct shape."""
    key = jax.random.PRNGKey(0)
    position = jnp.zeros(2)

    kernel = pseudo_marginal(inner_cls, _stochastic_logdensity)
    state = kernel.init(position, rng_key=key)
    params = kernel.init_params(state)

    new_state, info = kernel.step(key, state, params)

    assert new_state.position.shape == position.shape
    # logdensity is stored in state (noisy estimate)
    assert jnp.isfinite(new_state.logdensity)


def test_pseudo_marginal_hmc():
    """Pseudo-marginal with HMC (gradient-based, multiple logdensity evals)."""
    key = jax.random.PRNGKey(1)
    position = jnp.zeros(3)

    kernel = pseudo_marginal(hmc, _stochastic_logdensity, num_integration_steps=5)
    state = kernel.init(position, rng_key=key)
    params = kernel.init_params(state)

    new_state, info = kernel.step(key, state, params)

    assert new_state.position.shape == position.shape
    assert jnp.isfinite(new_state.logdensity)


def test_pseudo_marginal_num_samples():
    """num_samples > 1 averages multiple stochastic estimates."""
    key = jax.random.PRNGKey(2)
    position = jnp.array([0.5, -0.3])

    kernel_1 = pseudo_marginal(gauss_rwmh, _stochastic_logdensity, num_samples=1)
    kernel_10 = pseudo_marginal(gauss_rwmh, _stochastic_logdensity, num_samples=10)

    state_1 = kernel_1.init(position, rng_key=key)
    state_10 = kernel_10.init(position, rng_key=key)

    # With num_samples=10, the log-density estimate should be closer to truth
    true_logd = -0.5 * jnp.sum(position**2)
    # We can't assert exact values due to randomness, but both should be finite
    assert jnp.isfinite(state_1.logdensity)
    assert jnp.isfinite(state_10.logdensity)

    # Verify both can step
    params = kernel_10.init_params(state_10)
    new_state, _ = kernel_10.step(key, state_10, params)
    assert new_state.position.shape == position.shape


def test_pseudo_marginal_mcmc_runner():
    """Pseudo-marginal works with the MCMC runner (jax.lax.scan)."""
    key = jax.random.PRNGKey(3)
    position = jnp.zeros(2)

    kernel = pseudo_marginal(gauss_rwmh, _stochastic_logdensity)
    state = kernel.init(position, rng_key=key)
    params = kernel.init_params(state)

    runner = MCMC(kernel)
    final_state = runner.run(key, state, 50, params=params)

    assert final_state.position.shape == position.shape
    assert not jnp.allclose(final_state.position, position), (
        "Chain should move from the origin"
    )


def test_pseudo_marginal_mcmc_runner_hmc():
    """Pseudo-marginal HMC works with the MCMC runner."""
    key = jax.random.PRNGKey(4)
    position = jnp.zeros(2)

    kernel = pseudo_marginal(hmc, _stochastic_logdensity, num_integration_steps=3)
    state = kernel.init(position, rng_key=key)
    params = kernel.init_params(state)

    runner = MCMC(kernel)
    final_state = runner.run(key, state, 20, params=params)

    assert final_state.position.shape == position.shape
    assert jnp.isfinite(final_state.logdensity)


# ---------------------------------------------------------------------------
# MCMC runner configuration tests
# ---------------------------------------------------------------------------


def test_mcmc_runner_custom_tracked_stats():
    """tracked_stats can be customised at construction."""
    key = jax.random.PRNGKey(0)
    position = jnp.array([0.5, -0.3])

    def logdensity(x):
        return -0.5 * jnp.sum(x**2)

    kernel = hmc(logdensity, num_integration_steps=3)
    state = kernel.init(position)
    params = kernel.init_params(state)

    # Only track logdensity (skip acceptance_rate)
    runner = MCMC(kernel, tracked_stats=("logdensity",))
    assert runner.tracked_stats == ("logdensity",)
    final = runner.run(key, state, 10, params=params)
    assert final.position.shape == position.shape


def test_mcmc_runner_logdensity_tracked():
    """MCMC runner extracts logdensity from state in the scan output."""
    key = jax.random.PRNGKey(0)
    position = jnp.zeros(2)

    def logdensity(x):
        return -0.5 * jnp.sum(x**2)

    kernel = mala(logdensity)
    state = kernel.init(position)
    params = kernel.init_params(state)

    runner = MCMC(kernel)
    stats = runner._extract_stats(state, None)
    # logdensity should be extracted from state, acceptance_rate is NaN (no info)
    assert jnp.isfinite(stats[0])  # logdensity
    assert jnp.isnan(stats[1])  # acceptance_rate (no info object)
