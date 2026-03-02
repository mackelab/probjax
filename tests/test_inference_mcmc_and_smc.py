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
    slice,
)
from probjax.inference.smc import smc
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
