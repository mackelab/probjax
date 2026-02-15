import jax
import jax.numpy as jnp
import numpy as np

from probjax.inference.mcmc import hmc
from probjax.inference.smc import smc
from probjax.inference.smc.path import GeometricPath, PartialPosteriorsPath
from probjax.inference.smc_runner import SMC


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


# =============================================================================
# SMC Runner tests
# =============================================================================


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
