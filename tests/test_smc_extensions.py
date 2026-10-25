"""Temporal SMC extensions: default moves, joint path sampling, diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import norm

from probjax.inference.filtering.joint import sample_joint_paths
from probjax.inference.smc.temporal import run_temporal_smc, temporal_smc
from probjax.inference.smc.temporal_utils import (
    population_diagnostics,
    summarize_replicates,
)
from tests.test_temporal_smc import (
    MODELS,
    THETA,
    TS,
    YS,
    exact_joint,
    gaussian_model,
    kalman_log_likelihood,
)

KEY = jax.random.key(31)


@pytest.mark.parametrize('factory,num_particles', MODELS)
def test_default_random_walk_recovers_reference_posterior(factory, num_particles):
    grid = jnp.linspace(-5.0, 3.0, 801)
    log_post = norm.logpdf(grid) + jax.vmap(kalman_log_likelihood)(grid)
    weights = jax.nn.softmax(log_post)
    mean = jnp.sum(weights * grid)
    sd = jnp.sqrt(jnp.sum(weights * (grid - mean) ** 2))
    kernel = temporal_smc(
        factory(),
        norm.logpdf,
        num_particles=num_particles,
        ess_threshold=1.0,
        num_moves=3,
        batch_size=64,
    )
    init_key, run_key = jax.random.split(KEY)
    state = kernel.init(init_key, jax.random.normal(init_key, (512,)))
    result = jax.jit(
        lambda key: run_temporal_smc(kernel, key, state, TS, YS, history='full')
    )(run_key)
    assert result.trace.infos.valid.all()
    assert (result.trace.infos.acceptance_rate > 0).any()
    w = jnp.exp(result.state.log_weights)
    smc_mean = jnp.sum(w * result.state.parameters)
    smc_sd = jnp.sqrt(jnp.sum(w * (result.state.parameters - smc_mean) ** 2))
    np.testing.assert_allclose(smc_mean, mean, atol=0.1)
    np.testing.assert_allclose(smc_sd, sd, atol=0.1)
    assert result.state.proposal_scale > 0.0


def test_joint_sampler_matches_gaussian_conditional_moments():
    model = gaussian_model()
    kernel = temporal_smc(model, norm.logpdf)
    state = kernel.init(KEY, jnp.full((4,), THETA))
    state = run_temporal_smc(kernel, KEY, state, TS, YS).state
    sample = sample_joint_paths(
        KEY, state, model, TS, YS, num_samples=4000, batch_size=64
    )
    mean, cov, _ = exact_joint()
    np.testing.assert_allclose(sample.paths[:, :, 0].mean(0), mean, atol=0.025)
    np.testing.assert_allclose(np.cov(sample.paths[:, :, 0].T), cov, atol=0.02)


def test_diagnostics_distinguish_uniform_duplicates():
    d = population_diagnostics(
        jnp.array([[1.0], [1.0], [2.0], [2.0]]),
        jnp.full(4, -jnp.log(4)),
        jnp.array([0, 0, 0, 0]),
    )
    assert d.ess == 4 and d.unique_particles == 2 and d.unique_ancestors == 1
    stats = summarize_replicates(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(stats.standard_error, 1 / jnp.sqrt(3), atol=1e-6)
