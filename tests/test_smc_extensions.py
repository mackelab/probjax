"""Temporal SMC extensions: default moves, joint path sampling, diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import norm

from probjax.inference import run, temporal_smc
from probjax.inference.filtering.joint import sample_joint_paths
from probjax.inference.smc.temporal_utils import (
    population_diagnostics,
    summarize_replicates,
)
from tests.test_temporal_smc import (
    FILTERS,
    THETA,
    TS,
    YS,
    exact_joint,
    gaussian_model,
    kalman,
    kalman_log_likelihood,
)

KEY = jax.random.key(31)


@pytest.mark.parametrize('factory', FILTERS)
def test_default_random_walk_recovers_reference_posterior(factory):
    grid = jnp.linspace(-5.0, 3.0, 801)
    log_post = norm.logpdf(grid) + jax.vmap(kalman_log_likelihood)(grid)
    weights = jax.nn.softmax(log_post)
    mean = jnp.sum(weights * grid)
    sd = jnp.sqrt(jnp.sum(weights * (grid - mean) ** 2))
    kernel = temporal_smc(
        factory(),
        norm.logpdf,
        max_observations=len(TS),
        ess_threshold=1.0,
        num_moves=3,
        batch_size=64,
    )

    def one_run(key):
        init_key, run_key = jax.random.split(key)
        state = kernel.init(init_key, jax.random.normal(init_key, (512,)), YS[0])
        result = run(
            run_key, kernel, state, None, len(TS), args=(TS, YS), collect=('info',)
        )
        w = jnp.exp(result.state.log_weights)
        smc_mean = jnp.sum(w * result.state.parameters)
        smc_sd = jnp.sqrt(jnp.sum(w * (result.state.parameters - smc_mean) ** 2))
        return smc_mean, smc_sd, result.trace['info'], result.state.proposal_scale

    # One run's mean scatters by about 0.04; average a few.
    smc_mean, smc_sd, info, scale = jax.jit(jax.vmap(one_run))(jax.random.split(KEY, 8))
    assert info.valid.all()
    assert (info.acceptance_rate > 0).any()
    np.testing.assert_allclose(smc_mean.mean(), mean, atol=0.05)
    np.testing.assert_allclose(smc_sd.mean(), sd, atol=0.05)
    assert (scale > 0.0).all()


def test_joint_sampler_matches_gaussian_conditional_moments():
    kernel = temporal_smc(kalman(), norm.logpdf, max_observations=len(TS))
    state = kernel.init(KEY, jnp.full((4,), THETA), YS[0])
    state = run(KEY, kernel, state, None, len(TS), args=(TS, YS)).state
    sample = sample_joint_paths(
        KEY, state, kalman(), gaussian_model(), num_samples=4000, batch_size=64
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
