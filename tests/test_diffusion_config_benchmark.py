"""Validate diffusion benchmark oracles and named-config coverage."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from docs.scripts.benchmark_diffusion_configs import (
    CLASS_PROBS,
    GAUSSIAN_VARIANCES,
    MIXTURE_CENTERS,
    ROTATION,
    TRAININGS,
    cases,
    categorical_posterior,
    make_model,
    posterior_mean,
    sample_metrics,
    target_sample,
)
from probjax.nn.generative.discrete.config import MultinomialDiffusionSchedule

#: The benchmark suite is long-running; skip it with ``-m 'not slow'``.
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("target", ["gaussian", "mixture"])
def test_oracle_is_identity_at_zero_noise(target):
    values = jax.random.normal(jax.random.key(0), (16, 2))
    alpha = jnp.full((16, 1), 0.7)
    np.testing.assert_allclose(
        posterior_mean(target, alpha, jnp.zeros_like(alpha), values),
        values / alpha,
        atol=1e-6,
    )


def test_gaussian_oracle_matches_dense_bayes_formula():
    covariance = (
        np.asarray(ROTATION)
        @ np.diag(np.asarray(GAUSSIAN_VARIANCES))
        @ np.asarray(ROTATION.T)
    )
    values = np.array([[1.0, -0.3], [0.2, 1.5]], dtype=np.float32)
    alpha, sigma = 0.8, 0.7
    gain = (
        alpha * covariance @ np.linalg.inv(alpha**2 * covariance + sigma**2 * np.eye(2))
    )
    actual = posterior_mean(
        "gaussian", jnp.array(alpha), jnp.array(sigma), jnp.asarray(values)
    )
    np.testing.assert_allclose(actual, values @ gain.T, atol=1e-6)


def test_mixture_oracle_posterior_symmetry():
    predicted = posterior_mean(
        "mixture", jnp.ones((4, 1)), jnp.full((4, 1), 0.1), MIXTURE_CENTERS
    )
    np.testing.assert_allclose(predicted, MIXTURE_CENTERS, atol=1e-5)
    np.testing.assert_allclose(
        posterior_mean("mixture", jnp.array(1.0), jnp.array(0.5), jnp.zeros((8, 2))),
        0,
        atol=1e-6,
    )


def test_learned_panel_covers_every_training_config_and_head():
    matrix = list(cases(SimpleNamespace(phase="learned", seeds=2)))
    assert len(matrix) == 78
    assert {case["training"] for case in matrix} == set(TRAININGS)
    assert {case["head"] for case in matrix} == {"x0", "eps", "v"}
    assert all(
        case["preset"] == "EDM"
        for case in matrix
        if case["training"] == "EDMTrainingConfig"
    )


@pytest.mark.parametrize("kind", ["linear", "cosine", "sigmoid", "logsnr"])
def test_categorical_oracle_marginalizes_to_clean_prior(kind):
    schedule = getattr(MultinomialDiffusionSchedule, "from_" + kind)(32, 4)
    tokens = jnp.arange(4)[:, None]
    times = jnp.full((4, 1), 0.7)
    posterior = categorical_posterior(schedule, times, tokens)[:, 0, :]
    alpha = schedule.alpha_bar(jnp.array(0.7))
    noisy_prior = alpha * CLASS_PROBS + (1 - alpha) * schedule.base_probs
    np.testing.assert_allclose(noisy_prior @ posterior, CLASS_PROBS, atol=2e-6)


def test_sample_metric_is_zero_for_identical_samples_and_flags_nonfinite():
    values = target_sample("gaussian", jax.random.key(0), 32)
    assert sample_metrics(values, values, 1.0)["swd2"] == 0
    invalid = values.at[0, 0].set(jnp.nan)
    metrics = sample_metrics(invalid, values, 1.0)
    assert metrics["finite_fraction"] < 1
    assert metrics["swd2"] is None


@pytest.mark.parametrize("name", list(TRAININGS))
def test_training_factories_use_preset_time_bounds(name):
    model = make_model("EDM", "mixture", 0, 16, training=name)
    times = model.train_cfg.sample_times(jax.random.key(1), (128,))
    assert jnp.isfinite(times).all()
    assert jnp.all(times >= model.train_cfg.t_min)
    assert jnp.all(times <= model.train_cfg.t_max)
