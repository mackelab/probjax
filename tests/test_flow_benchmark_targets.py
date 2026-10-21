"""Check benchmark reference densities against independent SciPy formulas."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

from docs.scripts.benchmark_flow_mixing import (
    REALISTIC_TARGETS,
    REFERENCE_LOGPDF,
    TARGETS,
    beta_parameters,
    mixture_parameters,
    rotation8,
)

#: The benchmark suite is long-running; skip it with ``-m 'not slow'``.
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", REALISTIC_TARGETS)
def test_reference_densities_match_scipy(name):
    data = np.asarray(TARGETS[name](jax.random.key(0), 32))
    rotation = np.asarray(rotation8())
    if name == "correlated_gaussian8":
        scales = np.geomspace(0.2, 2.0, 8)
        covariance = rotation.T @ np.diag(scales**2) @ rotation
        expected = stats.multivariate_normal.logpdf(data, cov=covariance)
    elif name == "gaussian_mixture8":
        weights, means, scales = (np.asarray(value) for value in mixture_parameters())
        components = [
            stats.multivariate_normal.logpdf(
                data,
                mean=mean @ rotation,
                cov=rotation.T @ np.diag(scale**2) @ rotation,
            )
            + np.log(weight)
            for weight, mean, scale in zip(weights, means, scales, strict=True)
        ]
        expected = special.logsumexp(components, axis=0)
    elif name == "student_t8":
        scales = np.linspace(0.5, 2.0, 8)
        expected = stats.multivariate_t.logpdf(
            data,
            shape=rotation.T @ np.diag(scales**2) @ rotation,
            df=4,
        )
    elif name == "correlated_lognormal8":
        scales = 0.55 * np.geomspace(0.2, 2.0, 8)
        expected = stats.multivariate_normal.logpdf(
            np.log(data),
            mean=np.linspace(-0.4, 0.4, 8),
            cov=rotation.T @ np.diag(scales**2) @ rotation,
        ) - np.log(data).sum(-1)
    elif name == "bounded_beta8":
        alpha, beta = (np.asarray(value) for value in beta_parameters())
        expected = stats.beta.logpdf(data, alpha, beta).sum(-1)
    else:
        expected = stats.norm.logpdf(data[:, 0], scale=1.5) + stats.norm.logpdf(
            data[:, 1:],
            scale=np.exp(data[:, :1] / 2),
        ).sum(-1)
    assert data.shape == (32, 8)
    assert np.isfinite(data).all()
    actual = REFERENCE_LOGPDF[name](jnp.asarray(data))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
