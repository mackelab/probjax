"""Parity tests for SMC numerics vendored from unreleased BlackJAX main.

Covers the ESS offset stabilization (``6482e5e4`` + ``e06d8ec8``) and the
systematic-resampling boundary fixes (``50818d9c`` + ``6f0edc21``).
"""

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.smc.resampling import systematic as bj_systematic

from probjax.inference.smc.ess import ess, log_ess
from probjax.inference.smc.resampling import resample_systematic, systematic


def test_ess_offset_invariance():
    log_weights = jnp.array([-0.5, -1.0, -0.2, -2.0])
    assert float(ess(log_weights)) > 0
    np.testing.assert_allclose(
        ess(log_weights), ess(log_weights - 1000.0), rtol=1e-5
    )
    np.testing.assert_allclose(
        log_ess(log_weights), log_ess(log_weights - 1000.0), rtol=1e-5
    )


def test_ess_matches_softmax_reference():
    log_weights = jnp.array([-0.3, -1.2, -0.7, -0.1])
    weights = np.exp(np.asarray(log_weights) - np.max(log_weights))
    weights /= weights.sum()
    np.testing.assert_allclose(ess(log_weights), 1.0 / np.sum(weights**2), rtol=1e-6)


def test_ess_gradient_finite():
    log_weights = jnp.array([-100.0, -101.0, -99.5, -100.5])
    grads = jax.grad(lambda w: ess(w))(log_weights)
    assert jnp.all(jnp.isfinite(grads))


def test_systematic_matches_blackjax_off_boundary():
    key = jax.random.key(0)
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(
        np.asarray(systematic(key, weights, 4)),
        np.asarray(bj_systematic(key, weights, 4)),
    )


def test_systematic_never_selects_zero_weight():
    weights = jnp.array([0.0, 0.0, 0.5, 0.5, 0.0])
    for seed in range(20):
        idx = systematic(jax.random.key(seed), weights, 10)
        assert bool(jnp.all(weights[idx] > 0)), f"seed {seed}: {idx}"


def test_systematic_uniform_endpoints():
    # Reachable float32 uniform endpoints must not collapse onto zero-weight
    # particles either.
    weights = jnp.array([0.0, 0.25, 0.25, 0.25, 0.25])
    for seed, u in enumerate((0.0, np.nextafter(1.0, 0.0))):
        idx = systematic(jax.random.key(100 + seed), weights, weights.shape[0])
        assert bool(jnp.all(weights[idx] > 0)), f"u={u}: {idx}"


def test_resample_systematic_api():
    key = jax.random.key(0)
    particles = jnp.arange(4.0)
    log_weights = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4]))
    new_particles, new_log_weights, idx = resample_systematic(
        key, log_weights, particles
    )
    assert new_particles.shape == (4,)
    np.testing.assert_allclose(new_particles, particles[idx])
    np.testing.assert_allclose(
        new_log_weights, jnp.zeros(4) - jnp.log(4.0), rtol=1e-6
    )
