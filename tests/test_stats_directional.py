import jax
import jax.numpy as jnp

from probjax.stats.continuous import bingham, watson


def _unit_vector(x):
    return x / jnp.linalg.norm(x)


def test_watson_samples_live_on_sphere():
    key = jax.random.PRNGKey(0)
    mean_direction = _unit_vector(jnp.array([0.3, -0.4, 1.2]))
    samples = watson.rvs(key, mean_direction=mean_direction, kappa=2.5, shape=(128,))
    norms = jnp.linalg.norm(samples, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-6)


def test_watson_logpdf_symmetry_and_uniform_limit():
    mean_direction = _unit_vector(jnp.array([0.0, 0.0, 1.0]))
    x = _unit_vector(jnp.array([1.0, 0.5, -0.25]))
    logpdf_pos = watson.logpdf(x, mean_direction, kappa=3.0)
    logpdf_neg = watson.logpdf(-x, mean_direction, kappa=3.0)
    assert jnp.allclose(logpdf_pos, logpdf_neg, atol=1e-6)

    logpdf_uniform = watson.logpdf(x, mean_direction, kappa=0.0)
    logpdf_uniform_ref = watson.logpdf(
        _unit_vector(jnp.array([0.1, -0.3, 0.95])), mean_direction, kappa=0.0
    )
    assert jnp.allclose(logpdf_uniform, logpdf_uniform_ref, atol=1e-6)


def test_watson_natural_parameter_shape():
    mean_direction = _unit_vector(jnp.array([0.1, 0.3, 0.9]))
    kappa = jnp.array([2.0, 5.0])  # batched concentration
    params = watson.natural_parameters(
        mean_direction=jnp.broadcast_to(mean_direction, (2, 3)),
        kappa=kappa,
    )
    assert params.shape == (2, 3)


def test_bingham_samples_live_on_sphere():
    key = jax.random.PRNGKey(1)
    orientation = jnp.eye(3)
    concentration = jnp.array([-3.0, 0.5, 2.0])
    samples = bingham.rvs(
        key, orientation=orientation, concentration=concentration, shape=(64,)
    )
    norms = jnp.linalg.norm(samples, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-6)


def test_bingham_logpdf_symmetry_and_uniform_limit():
    orientation = jnp.eye(3)
    concentration = jnp.array([-2.0, 0.0, 1.0])
    x = _unit_vector(jnp.array([0.2, -0.7, 0.65]))
    logpdf_pos = bingham.logpdf(x, orientation, concentration)
    logpdf_neg = bingham.logpdf(-x, orientation, concentration)
    assert jnp.allclose(logpdf_pos, logpdf_neg, atol=1e-6)

    logpdf_uniform = bingham.logpdf(x, orientation, jnp.zeros_like(concentration))
    logpdf_uniform_ref = bingham.logpdf(
        _unit_vector(jnp.array([-0.5, 0.3, 0.81])),
        orientation,
        jnp.zeros_like(concentration),
    )
    assert jnp.allclose(logpdf_uniform, logpdf_uniform_ref, atol=1e-6)


def test_bingham_natural_parameter_shape():
    orientation = jnp.stack([jnp.eye(3), jnp.eye(3)], axis=0)
    concentration = jnp.stack(
        [jnp.array([-2.0, 0.0, 1.0]), jnp.array([0.5, -0.3, -0.2])],
        axis=0,
    )
    params = bingham.natural_parameters(
        orientation=orientation, concentration=concentration
    )
    assert params.shape == (2, 3, 3)
    assert jnp.allclose(params, jnp.swapaxes(params, -1, -2))


def test_watson_fit_estimates_direction():
    key = jax.random.PRNGKey(2)
    true_mu = _unit_vector(jnp.array([0.1, -0.4, 1.0]))
    samples = watson.rvs(key, mean_direction=true_mu, kappa=5.0, shape=(512,))
    mu_hat, kappa_hat = watson.fit(samples)
    alignment = jnp.abs(jnp.dot(mu_hat, true_mu))
    assert alignment > 0.95
    assert kappa_hat > 0


def test_bingham_fit_estimates_orientation():
    key = jax.random.PRNGKey(3)
    orientation = jnp.eye(3)
    concentration = jnp.array([-3.0, -1.0, 0.0])
    samples = bingham.rvs(
        key, orientation=orientation, concentration=concentration, shape=(512,)
    )
    orientation_hat, concentration_hat = bingham.fit(samples)
    overlap = jnp.abs(orientation_hat.T @ orientation)
    assert jnp.all(jnp.max(overlap, axis=1) > 0.8)
    assert concentration_hat.shape == concentration.shape
