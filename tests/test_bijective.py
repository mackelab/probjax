import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.core import inverse, inverse_and_logabsdet
from probjax.nn.density_estimator.normalizing_flows import (
    monotone_hermite_cubic_spline,
    piecewise_affine_spline,
    rational_linear_spline,
    rational_quadratic_spline,
)
from probjax.stats.bijective import additive_bijector, affine_bijector


@pytest.mark.parametrize("seed", np.random.randint(0, 1000, 2))
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("num_bins", [4, 8, 16])
def test_rational_quadratic_spline(seed, scale, num_bins):
    rng = jax.random.PRNGKey(seed)
    rng1, rng2 = jax.random.split(rng)
    x = jax.random.normal(rng1)
    params = jax.random.normal(rng2, (3 * num_bins)) * scale

    y = rational_quadratic_spline(params, x)
    inv_and_logdet = inverse_and_logabsdet(rational_quadratic_spline, invertible_arg=1)
    x_rec, logdet = inv_and_logdet(params, y)

    assert y.shape == x.shape
    assert jnp.allclose(jnp.abs(x - x_rec).mean(), 0.0, atol=1e-2), (
        "Spline inverse error is too large, should be 0 but is {}".format(
            jnp.abs(x - x_rec).mean()
        )
    )
    assert jnp.isfinite(logdet).all()
    assert jnp.isfinite(y).all()
    assert jnp.isfinite(x_rec).all()


@pytest.mark.parametrize("seed", np.random.randint(0, 1000, 2))
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("num_bins", [4, 16, 64])
def test_linear_spline(seed, scale, num_bins):
    rng = jax.random.PRNGKey(seed)
    rng1, rng2 = jax.random.split(rng)
    x = jax.random.normal(rng1)
    params = jax.random.normal(rng2, (2 * num_bins)) * scale

    y = piecewise_affine_spline(params, x)
    inv_and_logdet = inverse_and_logabsdet(piecewise_affine_spline, invertible_arg=1)
    x_rec, logdet = inv_and_logdet(params, y)

    assert y.shape == x.shape
    assert jnp.allclose(x, x_rec, atol=1e-2)
    assert jnp.isfinite(logdet).all()


@pytest.mark.parametrize("seed", np.random.randint(0, 1000, 2))
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("num_bins", [4, 16, 64])
def test_rational_linear_spline(seed, scale, num_bins):
    rng = jax.random.PRNGKey(seed)
    rng1, rng2 = jax.random.split(rng)
    x = jax.random.normal(rng1)
    params = jax.random.normal(rng2, (3 * num_bins)) * scale

    y = rational_linear_spline(params, x)
    inv_and_logdet = inverse_and_logabsdet(rational_linear_spline, invertible_arg=1)
    x_rec, logdet = inv_and_logdet(params, y)

    assert y.shape == x.shape
    assert jnp.allclose(x, x_rec, atol=1e-2)
    assert jnp.isfinite(logdet).all()


@pytest.mark.parametrize("seed", np.random.randint(0, 1000, 2))
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("num_bins", [4, 16, 64])
def test_monotone_hermite_cubic_spline(seed, scale, num_bins):
    rng = jax.random.PRNGKey(seed)
    rng1, rng2 = jax.random.split(rng)
    x = jax.random.normal(rng1)
    params = jax.random.normal(rng2, (3 * num_bins)) * scale

    y = monotone_hermite_cubic_spline(params, x)
    inv_and_logdet = inverse_and_logabsdet(
        monotone_hermite_cubic_spline, invertible_arg=1
    )
    x_rec, logdet = inv_and_logdet(params, y)

    assert y.shape == x.shape
    assert jnp.allclose(x, x_rec, atol=1e-2)
    assert jnp.isfinite(logdet).all()


def test_affine_bijector():
    rng = jax.random.PRNGKey(0)
    params = jax.random.normal(rng, (10, 20))
    x = jax.random.normal(rng, (10, 10))

    y = affine_bijector(params, x)
    x_rec = inverse(affine_bijector, invertible_arg=1)(params, y)
    assert y.shape == x.shape
    assert jnp.allclose(x, x_rec, atol=1e-2)


def test_additive_bijector():
    rng = jax.random.PRNGKey(0)
    params = jax.random.normal(rng, (10, 10))
    x = jax.random.normal(rng, (10, 10))

    y = additive_bijector(params, x)

    x_rec = inverse(additive_bijector, invertible_arg=1)(params, y)
    assert y.shape == x.shape
    assert jnp.allclose(x, x_rec, atol=1e-2)
