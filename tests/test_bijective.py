import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.core import inverse
from probjax.nn.bijective import (
    additive_bijector,
    affine_bijector,
    linear_spline,
    rational_quadratic_spline,
)


@pytest.mark.parametrize("seed", np.random.randint(0, 1000, 2))
@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("num_bins", [4, 16, 64])
def test_rational_quadratic_spline(seed, scale, num_bins):
    rng = jax.random.PRNGKey(seed)
    rng1, rng2 = jax.random.split(rng)
    x = jax.random.normal(rng1)
    params = jax.random.normal(rng2, (3 * num_bins)) * scale

    y = rational_quadratic_spline(params, x)
    x_rec, logdet = rational_quadratic_spline.inv_and_logdet(params, y)

    assert y.shape == x.shape
    assert jnp.allclose(x, x_rec, atol=1e-1, rtol=0.5)
    assert jnp.isfinite(logdet).all()
    assert jnp.isfinite(y).all()
    assert jnp.isfinite(x_rec).all()


@pytest.mark.xfail(reason="Bug in the implementation")
@pytest.mark.parametrize("seed", np.random.randint(0, 1000, 2))
@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("num_bins", [4, 16, 64, 128])
def test_linear_spline(seed, scale, num_bins):
    rng = jax.random.PRNGKey(seed)
    rng1, rng2 = jax.random.split(rng)
    x = jax.random.normal(rng1)
    params = jax.random.normal(rng2, (2 * num_bins)) * scale

    y = linear_spline(params, x)
    x_rec, logdet = linear_spline.inv_and_logdet(params, y)

    assert y.shape == x.shape
    assert jnp.allclose(x, x_rec, atol=1e-2)
    assert jnp.allclose(logdet, 0.0, atol=1e-2)


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
    # x_rec, logdet = inverse_and_logabsdet(additive_bijector, invertible_arg=1)(
    #     params, y
    # )
    assert y.shape == x.shape
    assert jnp.allclose(x, x_rec, atol=1e-2)
    # assert jnp.allclose(logdet, 0.0, atol=1e-2)
