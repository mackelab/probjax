import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.core import inverse, inverse_and_logabsdet

from probjax.nn.bijective import (
    rational_quadratic_spline,
    inv_rational_quadratic_spline,
    piecwise_linear_spline,
    _pieceswise_linear_spline_inv,
    learnable_mixture_cdf,
    affine_bijector,
    additive_bijector,
)


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
