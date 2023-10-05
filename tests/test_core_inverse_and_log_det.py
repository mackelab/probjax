from probjax.core import inverse_and_logabsdet, inverse
from typing import Callable, Sequence

import pytest
import jax
import jax.numpy as jnp


def test_inverse(invertible_function: Callable):
    x = jnp.linspace(0, 1, 100)
    y = invertible_function(x)

    mask = jnp.isfinite(y)

    inv_fun = inverse(invertible_function)
    inv_y = inv_fun(y)

    assert x.shape == inv_y.shape, "Inverse function shape is not correct."
    assert jnp.allclose(
        x[mask], inv_y[mask], atol=1e-3, rtol=1e-3
    ), "Inverse function value is not correct."


def test_inverse_and_logabsdet(invertible_function: Callable):
    x = jnp.linspace(0, 10, 100)
    y = invertible_function(x)
    mask = jnp.isfinite(y)

    inv_and_det_fn = inverse_and_logabsdet(invertible_function)
    inv_y, logabsdet = inv_and_det_fn(y)

    assert x.shape == inv_y.shape, "Inverse function shape is not correct."
    assert jnp.allclose(
        x[mask], inv_y[mask], atol=1e-3, rtol=1e-3
    ), "Inverse function value is not correct."

    logabsdet_true = -jnp.log(jax.vmap(jax.grad(invertible_function))(inv_y))
    assert logabsdet.shape == logabsdet_true.shape, "Logabsdet shape is not correct."
    assert jnp.allclose(
        logabsdet[mask], logabsdet_true[mask], atol=1e-3, rtol=1e-3
    ), "Logabsdet is not correct."
