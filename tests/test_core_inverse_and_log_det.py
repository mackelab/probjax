from probjax.core import inverse_and_logabsdet, inverse
from typing import Callable, Sequence

import pytest
import jax
import jax.numpy as jnp

funs = [jnp.log, jnp.exp, lambda x: x + 1, lambda x: x - 1, lambda x: x ** 3, lambda x: x* 2, lambda x: x/2]

@pytest.mark.parametrize("fun", funs)
def test_inverse(fun: Callable):
    x = jnp.linspace(0, 10, 100)
    y = jax.vmap(fun)(x)

    inv_fun = inverse(fun)
    inv_y = inv_fun(y)

    assert x.shape == inv_y.shape, "Inverse function shape is not correct."
    assert jnp.allclose(x, inv_y), "Inverse function value is not correct."


@pytest.mark.parametrize("fun", funs)
def test_inverse_and_logabsdet(fun: Callable):
    x = jnp.linspace(0, 10, 100)
    y = jax.vmap(fun)(x)

    inv_and_det_fn = inverse_and_logabsdet(fun)
    inv_y, logabsdet = inv_and_det_fn(y)

    assert x.shape == inv_y.shape, "Inverse function shape is not correct."
    assert jnp.allclose(x, inv_y), "Inverse function value is not correct."

    logabsdet_true = -jnp.log(jax.vmap(jax.grad(fun))(inv_y))
    assert logabsdet.shape == logabsdet_true.shape, "Logabsdet shape is not correct."
    assert jnp.allclose(logabsdet, logabsdet_true), "Logabsdet is not correct."
