import jax
import jax.numpy as jnp
import pytest

from probjax.core import inverse, inverse_and_logabsdet
from probjax.utils.odeint import odeint


def test_inverse_1d(invertible_function_1d):
    x = jnp.linspace(0, 1, 100)
    y = invertible_function_1d(x)

    mask = jnp.isfinite(y)

    inv_fun = inverse(invertible_function_1d)
    inv_y = inv_fun(y)

    assert x.shape == inv_y.shape, "Inverse function shape is not correct."
    assert jnp.allclose(x[mask], inv_y[mask], atol=1e-3, rtol=1e-3), (
        "Inverse function value is not correct."
    )


def test_inverse_and_logabsdet_1d(invertible_function_1d):
    x = jnp.linspace(0, 10, 100).reshape(-1, 1)
    y = jax.vmap(invertible_function_1d)(x)
    mask = jnp.isfinite(y).all(-1)

    inv_and_det_fn = inverse_and_logabsdet(invertible_function_1d)
    inv_y, logabsdet = jax.vmap(inv_and_det_fn)(y)

    print(x.shape, inv_y.shape, logabsdet.shape)

    assert x.shape == inv_y.shape, "Inverse function shape is not correct."
    assert jnp.allclose(x[mask], inv_y[mask], atol=1e-3, rtol=1e-3), (
        "Inverse function value is not correct."
    )

    J = jax.grad(lambda x: invertible_function_1d(x).sum())(x)
    logabsdet_true = -jnp.log(jnp.abs(J)).sum(-1)

    assert jnp.allclose(logabsdet[mask], logabsdet_true[mask], atol=1e-3, rtol=1e-3), (
        "Logabsdet is not correct."
    )


def test_inverse_split():
    x = jnp.ones((10, 2))

    def f(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x2, x1], axis=-1)

    inv_f = inverse(f)
    inv_x = inv_f(x)

    assert jnp.allclose(x, inv_x, atol=1e-3, rtol=1e-3), (
        "Inverse function value is not correct."
    )


@pytest.mark.parametrize(
    "fun,x",
    [
        (lambda x: jnp.tanh(x), jnp.linspace(-0.5, 0.5, 7)),
        (lambda x: jnp.sinh(x), jnp.linspace(-0.4, 0.4, 7)),
        (lambda x: jnp.exp(x), jnp.linspace(-1.0, 1.0, 7)),
        (lambda x: jnp.sqrt(x), jnp.linspace(0.25, 2.0, 7)),
        (lambda x: jnp.log1p(x), jnp.linspace(0.05, 0.95, 7)),
    ],
    ids=["tanh", "sinh", "exp", "sqrt", "log1p"],
)
def test_inverse_univariate_registry(fun, x):
    inv_fun = inverse(fun)
    y = fun(x)
    x_rec = inv_fun(y)
    assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "fun,x",
    [
        (lambda x: x * 3.0, jnp.linspace(0.1, 1.0, 7)),
        (lambda x: x / 2.5, jnp.linspace(0.5, 2.0, 7)),
        (lambda x: x + 1.25, jnp.linspace(-0.5, 0.5, 7)),
        (lambda x: x - 0.75, jnp.linspace(-1.5, 1.5, 7)),
        (lambda x: jnp.power(x, 3.0), jnp.linspace(0.5, 1.5, 7)),
    ],
    ids=["mul", "div", "add", "sub", "pow"],
)
def test_inverse_bivariate_registry(fun, x):
    inv_fun = inverse(fun)
    y = fun(x)
    x_rec = inv_fun(y)
    assert jnp.allclose(x, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_scatter():
    x0 = jnp.array([1.0, 2.0, 3.0])

    def f(x):
        y = jnp.ones((10, 3))
        y = y.at[-1].set(x * 2.0)
        return y[-1]

    inv_f = inverse(f)
    x_inv = inv_f(f(x0))

    assert jnp.allclose(x0, x_inv, atol=1e-6, rtol=1e-6), (
        "Inverse function failed for scatter updates."
    )


def test_inverse_gather_permutation():
    indices = jnp.array([2, 0, 1], dtype=jnp.int32)

    def f(x):
        return x[indices]

    x0 = jnp.array([0.3, -1.2, 2.5])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_squeeze_broadcast():
    def f(x):
        y = jnp.expand_dims(x, axis=0)
        y = jnp.broadcast_to(y, (1,) + x.shape)
        return jnp.squeeze(y, axis=0)

    x0 = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec)


def test_inverse_rev_reshape():
    def f(x):
        y = jnp.reshape(x, (2, 2))
        y = jnp.flip(y, axis=0)
        return jnp.reshape(y, (-1,))

    x0 = jnp.array([1.0, 2.0, 3.0, 4.0])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec)


@pytest.mark.xfail(
    reason="Dynamic slice inverse rule does not fully reconstruct inputs", strict=False
)
def test_inverse_slice_dynamic_slice():
    def f(x):
        head = jax.lax.dynamic_slice(x, (0,), (2,))
        tail = jax.lax.dynamic_slice(x, (2,), (x.shape[0] - 2,))
        prefix = head[:1]
        suffix = jnp.concatenate([head[1:], tail], axis=0)
        return jnp.concatenate([prefix, suffix], axis=0)

    x0 = jnp.array([5.0, 6.0, 7.0, 8.0])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec)


@pytest.mark.xfail(
    reason="convert_element_type inverse rule currently incomplete", strict=False
)
def test_inverse_convert_element_type():
    def f(x):
        return x.astype(jnp.float64)

    x0 = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float32)
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert x_rec.dtype == x0.dtype
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_select_n():
    def f(x):
        cond = jnp.ones_like(x, dtype=bool)
        return jnp.select([cond], [x], default=-x)

    x0 = jnp.array([0.5, 1.2, 3.4])
    inv_f = inverse(f)
    x_rec = inv_f(f(x0))
    assert jnp.allclose(x0, x_rec, atol=1e-6, rtol=1e-6)


def test_inverse_odeint_linear_system():
    ts = jnp.linspace(0.0, 1.0, 1000)

    def drift(t, x):
        return -x

    def forward(x):
        return odeint(drift, x, ts)[-1]

    x0 = jnp.array([1.0, 2.0, 3.0])
    y = forward(x0)

    inv_forward = inverse(forward)
    x_inv = inv_forward(y)

    assert jnp.allclose(x0, x_inv, atol=1e-3, rtol=1e-3), (
        "Inverse function failed for ODE-based transform."
    )
