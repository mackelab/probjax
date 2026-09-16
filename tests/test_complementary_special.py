"""Accuracy and differentiation checks for both tails and stable log subtraction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from probjax.stats.continuous import beta, gamma
from probjax.utils import (
    betainccinv,
    betaincinv,
    gammainccinv,
    gammaincinv,
    log1mexp,
    logdiffexp,
)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('family', ['beta', 'gamma'])
def test_tail_grid(dtype, upper, family):
    with jax.enable_x64():
        a = np.array([0.1, 0.5, 1, 2, 10, 100, 1000], dtype)[:, None]
        p = np.array([0, 1e-30, 1e-12, 1e-6, 0.01, 0.2, 0.5, 0.9, 1], dtype)[None, :]
        if family == 'beta':
            b = np.array([0.2, 4, 0.7, 10, 100, 0.2, 1000], dtype)[:, None]
            fn = betainccinv if upper else betaincinv
            ref = special.betainccinv if upper else special.betaincinv
            actual, expected = (
                fn(a, b, p),
                ref(a.astype(float), b.astype(float), p.astype(float)),
            )
        else:
            fn = gammainccinv if upper else gammaincinv
            ref = special.gammainccinv if upper else special.gammaincinv
            actual, expected = fn(a, p), ref(a.astype(float), p.astype(float))
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2e-3 if dtype == np.float32 else 2e-10,
            atol=np.finfo(dtype).tiny * 8,
        )
        assert np.all(np.diff(np.asarray(actual), axis=-1) * (-1 if upper else 1) >= 0)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('upper', [False, True])
def test_seeded_stress(dtype, upper):
    with jax.enable_x64():
        rng = np.random.default_rng(42)
        a, b = np.exp(rng.uniform(np.log(0.05), np.log(1000), (2, 2048))).astype(dtype)
        p = special.expit(rng.uniform(-16, 16, 2048)).astype(dtype)
        for fn, ref, args in [
            (
                betainccinv if upper else betaincinv,
                special.betainccinv if upper else special.betaincinv,
                (a, b, p),
            ),
            (
                gammainccinv if upper else gammaincinv,
                special.gammainccinv if upper else special.gammaincinv,
                (a, p),
            ),
        ]:
            actual = np.asarray(fn(*args))
            expected = ref(*(x.astype(float) for x in args))
            mask = expected >= np.finfo(dtype).tiny
            np.testing.assert_allclose(
                actual[mask],
                expected[mask],
                # JAX's FP32 beta CDF loses normalization accuracy for highly
                # unequal shapes (e.g. a=.051, b=784), limiting inverse accuracy.
                rtol=0.03 if dtype == np.float32 else 1e-8,
                atol=0,
            )


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_supplied_tiny(dtype):
    with jax.enable_x64():
        q = np.array([1e-20, 1e-30], dtype)
        assert np.all(1 - q == 1)
        np.testing.assert_allclose(gammainccinv(1.0, q), -np.log(q), rtol=2e-6)
        np.testing.assert_allclose(
            betainccinv(1.0, 100.0, q), -np.expm1(np.log(q) / 100), rtol=2e-6
        )


def test_skewed_beta():
    a, b, p = map(
        np.float32, (0.06992902606725693, 626.0996704101562, 0.6951475739479065)
    )
    for fn, ref in [
        (betaincinv, special.betaincinv),
        (betainccinv, special.betainccinv),
    ]:
        np.testing.assert_allclose(
            fn(a, b, p), ref(float(a), float(b), float(p)), rtol=0.005
        )
        assert np.isfinite(jax.grad(lambda q, fn=fn: fn(a, b, q))(p))


@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('family', ['beta', 'gamma'])
def test_broadcast_gradients(upper, family):
    with jax.enable_x64():
        args = (jnp.array(2.3), jnp.array([0.2, 0.7]))
        if family == 'beta':
            args = (args[0], jnp.array([3.1, 4.2]), args[1])
            fn, ref = (
                (betainccinv, special.betainccinv)
                if upper
                else (betaincinv, special.betaincinv)
            )
        else:
            fn, ref = (
                (gammainccinv, special.gammainccinv)
                if upper
                else (gammaincinv, special.gammaincinv)
            )
        grads = jax.grad(lambda *xs: fn(*xs).sum(), argnums=tuple(range(len(args))))(
            *args
        )
        for i, grad in enumerate(grads):
            expected = np.empty(args[i].shape)
            for index in np.ndindex(args[i].shape):
                plus, minus = [np.array(x) for x in args], [np.array(x) for x in args]
                plus[i][index] += 1e-5
                minus[i][index] -= 1e-5
                expected[index] = (ref(*plus).sum() - ref(*minus).sum()) / 2e-5
            np.testing.assert_allclose(grad, expected, rtol=2e-5, atol=1e-8)


def test_gamma_jvp():
    with jax.enable_x64():
        q = jnp.array(1e-20)
        fn = lambda q: gammainccinv(1.0, q)
        np.testing.assert_allclose(
            jax.jvp(fn, (q,), (jnp.array(1.0),))[1], -1 / q, rtol=1e-12
        )
        np.testing.assert_allclose(jax.grad(jax.grad(fn))(q), 1 / q**2, rtol=1e-12)


def test_endpoints_and_domains():
    for fn, args, expected in [
        (betaincinv, (2.0, 3.0), [0, 1]),
        (betainccinv, (2.0, 3.0), [1, 0]),
        (gammaincinv, (2.0,), [0, np.inf]),
        (gammainccinv, (2.0,), [np.inf, 0]),
    ]:
        np.testing.assert_array_equal(fn(*args, jnp.array([0.0, 1.0])), expected)
        for p in (0.0, 1.0):
            assert (
                jax.grad(lambda a, fn=fn, args=args, p=p: fn(a, *args[1:], p))(2.0) == 0
            )
        assert jnp.isnan(fn(-1.0, *args[1:], 0.5))
        assert jnp.isnan(fn(*args, jnp.nan))
        assert jax.grad(lambda p, fn=fn, args=args: fn(*args, p))(-1.0) == 0


def test_distribution_inverse():
    np.testing.assert_allclose(
        gamma.isf(1e-30, alpha=1.0, beta=2.0), -np.log(1e-30) / 2, rtol=2e-6
    )
    np.testing.assert_allclose(
        beta.isf(1e-30, alpha=1.0, beta=100.0),
        -np.expm1(np.log(1e-30) / 100),
        rtol=2e-6,
    )
    assert gamma.isf(0) == np.inf
    assert gamma.isf(1) == 0
    assert beta.isf(0) == 1
    assert beta.isf(1) == 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_logspace(dtype):
    with jax.enable_x64():
        x = np.array([-1000, -10, -1, -0.1, -1e-10, -1e-30], dtype)
        expected = np.log(-np.expm1(x.astype(float)))
        np.testing.assert_allclose(log1mexp(x), expected, rtol=1e-6, atol=1e-7)
        grad = jax.grad(lambda x: log1mexp(x).sum())(jnp.array(x))
        np.testing.assert_allclose(
            grad, -np.exp(x.astype(float)) / -np.expm1(x.astype(float)), rtol=2e-6
        )
        a, b = jnp.array([1000.0, -1000.0], dtype), jnp.array([999.0, -1001.0], dtype)
        np.testing.assert_allclose(
            logdiffexp(a, b), np.array(a) + np.log1p(-np.exp(-1)), rtol=1e-7
        )
        ga, gb = jax.grad(lambda a, b: logdiffexp(a, b).sum(), argnums=(0, 1))(a, b)
        np.testing.assert_allclose(ga, 1 / -np.expm1(-1), rtol=1e-6)
        np.testing.assert_allclose(gb, -1 / np.expm1(1), rtol=1e-6)
        assert log1mexp(0.0) == -np.inf
        assert log1mexp(-jnp.inf) == 0
        assert jnp.isnan(log1mexp(0.1))
        assert logdiffexp(-jnp.inf, -jnp.inf) == -jnp.inf
        assert logdiffexp(2.0, 2.0) == -jnp.inf
        assert jnp.isnan(logdiffexp(1.0, 2.0))


def test_nested_batching_and_budgets():
    from probjax.utils.special.betaincinv import _make_betaincinv_core

    q = jnp.array([[0.1, 0.3], [0.7, 0.9]])
    for fn, args in [(betainccinv, (2.0, 3.0)), (gammainccinv, (2.0,))]:
        mapped = jax.jit(jax.vmap(jax.vmap(lambda q, fn=fn, args=args: fn(*args, q))))(
            q
        )
        np.testing.assert_allclose(mapped, fn(*args, q), rtol=2e-6)
    assert _make_betaincinv_core(6, 15, True) is _make_betaincinv_core(6, 15, True)
    for budget in (-1, 1.5, True):
        with pytest.raises(ValueError, match='nonnegative integers'):
            betainccinv(2.0, 3.0, 0.5, max_halley_steps=budget)
