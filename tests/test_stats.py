import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import betainc, digamma, gammainc
from scipy.special import betaincinv as scipy_betaincinv
from scipy.special import gammaincinv as scipy_gammaincinv

from probjax.stats.continuous import gamma, norm, uniform

# Import your betaincinv function here
from probjax.utils.special.betaincinv import betaincinv
from probjax.utils.special.digammainv import digammainv
from probjax.utils.special.gammaincinv import gammaincinv
from probjax.utils.stats import differential_entropy, mle_dirichlet


@pytest.mark.parametrize(
    "a, b",
    list(
        zip(
            np.random.uniform(0.0001, 10.0, size=(100,)),
            np.random.uniform(0.0001, 10.0, size=(100,)), strict=False,
        )
    ),
)
def test_betaincinv(a, b):
    """
    Tests that betaincinv(a, b, p) produces an x-value such that
    betainc(a, b, x) is approximately p.
    """

    a_ = jnp.array(a)
    b_ = jnp.array(b)
    x = jnp.linspace(0.005, 0.995, 1000)
    p = betainc(a_, b_, x)
    # Calculate x-values using your betaincinv function
    x = betaincinv(a_, b_, p)
    x_scipy = scipy_betaincinv(a, b, p)

    # Should be close to the original p
    assert jnp.allclose(x, x_scipy, atol=1e-4, rtol=1e-4), (
        "Avg absolute error: {}".format(jnp.mean(jnp.abs(x - x_scipy)))
    )


@pytest.mark.parametrize("a", np.random.uniform(0.001, 20.0, size=(100,)))
def test_gammaincinv(a):
    """
    Tests that gammaincinv(a, p) produces an x-value such that
    gammainc(a, x) is approximately p.
    """

    a_ = jnp.array(a)
    x = jnp.linspace(0, a + 3 * jnp.sqrt(a), 1000)
    p = gammainc(a_, x)

    # Calculate x-values using your gammaincinv function
    x = gammaincinv(a_, p)
    x_scipy = scipy_gammaincinv(a, p)

    # Should be close to the original p
    assert jnp.allclose(x, x_scipy, atol=1e-3)


@pytest.mark.parametrize("dist", [norm(0, 1), uniform(0, 1), norm(2, 3), gamma(2, 3)])
@pytest.mark.parametrize("num_samples", [1001, 5000, 10000])
def test_differential_entropy(dist, num_samples):
    """
    Tests that the differential entropy of the beta distribution is
    calculated correctly.
    """

    # NOTE only tests vasicek, the =1000 fails
    p = dist
    key = jax.random.PRNGKey(0)
    samples = p.rvs(key, shape=(num_samples,))

    # Calculate the differential entropy
    h = differential_entropy(samples)
    h_true = p.entropy()

    print(h, h_true)

    # Should be close to the true entropy
    assert jnp.allclose(h, h_true, atol=1e-1, rtol=2e-1)


def test_digammainv():
    x = jnp.linspace(0.0, 100.0, 1000)
    y = digamma(x)
    x_recovered = jax.vmap(digammainv)(y)

    assert jnp.allclose(x, x_recovered, atol=1e-3), "Avg absolute error: {}".format(
        jnp.mean(jnp.abs(x - x_recovered))
    )


@pytest.mark.parametrize(
    "alpha",
    [jnp.ones(4), jnp.ones(4) * 0.1, np.random.uniform(0.0001, 10.0, size=(4,))],
)
def test_mle_dirichlet(alpha):
    xs = jax.random.dirichlet(jax.random.key(0), alpha, (100000,))
    alpha_mle = mle_dirichlet(xs)
    assert jnp.allclose(alpha, alpha_mle, atol=1e-2, rtol=1e-2), (
        "Avg absolute error: {}".format(jnp.mean(jnp.abs(alpha - alpha_mle)))
    )
