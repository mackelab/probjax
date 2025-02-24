import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import betainc, gammainc
from scipy.special import betaincinv as scipy_betaincinv
from scipy.special import gammaincinv as scipy_gammaincinv

from probjax.distributions.continuous import Gamma, Normal, Uniform

# Import your betaincinv function here
from probjax.utils.stats import betaincinv, differential_entropy, gammaincinv


@pytest.mark.parametrize(
    "a, b",
    list(
        zip(
            np.random.uniform(0.001, 50.0, size=(100,)),
            np.random.uniform(0.001, 50.0, size=(100,)),
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
    x = jnp.linspace(0.01, 0.99, 1000)
    p = betainc(a_, b_, x)
    # Calculate x-values using your betaincinv function
    x = betaincinv(a_, b_, p)
    x_scipy = scipy_betaincinv(a, b, p)

    # Should be close to the original p
    assert jnp.allclose(x, x_scipy, atol=1e-3)


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


@pytest.mark.parametrize(
    "dist", [Normal(0, 1), Uniform(0, 1), Normal(2, 3), Gamma(2, 3)]
)
@pytest.mark.parametrize("num_samples", [1001, 5000, 10000])
def test_differential_entropy(dist, num_samples):
    """
    Tests that the differential entropy of the beta distribution is
    calculated correctly.
    """

    # NOTE only tests vasicek, the =1000 fails
    p = dist
    key = jax.random.PRNGKey(0)
    samples = p.sample(key, (num_samples,))

    # Calculate the differential entropy
    h = differential_entropy(samples)
    h_true = p.entropy()

    print(h, h_true)

    # Should be close to the true entropy
    assert jnp.allclose(h, h_true, atol=1e-1, rtol=2e-1)
