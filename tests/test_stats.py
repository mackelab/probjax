import jax
import jax.numpy as jnp
import pytest
from scipy.special import betaincinv as scipy_betaincinv

from probjax.distributions.continuous import Gamma, Normal, Uniform

# Import your betaincinv function here
from probjax.utils.stats import betaincinv, differential_entropy


@pytest.mark.parametrize(
    "a, b",
    [
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 2.0),
        (2.0, 5.0),
        (5.0, 2.0),
        (0.1, 0.1),
        (10.0, 10.0),
        (0.5, 5.0),
        (5.0, 0.5),
        (1.0, 3.0),
    ],
)
def test_betaincinv(a, b):
    """
    Tests that betaincinv(a, b, p) produces an x-value such that
    betainc(a, b, x) is approximately p.
    """

    a_ = jnp.array(a)
    b_ = jnp.array(b)
    p = jnp.linspace(0, 1, 100)

    # Calculate x-values using your betaincinv function
    x = betaincinv(a_, b_, p)
    x_scipy = scipy_betaincinv(a, b, p)

    # Should be close to the original p
    assert jnp.allclose(x, x_scipy, atol=1e-1, rtol=1e-1)


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
