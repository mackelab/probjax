import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.core import inverse_and_logabsdet
from probjax.nn import bpf, gf, naf, sospf, unaf
from probjax.nn.generative.nflows import (
    BernsteinBijectorConfig,
    DeepSigmoidBijectorConfig,
    MixtureCDFBijectorConfig,
    SumOfSquaresBijectorConfig,
    UMNNBijectorConfig,
)
from probjax.stats.bijective import (
    inv_bernstein,
    inv_deep_sigmoid,
    inv_mixture_cdf,
    inv_sos_polynomial,
    inv_unconstrained_monotone,
)

# (name, bijector config, analytic data -> base direction)
BIJECTORS = [
    ("dsf", DeepSigmoidBijectorConfig(num_components=3), inv_deep_sigmoid),
    ("umnn", UMNNBijectorConfig(num_hidden=3), inv_unconstrained_monotone),
    ("sos", SumOfSquaresBijectorConfig(num_polys=2, degree=3), inv_sos_polynomial),
    ("bernstein", BernsteinBijectorConfig(degree=8), inv_bernstein),
    ("mixcdf", MixtureCDFBijectorConfig(num_components=3), inv_mixture_cdf),
]

FLOWS = [naf, unaf, sospf, bpf, gf]


@pytest.mark.parametrize("name,config,analytic", BIJECTORS)
def test_monotone_bijector_roundtrip_and_logdet(name, config, analytic):
    params = 0.3 * jax.random.normal(
        jax.random.key(abs(hash(name)) % 100), (3, config.params_dim())
    )
    x = jnp.array([-1.2, 0.1, 2.3])

    # forward (root solve) then analytic inverse recovers x
    y = config(params, x)
    x_rec, logdet = inverse_and_logabsdet(config, invertible_arg=1)(params, y)
    assert jnp.allclose(x_rec, x, atol=1e-3)

    # the analytic log-det matches autodiff of the analytic map, summed over
    # the batch (inverse_and_logabsdet reduces it)
    def analytic_value(t, p):
        return analytic(t, *config.unpack(p))[0]

    grad = jax.vmap(jax.grad(analytic_value))(y, params)
    assert jnp.allclose(logdet, jnp.sum(jnp.log(jnp.abs(grad))), atol=1e-4)

    # zero params give exactly the identity
    zeros = jnp.zeros((3, config.params_dim()))
    assert jnp.allclose(config(zeros, x), x, atol=1e-5)


@pytest.mark.parametrize("ctor", FLOWS)
def test_flow_trains_and_normalizes(ctor):
    data = jax.random.normal(jax.random.key(0), (512, 2)) * jnp.array([
        1.5,
        0.5,
    ]) + jnp.array([2.0, -1.0])

    flow = ctor(2, 2, rngs=nnx.Rngs(0))
    losses = flow.fit(jax.random.key(1), data, num_steps=150, batch_size=128)
    distribution = flow.as_dist()
    assert losses[-1] < losses[0]
    assert jnp.all(jnp.isfinite(losses))

    # density normalizes after training
    grid = jnp.linspace(-8.0, 10.0, 200)
    mesh_x, mesh_y = jnp.meshgrid(grid, grid)
    points = jnp.stack([mesh_x.ravel(), mesh_y.ravel()], -1)
    integral = jnp.sum(jnp.exp(distribution.logpdf(points))) * (18.0 / 200) ** 2
    assert jnp.abs(integral - 1.0) < 0.05

    # sampling (root-solve direction) is finite and consistent with logpdf
    samples = distribution.sample(jax.random.key(2), (64,))
    assert samples.shape == (64, 2)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(distribution.logpdf(samples)))


@pytest.mark.parametrize("ctor", [naf, unaf, sospf, bpf])
def test_autoregressive_monotone_flow_conditional(ctor):
    flow = ctor(2, 2, rngs=nnx.Rngs(0), context_features=1)
    context = jax.random.normal(jax.random.key(1), (128, 1))
    data = jax.random.normal(jax.random.key(2), (128, 2)) + context
    losses = flow.fit(
        jax.random.key(3), data, context=context, num_steps=40, batch_size=64
    )
    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < losses[0]
