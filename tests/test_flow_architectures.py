import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn import bpf, gf, naf, sospf, unaf
from probjax.stats.bijective.monotone import (
    _bernstein_value_and_logdet,
    _dsf_value_and_logdet,
    _mixture_cdf_value_and_logdet,
    _sos_value_and_logdet,
    _umnn_value_and_logdet,
    bernstein_bijector,
    deep_sigmoid_bijector,
    mixture_cdf_bijector,
    sos_polynomial_bijector,
    unconstrained_monotone_bijector,
)

BIJECTORS = [
    ("dsf", deep_sigmoid_bijector, _dsf_value_and_logdet, 9),
    ("umnn", unconstrained_monotone_bijector, _umnn_value_and_logdet, 11),
    ("sos", sos_polynomial_bijector, _sos_value_and_logdet, 9),
    ("bernstein", bernstein_bijector, _bernstein_value_and_logdet, 8),
    ("mixcdf", mixture_cdf_bijector, _mixture_cdf_value_and_logdet, 9),
]

FLOWS = [naf, unaf, sospf, bpf, gf]


@pytest.mark.parametrize("name,forward,analytic,bijector_dim", BIJECTORS)
def test_monotone_bijector_roundtrip_and_logdet(name, forward, analytic, bijector_dim):
    params = 0.3 * jax.random.normal(jax.random.key(abs(hash(name)) % 100), (3, bijector_dim))
    x = jnp.array([-1.2, 0.1, 2.3])

    # forward (root solve) then analytic inverse recovers x
    y = forward(params, x)
    x_rec, logdet = analytic(params, y)
    assert jnp.allclose(x_rec, x, atol=1e-3)

    # analytic logdet matches autodiff of the analytic map
    grad = jax.vmap(
        jax.grad(lambda t, p: analytic(p[None], t[None])[0][0], argnums=0)
    )(y, params)
    assert jnp.allclose(logdet, jnp.log(jnp.abs(grad)), atol=1e-4)

    # zero params give a well-conditioned (near-identity) map
    _, logdet0 = analytic(jnp.zeros((3, bijector_dim)), x)
    slope = jnp.exp(logdet0)
    assert jnp.all(slope > 0.3) and jnp.all(slope < 3.0)


@pytest.mark.parametrize("ctor", FLOWS)
def test_flow_trains_and_normalizes(ctor):
    data = jax.random.normal(jax.random.key(0), (512, 2)) * jnp.array(
        [1.5, 0.5]
    ) + jnp.array([2.0, -1.0])

    flow = ctor(2, 2, rngs=nnx.Rngs(0))
    losses = flow.fit(jax.random.key(1), data, num_steps=150, batch_size=128)
    assert losses[-1] < losses[0]
    assert jnp.all(jnp.isfinite(losses))

    # density normalizes after training
    grid = jnp.linspace(-8.0, 10.0, 200)
    mesh_x, mesh_y = jnp.meshgrid(grid, grid)
    points = jnp.stack([mesh_x.ravel(), mesh_y.ravel()], -1)
    integral = jnp.sum(jnp.exp(flow.logpdf(points))) * (18.0 / 200) ** 2
    assert jnp.abs(integral - 1.0) < 0.05

    # sampling (root-solve direction) is finite and consistent with logpdf
    samples = flow.sample(jax.random.key(2), (64,))
    assert samples.shape == (64, 2)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(flow.logpdf(samples)))


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
