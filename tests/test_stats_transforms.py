import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.generative.nflows import (
    AffineBijectorConfig,
    AutoregressiveMLP,
    nsf,
)
from probjax.stats import (
    InvertibleTransformProtocol,
    TransformedDistribution,
    ensure_invertible,
    forward_and_logdet,
    indep,
    norm,
    transformed,
)


def _standard_normal_base(dim):
    return indep(norm(jnp.zeros(dim), jnp.ones(dim)))


def test_transformed_distribution_affine_matches_analytic():
    scale = jnp.array([2.0, 0.5])
    shift = jnp.array([1.0, -3.0])
    base = _standard_normal_base(2)
    dist = TransformedDistribution(base, lambda x: scale * x + shift)

    x = jax.random.normal(jax.random.key(0), (16, 2))
    expected = jnp.sum(
        norm.logpdf((x - shift) / scale, 0.0, 1.0) - jnp.log(jnp.abs(scale)), axis=-1
    )
    assert jnp.allclose(dist.logpdf(x), expected, atol=1e-5)

    samples = dist.rvs(jax.random.key(1), shape=(1000,))
    assert samples.shape == (1000, 2)
    assert jnp.allclose(jnp.mean(samples, axis=0), shift, atol=0.2)


def test_transformed_distribution_round_trip_on_flow_base():
    flow = nsf(2, 2, rngs=nnx.Rngs(0))
    flow_dist = flow.as_dist()
    dist = TransformedDistribution(flow_dist, lambda x: x + 1.0)

    samples = dist.rvs(jax.random.key(0), shape=(8,))
    assert samples.shape == (8, 2)
    logprob = dist.logpdf(samples)
    assert logprob.shape == (8,)
    assert jnp.all(jnp.isfinite(logprob))

    # Shifting by a constant preserves density values at shifted points.
    assert jnp.allclose(logprob, flow_dist.logpdf(samples - 1.0), atol=1e-4)


def test_transformed_distribution_stacks():
    base = _standard_normal_base(2)
    inner = TransformedDistribution(base, lambda x: 2.0 * x)
    outer = TransformedDistribution(inner, lambda x: x + 1.0)

    x = jax.random.normal(jax.random.key(2), (4, 2))
    expected = jnp.sum(norm.logpdf((x - 1.0) / 2.0, 0.0, 1.0) - jnp.log(2.0), axis=-1)
    assert jnp.allclose(outer.logpdf(x), expected, atol=1e-5)


def test_ensure_invertible_passthrough_and_wrap():
    ar = AutoregressiveMLP(2, 2, AffineBijectorConfig(), rngs=nnx.Rngs(0))
    assert ensure_invertible(ar) is ar  # satisfies the protocol already

    # Scalar broadcast: the logdet must count once per output element.
    wrapped = ensure_invertible(lambda x: 3.0 * x)
    assert isinstance(wrapped, InvertibleTransformProtocol)
    y = jnp.array([6.0, -3.0])
    x, logdet = wrapped.inverse_and_logdet(y)
    assert jnp.allclose(x, y / 3.0)
    assert jnp.allclose(logdet, -2.0 * jnp.log(3.0))


def test_forward_and_logdet_matches_inverse():
    def transform(x):
        return 2.0 * x + 1.0

    x = jnp.array([0.5, -1.5])
    y, logdet = forward_and_logdet(transform, x)
    assert jnp.allclose(y, transform(x))
    assert jnp.allclose(logdet, 2.0 * jnp.log(2.0))


def test_frozen_transformed_fast_path_matches_auto_inversion():
    ar = AutoregressiveMLP(2, 2, AffineBijectorConfig(), rngs=nnx.Rngs(0))
    base = _standard_normal_base(2)

    # `ar` has inverse_and_logdet -> protocol fast path.
    fast = transformed(base_dist=base, bijector=ar)
    # A bare lambda hides the method -> jaxpr auto-inversion.
    auto = transformed(base_dist=base, bijector=lambda x: ar(x))

    x = jax.random.normal(jax.random.key(3), (8, 2))
    assert jnp.allclose(fast.logpdf(x), auto.logpdf(x), atol=1e-4)


def test_frozen_transformed_accepts_neural_bases():
    flow = nsf(2, 2, rngs=nnx.Rngs(0))
    dist = transformed(base_dist=flow.as_dist(), bijector=lambda x: x + 2.0)
    samples = dist.rvs(jax.random.key(0), shape=(8,))
    assert samples.shape == (8, 2)
    logprob = dist.logpdf(samples)
    assert jnp.all(jnp.isfinite(logprob))
