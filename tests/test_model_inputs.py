"""Named model inputs reach the network through fit, as_dist and sampling."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn import EDM, DiffusionTransformer, LinearFlow, maf

TOKENS, FEATURES, LOCAL = 5, 3, 2


def make_net():
    return DiffusionTransformer(
        FEATURES,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        attn_size=4,
        time_embed_dim=8,
        fourier_dim=8,
        local_context_dim=LOCAL,
        rngs=nnx.Rngs(0),
    )


@pytest.fixture
def data():
    key = jax.random.key(0)
    x = jax.random.normal(key, (16, TOKENS, FEATURES))
    local = jax.random.normal(jax.random.key(1), (16, TOKENS, LOCAL))
    return x, local


def test_fit_forwards_any_named_input(data):
    x, local = data
    model = EDM(make_net(), event_spec=(TOKENS, FEATURES), num_steps=2)
    losses = model.fit(
        jax.random.key(2),
        {"data": x, "local_context": local},
        num_steps=2,
        batch_size=8,
    )
    assert jnp.all(jnp.isfinite(losses))


def test_fit_rejects_loose_inputs(data):
    x, local = data
    model = EDM(make_net(), event_spec=(TOKENS, FEATURES), num_steps=2)
    with pytest.raises(ValueError, match="pass named model inputs inside data"):
        model.fit(jax.random.key(2), x, local_context=local)


def test_sampling_and_conditioning_with_named_inputs(data):
    _, local = data
    model = LinearFlow(make_net(), event_spec=(TOKENS, FEATURES))
    dist = model.as_dist(inputs_spec={"local_context": (TOKENS, LOCAL)})
    out = dist.sample(jax.random.key(3), (4,), local_context=local[:4])
    assert out.shape == (4, TOKENS, FEATURES)
    # Bound inputs are used by default and can be overridden per call.
    bound = dist.condition(local_context=local[:4])
    assert bound.sample(jax.random.key(3), (4,)).shape == out.shape
    assert jnp.allclose(bound.sample(jax.random.key(3), (4,)), out)
    other = bound.sample(jax.random.key(3), (4,), local_context=local[4:8])
    # A freshly initialised net barely reacts to its input (~1e-6), which
    # allclose's default tolerance would hide; exact equality would not.
    assert not jnp.array_equal(other, out)


def test_bound_inputs_without_a_spec_describe_a_single_example(data):
    _, local = data
    model = LinearFlow(make_net(), event_spec=(TOKENS, FEATURES))
    dist = model.as_dist(inputs={"local_context": local[0]})
    assert dist.sample(jax.random.key(3), (4,)).shape == (4, TOKENS, FEATURES)


def test_context_stays_sugar_for_the_context_input():
    flow = maf(2, 2, context_features=3, rngs=nnx.Rngs(0))
    context = jnp.ones((4, 3))
    via_context = flow.as_dist(context_spec=(3,)).logpdf(
        jnp.ones((4, 2)), context=context
    )
    via_inputs = flow.as_dist(inputs_spec={"context": (3,)}).logpdf(
        jnp.ones((4, 2)), context=context
    )
    assert jnp.allclose(via_context, via_inputs)


def test_context_only_families_reject_other_inputs():
    flow = maf(2, 2, context_features=3, rngs=nnx.Rngs(0))
    dist = flow.as_dist(inputs_spec={"extra": (3,)})
    with pytest.raises(ValueError, match="only accepts a context input"):
        dist.logpdf(jnp.ones((4, 2)), extra=jnp.ones((4, 3)))
