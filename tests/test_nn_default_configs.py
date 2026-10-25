"""Production-sized default profiles, causal behavior, and meaningful noising."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probjax.nn import (
    Autoregressive,
    DiffusionTransformer,
    InducedSelfAttention,
    NormalHead,
    SpatialSelfAttention,
    SplineHead,
    Transformer,
    TransformerARConditionerConfig,
    TransformerConditionerConfig,
)
from probjax.nn.generative.discrete.config import MultinomialDiffusionSchedule
from probjax.nn.generative.nflows.autoregressive import AutoregressiveTransformer
from probjax.nn.generative.nflows.coupling import CouplingTransformer
from probjax.nn.layers.attention import dot_product_attention


def assert_standard_profile(model):
    assert (model.model_dim, model.num_heads, model.attn_size, model.num_layers) == (
        256,
        8,
        32,
        4,
    )
    assert model.widening_factor == 4
    assert all(block.head_dim == 32 for block in model.attention_blocks)


def test_transformer_defaults_and_compact_override():
    model = Transformer(rngs=nnx.Rngs(0))
    assert_standard_profile(model)
    values = jnp.ones((1, 4, 256))
    assert jax.jit(model)(values).shape == values.shape
    compact = Transformer(model_dim=128, num_layers=1, rngs=nnx.Rngs(1))
    assert compact.model_dim == 128 and compact.attention_blocks[0].head_dim == 32
    assert compact(jnp.ones((1, 4, 128))).shape == (1, 4, 128)


def test_diffusion_transformer_uses_standard_heads():
    model = DiffusionTransformer(2, rngs=nnx.Rngs(0))
    assert_standard_profile(model)
    assert model(0.5, jnp.ones((1, 4, 2))).shape == (1, 4, 2)


def test_conditioner_configs_and_direct_layers_agree():
    for config in (TransformerARConditionerConfig(), TransformerConditionerConfig()):
        assert (
            config.model_dim,
            config.num_heads,
            config.attn_size,
            config.num_layers,
        ) == (256, 8, 32, 4)
        assert config.widening_factor == 4
    ar = AutoregressiveTransformer(
        1, 2, None, nnx.Rngs(0), attention_fn=dot_product_attention
    )
    assert_standard_profile(ar.transformer)
    coupling = CouplingTransformer(2, 4, lambda params, x: x, nnx.Rngs(1))
    assert_standard_profile(coupling.transformer)


def test_default_ar_transformer_is_causal_with_nonzero_head():
    config = TransformerARConditionerConfig(attention_fn=dot_product_attention)
    model = Autoregressive(4, NormalHead(), nnx.Rngs(0), conditioner=config)
    decoder = model.conditioner.inner.decoder
    decoder.kernel[...] = 0.01 * jax.random.normal(
        jax.random.key(1), decoder.kernel.shape
    )
    values = jax.random.normal(jax.random.key(2), (2, 4))
    changed = values.at[:, 2].add(10)
    np.testing.assert_allclose(
        model.predict_params(values)[:, :3],
        model.predict_params(changed)[:, :3],
        atol=1e-6,
    )
    assert jnp.isfinite(model._logpdf(values)).all()
    key = jax.random.key(3)
    np.testing.assert_allclose(
        model.sample(key, (2,)),
        model.sample(key, (2,), use_cache=False),
        rtol=1e-5,
        atol=1e-5,
    )


def test_adaptive_attention_width_is_total_not_per_head():
    induced = InducedSelfAttention(256, num_inducing_points=2, rngs=nnx.Rngs(0))
    assert induced.inducing_attn.head_dim == induced.output_attn.head_dim == 32
    assert induced(jnp.ones((1, 4, 256))).shape == (1, 4, 256)
    spatial = SpatialSelfAttention(256, nnx.Rngs(1))
    assert spatial.attn.head_dim == 32
    assert spatial(jnp.ones((1, 2, 2, 256))).shape == (1, 2, 2, 256)


@pytest.mark.parametrize("kind", ["linear", "sigmoid"])
@pytest.mark.parametrize("steps", [16, 128, 1000])
def test_default_categorical_rates_reach_base_prior(kind, steps):
    schedule = getattr(MultinomialDiffusionSchedule, "from_" + kind)(steps, 4)
    tokens = jnp.arange(4)[:, None]
    probabilities = schedule.q_xt_given_x0_probs(tokens, jnp.ones((4, 1)))
    assert float(schedule.alpha_bar(jnp.array(1.0))) < 1e-3
    np.testing.assert_allclose(
        probabilities,
        jnp.broadcast_to(schedule.base_probs, probabilities.shape),
        atol=1e-4,
    )
    legacy = getattr(MultinomialDiffusionSchedule, "from_" + kind)(
        steps, 4, beta_start=1e-4, beta_end=0.1
    )
    assert float(legacy.alpha_bar(jnp.array(1.0))) > 0.9


def test_cosine_beta_is_a_continuous_rate():
    schedule = MultinomialDiffusionSchedule.from_cosine(64, 4)
    rate = float(schedule.beta(jnp.array(0.99)))
    assert 1 < rate <= 50


def test_ar_spline_tails_are_damped_but_learnable():
    damped, undamped = SplineHead(), SplineHead(tail_logit_scale=1.0)
    params = jnp.full((damped.num_params,), 0.3)
    a, b = damped.params(params), undamped.params(params)
    np.testing.assert_allclose(a["knot_slopes"][1:-1], b["knot_slopes"][1:-1])
    endpoints = jnp.array([0, -1])
    assert jnp.all(
        (a["knot_slopes"][endpoints] > 1)
        & (a["knot_slopes"][endpoints] < b["knot_slopes"][endpoints])
    )
