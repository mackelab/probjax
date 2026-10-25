"""Tests for the autoregressive density model.

The load-bearing property is the masking: dimension ``i``'s parameters must
depend on ``x_<i`` and nothing else. If that leaks, the model still trains and
still produces plausible-looking numbers -- it just is not a density any more,
because the factorisation it claims no longer holds.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn.generative.autoregressive import (
    Autoregressive,
    CategoricalHead,
    HistogramHead,
    LocScaleHead,
    MLPARConditionerConfig,
    MixtureHead,
    NormalHead,
    SplineHead,
    TransformerARConditionerConfig,
)
from probjax.stats import gennorm, laplace, logistic, norm, t

HEADS = [
    pytest.param(NormalHead(), id="normal"),
    pytest.param(LocScaleHead(laplace), id="laplace"),
    pytest.param(LocScaleHead(logistic), id="logistic"),
    pytest.param(MixtureHead(6), id="mixture"),
    pytest.param(MixtureHead(6, kernel="logistic"), id="logistic-mixture"),
    pytest.param(SplineHead(6), id="spline"),
    pytest.param(HistogramHead(12, tails=True), id="histogram-tails"),
    pytest.param(HistogramHead(12, tails=False), id="histogram-bounded"),
]


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head", HEADS)
def test_params_cover_the_family_and_check_the_width(head):
    params = head.params(jnp.zeros((3, head.num_params)))
    assert set(params) == set(head.dist.parameters)

    with pytest.raises(ValueError, match="raw parameters"):
        head.params(jnp.zeros((3, head.num_params + 1)))


@pytest.mark.parametrize("head", HEADS)
def test_zero_parameters_give_a_usable_density(head):
    """A zero-initialised head must already be a sane distribution."""
    x = jnp.array([-1.0, 0.0, 0.8])
    lp = head.logpdf(jnp.zeros((3, head.num_params)), x)
    assert jnp.all(jnp.isfinite(lp))
    # a sane density over standardised data, not a spike or a near-zero
    assert jnp.all(lp > -8.0) and jnp.all(lp < 2.0)


@pytest.mark.parametrize("head", HEADS)
def test_head_logpdf_is_differentiable(head):
    def total(raw):
        return jnp.sum(head.logpdf(raw, jnp.array([0.2, -0.4])))

    g = jax.grad(total)(jnp.zeros((2, head.num_params)))
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0)


def test_location_scale_head_takes_any_such_family():
    """``gennorm`` is here for ``logpdf`` only -- its ``_rvs_impl`` branches on
    ``if beta == 2.0``, which raises for array-valued parameters."""
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 3))
    heads = [
        LocScaleHead(norm),
        LocScaleHead(laplace),
        LocScaleHead(logistic),
        LocScaleHead(gennorm, {"beta": 1.5}),
        # integer-constrained parameters are fixed, not predicted
        LocScaleHead(t, {"df": 5.0}),
    ]
    for head in heads:
        assert head.num_params == 2
        lp = Autoregressive(3, head, nnx.Rngs(0))._logpdf(x)
        assert lp.shape == (8,)
        assert jnp.all(jnp.isfinite(lp))


def test_spline_head_has_the_same_width_as_the_flow_config():
    """Both pack the same spline, so both must need the same parameter count."""
    from probjax.nn.generative.nflows.config import RationalQuadraticSplineConfig

    for num_bins in (4, 8, 16):
        assert (
            SplineHead(num_bins).num_params
            == RationalQuadraticSplineConfig(num_bins=num_bins).params_dim()
        )


# ---------------------------------------------------------------------------
# Autoregressive structure -- the property everything else rests on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_dim", [2, 4, 6])
def test_parameters_of_dimension_i_depend_only_on_earlier_inputs(input_dim):
    model = Autoregressive(input_dim, rngs=nnx.Rngs(0))
    _perturb(model, jax.random.PRNGKey(1))

    params_dim = model.head.num_params
    jac = jax.jacrev(lambda v: model.predict_params(v).reshape(-1))(
        jnp.zeros(input_dim)
    )  # (input_dim * params_dim, input_dim)

    for i in range(input_dim):
        block = jac[i * params_dim : (i + 1) * params_dim]
        for j in range(input_dim):
            dependence = jnp.max(jnp.abs(block[:, j]))
            if j >= i:
                assert dependence == 0.0, (
                    f"dimension {i} parameters depend on x_{j}, breaking the "
                    "autoregressive factorisation"
                )


def test_categorical_masking_is_autoregressive():
    """Probed discretely: a one-hot of an integer has zero gradient everywhere,
    so the jacobian test above cannot see a discrete head."""
    input_dim, num_categories = 4, 3
    model = Autoregressive(input_dim, CategoricalHead(num_categories), nnx.Rngs(0))
    _perturb(model, jax.random.PRNGKey(1))

    base = model.predict_params(jnp.zeros(input_dim, dtype=jnp.int32))
    for j in range(input_dim):
        probe = jnp.zeros(input_dim, dtype=jnp.int32).at[j].set(2)
        changed = jnp.max(jnp.abs(model.predict_params(probe) - base), axis=-1)
        for i in range(input_dim):
            if i <= j:
                assert changed[i] == 0.0, f"dimension {i} saw x_{j}"
        assert changed[j + 1] > 0.0 if j + 1 < input_dim else True


def test_later_dimensions_actually_use_the_earlier_ones():
    """The complement of the masking test: the model must not ignore its input."""
    model = Autoregressive(4, rngs=nnx.Rngs(0))
    _perturb(model, jax.random.PRNGKey(2))
    params_dim = model.head.num_params
    jac = jax.jacrev(lambda v: model.predict_params(v).reshape(-1))(jnp.zeros(4))
    for i in range(1, 4):
        block = jac[i * params_dim : (i + 1) * params_dim]
        assert jnp.max(jnp.abs(block[:, :i])) > 0.0, f"dimension {i} ignores x_<{i}"


def _perturb(model, key, scale=0.4):
    state = nnx.state(model, nnx.Param)
    leaves, tree = jax.tree_util.tree_flatten(state)
    keys = jax.random.split(key, len(leaves))
    nnx.update(
        model,
        jax.tree_util.tree_unflatten(
            tree,
            [
                l + scale * jax.random.normal(k, jnp.shape(l))
                for l, k in zip(leaves, keys)
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Density, sampling, training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head", HEADS)
def test_logpdf_and_sampling_are_finite(head):
    model = Autoregressive(3, head, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 3))

    lp = model._logpdf(x)
    assert lp.shape == (8,)
    assert jnp.all(jnp.isfinite(lp))
    # the joint is the sum of the conditionals
    assert jnp.allclose(lp, jnp.sum(model.conditional_logpdfs(x), axis=-1))

    samples = model.sample(jax.random.PRNGKey(1), (16,))
    assert samples.shape == (16, 3)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(model._logpdf(samples)))


@pytest.mark.parametrize(
    "head",
    [
        pytest.param(NormalHead(), id="normal"),
        pytest.param(MixtureHead(6), id="mixture"),
        pytest.param(SplineHead(6), id="spline"),
        pytest.param(HistogramHead(16), id="histogram"),
    ],
)
def test_trained_density_is_normalised(head):
    """A wrong conditional or a leaky mask shows up as mass != 1."""
    from tests.test_nflow_density_estimation import crescent

    model = Autoregressive(2, head, nnx.Rngs(0))
    train = crescent(jax.random.PRNGKey(0), 4000)
    losses = model.fit(
        jax.random.PRNGKey(1),
        train,
        num_steps=400,
        batch_size=512,
    ).losses
    assert jnp.all(jnp.isfinite(losses))
    assert losses[-1] < losses[0]

    z = jax.random.normal(jax.random.PRNGKey(2), (40_000, 2)) * 6.0
    log_q = jnp.sum(jax.scipy.stats.norm.logpdf(z, 0.0, 6.0), -1)
    lp = jax.vmap(model._logpdf)(z)
    finite = jnp.isfinite(lp)
    assert jnp.mean(finite) > 0.95
    mass = jnp.mean(jnp.where(finite, jnp.exp(lp - log_q), 0.0))
    assert 0.9 < mass < 1.1, f"total mass {float(mass)}"


def test_reaches_the_analytic_optimum_on_a_gaussian_target():
    """A curved Gaussian target whose entropy is known in closed form."""
    from tests.test_nflow_density_estimation import OPTIMAL_NLL, crescent

    model = Autoregressive(2, MixtureHead(8), nnx.Rngs(0))
    train, test = (
        crescent(jax.random.PRNGKey(0), 8000),
        crescent(jax.random.PRNGKey(1), 2000),
    )
    model.fit(
        jax.random.PRNGKey(2),
        train,
        num_steps=800,
        batch_size=512,
    )

    nll = float(-jnp.mean(jax.vmap(model._logpdf)(test)))
    optimum = OPTIMAL_NLL["crescent"]
    assert optimum - 0.1 < nll < optimum + 0.15, f"NLL {nll} vs optimum {optimum}"


def test_context_conditioning_changes_the_density():
    model = Autoregressive(3, rngs=nnx.Rngs(0), context_features=2)
    _perturb(model, jax.random.PRNGKey(3))
    x = jax.random.normal(jax.random.PRNGKey(4), (8, 3))
    c1 = jnp.zeros((8, 2))
    c2 = jnp.ones((8, 2))

    lp1 = jax.vmap(lambda a, c: model._logpdf(a, context=c))(x, c1)
    lp2 = jax.vmap(lambda a, c: model._logpdf(a, context=c))(x, c2)
    assert jnp.all(jnp.isfinite(lp1)) and jnp.all(jnp.isfinite(lp2))
    assert not jnp.allclose(lp1, lp2), "context is ignored"

    loss = model.loss(None, x, context=c1)
    assert jnp.isfinite(loss)


def test_as_dist_samples_and_scores():
    model = Autoregressive(3, rngs=nnx.Rngs(0))
    dist = model.as_dist()

    samples = dist.rvs(jax.random.PRNGKey(0), (16,))
    assert samples.shape == (16, 3)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(dist.logpdf(samples)))
    assert dist.rvs(jax.random.PRNGKey(1), (5,)).shape == (5, 3)


def test_bounded_head_rejects_out_of_range_training_data():
    head = HistogramHead(8, low=-1.0, high=1.0, tails=False)
    model = Autoregressive(2, head, nnx.Rngs(0), standardize=False)
    with pytest.raises(ValueError, match="bounded support"):
        model.fit(jax.random.key(0), jnp.array([[0.0, 5.0]]), num_steps=1)
    model.fit(jax.random.key(0), jnp.array([[0.0, 0.5]]), num_steps=1)


def test_tailed_head_accepts_any_range():
    model = Autoregressive(2, HistogramHead(8, low=-1.0, high=1.0), nnx.Rngs(0))
    assert jnp.isfinite(model.loss(None, jnp.array([[0.0, 50.0]])))


# ---------------------------------------------------------------------------
# Discrete data
# ---------------------------------------------------------------------------


def test_categorical_head_starts_uniform_and_samples_in_range():
    num_categories, input_dim = 4, 3
    model = Autoregressive(input_dim, CategoricalHead(num_categories), nnx.Rngs(0))
    assert model.head.num_params == num_categories

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (32, input_dim), 0, num_categories
    )
    expected = input_dim * jnp.log(1.0 / num_categories)
    assert jnp.allclose(jnp.mean(model._logpdf(tokens)), expected, atol=1e-5)

    samples = model.sample(jax.random.PRNGKey(1), (64,))
    assert samples.shape == (64, input_dim)
    assert samples.dtype == jnp.int32
    assert jnp.all((samples >= 0) & (samples < num_categories))


def test_categorical_model_recovers_a_known_joint():
    """Fit a chain where x1 is a deterministic-ish function of x0."""
    num_categories, n = 3, 4000
    key = jax.random.PRNGKey(0)
    k0, k1 = jax.random.split(key)
    x0 = jax.random.categorical(k0, jnp.log(jnp.array([0.6, 0.3, 0.1])), shape=(n,))
    # x1 = x0 shifted by one, with 10% noise
    noise = jax.random.bernoulli(k1, 0.1, (n,))
    x1 = jnp.where(noise, (x0 + 2) % num_categories, (x0 + 1) % num_categories)
    data = jnp.stack([x0, x1], -1)

    model = Autoregressive(2, CategoricalHead(num_categories), nnx.Rngs(0))
    losses = model.fit(
        jax.random.PRNGKey(2),
        data,
        num_steps=800,
        batch_size=512,
    ).losses
    assert jnp.all(jnp.isfinite(losses))

    # the learned p(x1 | x0) should concentrate on (x0 + 1) % K
    for value in range(num_categories):
        probe = jnp.array([[value, 0]])
        params = model.predict_params(probe)
        probs = model.head.params(params[:, 1, :])["probs"][0]
        assert jnp.argmax(probs) == (value + 1) % num_categories
        assert probs[(value + 1) % num_categories] > 0.6


def test_marginal_of_the_first_dimension_matches_the_data():
    """The first conditional has no inputs, so it must fit the marginal exactly.

    Standardisation is off here: with it on the head works in standardised
    coordinates and would report loc 0 / scale 1 whatever the data looks like.
    """
    n = 20_000
    data = jnp.stack(
        [
            jax.random.normal(jax.random.PRNGKey(0), (n,)) * 0.5 + 2.0,
            jax.random.normal(jax.random.PRNGKey(1), (n,)),
        ],
        -1,
    )
    model = Autoregressive(2, rngs=nnx.Rngs(0), standardize=False)
    model.fit(
        jax.random.PRNGKey(2),
        data,
        num_steps=600,
        batch_size=512,
    )

    params = model.predict_params(jnp.zeros(2))
    natural = model.head.params(params[0])
    assert jnp.allclose(natural["loc"], 2.0, atol=0.15)
    assert jnp.allclose(natural["scale"], 0.5, atol=0.15)


def test_conditioner_output_width_matches_the_head():
    """Regression guard: every conditioner emits (..., input_dim, num_params)."""
    for head in (NormalHead(), MixtureHead(7), SplineHead(5)):
        model = Autoregressive(4, head, nnx.Rngs(0))
        out = model.conditioner(model._tokens(jnp.zeros((2, 4))))
        assert out.shape == (2, 4, head.num_params)


def test_custom_conditioner_depth_is_respected():
    model = Autoregressive(
        3,
        NormalHead(),
        nnx.Rngs(0),
        conditioner=MLPARConditionerConfig(hidden_dims=(16, 16, 16)),
    )
    assert jnp.isfinite(model.loss(None, jnp.zeros((4, 3))))


# ---------------------------------------------------------------------------
# Transformer conditioner: discrete support, KV-cache parity, prefixes
# ---------------------------------------------------------------------------


def _tiny_transformer():
    return TransformerARConditionerConfig(
        num_layers=1, num_heads=2, model_dim=8, attn_size=4
    )


def test_transformer_conditioner_supports_categorical():
    """One-hot inputs must stay one token per position, not widen into
    one token per class."""
    model = Autoregressive(
        4, CategoricalHead(3), nnx.Rngs(1), conditioner=_tiny_transformer()
    )
    x = jnp.zeros((2, 4), jnp.int32)
    assert model.predict_params(x).shape == (2, 4, 3)
    assert model._logpdf(x).shape == (2,)
    assert jnp.all(jnp.isfinite(model._logpdf(x)))
    samples = model.sample(jax.random.key(0), (5,))
    assert samples.shape == (5, 4)
    assert samples.dtype == jnp.int32


def test_ssm_conditioner_supports_categorical():
    """The recurrent conditioner trains and samples without a cache."""
    from probjax.nn.generative.autoregressive import SSMARConditionerConfig

    model = Autoregressive(
        4,
        CategoricalHead(3),
        nnx.Rngs(1),
        conditioner=SSMARConditionerConfig(model_dim=16, num_layers=1),
    )
    x = jnp.zeros((2, 4), jnp.int32)
    assert model.predict_params(x).shape == (2, 4, 3)
    assert model._logpdf(x).shape == (2,)
    assert jnp.all(jnp.isfinite(model._logpdf(x)))
    data = (jax.random.uniform(jax.random.key(0), (256, 4)) > 0.5).astype(jnp.int32)
    losses = model.fit(
        jax.random.key(1),
        data,
        num_steps=10,
        batch_size=64,
    ).losses
    assert jnp.all(jnp.isfinite(losses))
    samples = model.sample(jax.random.key(0), (5,))
    assert samples.shape == (5, 4)
    assert samples.dtype == jnp.int32


@pytest.mark.parametrize(
    "head",
    [
        pytest.param(NormalHead(), id="normal"),
        pytest.param(CategoricalHead(3), id="categorical"),
    ],
)
def test_cached_sampling_matches_naive(head):
    """The compiled scan carries its KV cache and matches naive decoding."""
    input_dim = 4
    model = Autoregressive(input_dim, head, nnx.Rngs(0), conditioner=_tiny_transformer())
    decoder = model.conditioner.inner.decoder
    decoder.kernel[...] = 0.1 * jax.random.normal(
        jax.random.key(1), decoder.kernel.shape
    )
    key = jax.random.key(0)
    cached = model.sample(key, (3,))
    naive = model.sample(key, (3,), use_cache=False)

    for block in model.conditioner.inner.transformer.attention_blocks:
        assert block.cache_index[...] == input_dim
    assert jnp.allclose(cached, naive, atol=1e-5, rtol=1e-5)


def test_trained_cached_matches_naive_up_to_float_dust():
    """Untrained parity is vacuous (the zero-init head emits exact zeros),
    so train briefly and compare parameters with tolerance: the compiled
    scan and the eager loop order operations differently."""
    model = Autoregressive(
        4, CategoricalHead(3), nnx.Rngs(1), conditioner=_tiny_transformer()
    )
    data = (jax.random.uniform(jax.random.key(0), (256, 4)) > 0.5).astype(jnp.int32)
    model.fit(
        jax.random.key(1),
        data,
        num_steps=10,
        batch_size=64,
    )
    x = (jax.random.uniform(jax.random.key(2), (3, 4)) > 0.5).astype(jnp.int32)
    full = model.predict_params(x)
    model.conditioner.init_decode((3,))
    prev = jnp.zeros((3, 3))
    for i in range(4):
        stepwise = model.conditioner.predict_next(prev, i, None)
        assert jnp.allclose(stepwise, full[:, i, :], atol=1e-5)
        prev = model._tokens(x[:, i : i + 1])[:, 0, :]


def test_prefix_completion_clamps_and_matches_naive():
    model = Autoregressive(
        4, CategoricalHead(3), nnx.Rngs(1), conditioner=_tiny_transformer()
    )
    key = jax.random.key(0)
    prefix = jnp.arange(12).reshape(3, 4) % 3
    cached = model.sample(key, (3,), prefix=prefix, prefix_len=2)
    naive = model.sample(key, (3,), prefix=prefix, prefix_len=2, use_cache=False)
    assert jnp.array_equal(cached[:, :2], prefix[:, :2])
    assert jnp.array_equal(cached, naive)
    # prefix_len=0 reproduces plain sampling
    assert jnp.array_equal(
        model.sample(key, (3,), prefix=prefix, prefix_len=0),
        model.sample(key, (3,)),
    )


def test_prefix_arguments_are_validated():
    model = Autoregressive(
        4, NormalHead(), nnx.Rngs(0), conditioner=_tiny_transformer()
    )
    key = jax.random.key(0)
    with pytest.raises(ValueError, match="prefix_len"):
        model.sample(key, (2,), prefix_len=5)
    with pytest.raises(ValueError, match="prefix_len"):
        model.sample(key, (2,), prefix_len=1)
    with pytest.raises(ValueError, match="prefix_len"):
        model.sample(key, (2,), prefix=jnp.zeros((2, 4)), prefix_len=5)
