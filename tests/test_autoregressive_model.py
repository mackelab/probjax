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
    ARFamily,
    AutoregressiveModel,
    CategoricalAutoregressive,
    HistogramAutoregressive,
    MADE,
    MixtureAutoregressive,
    MLPARConditionerConfig,
    SplineAutoregressive,
)
from probjax.stats import gennorm, laplace, logistic, norm, t

FAMILIES = [
    pytest.param(ARFamily.normal(), id="normal"),
    pytest.param(ARFamily(laplace), id="laplace"),
    pytest.param(ARFamily(logistic), id="logistic"),
    pytest.param(ARFamily.mixture(6), id="mixture"),
    pytest.param(ARFamily.mixture(6, kernel="logistic"), id="logistic-mixture"),
    pytest.param(ARFamily.spline(6), id="spline"),
    pytest.param(ARFamily.histogram(12, tails=True), id="histogram-tails"),
    pytest.param(ARFamily.histogram(12, tails=False), id="histogram-bounded"),
]


# ---------------------------------------------------------------------------
# The family adapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_params_dim_matches_what_unpack_consumes(family):
    params = jnp.zeros((3, family.params_dim()))
    natural = family.unpack(params)
    assert set(natural) == set(family.dist.parameters)

    with pytest.raises(ValueError, match="trailing dimension"):
        family.unpack(jnp.zeros((3, family.params_dim() + 1)))


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_parameters_give_a_usable_density(family):
    """A zero-initialised head must already be a sane distribution."""
    x = jnp.array([-1.0, 0.0, 0.8])
    natural = family.unpack(jnp.zeros((3, family.params_dim())))
    lp = family.logpdf(x, natural)
    assert jnp.all(jnp.isfinite(lp))
    # a sane density over standardised data, not a spike or a near-zero
    assert jnp.all(lp > -8.0) and jnp.all(lp < 2.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_unpack_is_differentiable(family):
    def total(p):
        return jnp.sum(family.logpdf(jnp.array([0.2, -0.4]), family.unpack(p)))

    g = jax.grad(total)(jnp.zeros((2, family.params_dim())))
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0)


def test_any_univariate_stats_family_can_be_used_for_the_density():
    """The point of the adapter: no per-family code, just the distribution.

    ``gennorm`` is here for ``logpdf`` only -- its ``_rvs_impl`` branches on
    ``if beta == 2.0``, which raises for array-valued parameters, so it cannot
    be sampled per-dimension. That is an upstream limitation of the family, not
    of this adapter.
    """
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 3))
    for dist in (norm, laplace, logistic, gennorm):
        model = AutoregressiveModel(3, ARFamily(dist), nnx.Rngs(0))
        lp = model._logpdf(x)
        assert lp.shape == (8,)
        assert jnp.all(jnp.isfinite(lp))


def test_integer_constrained_parameters_are_rejected_with_a_clear_error():
    """`t.df` cannot be predicted; the error must say so and name it."""
    with pytest.raises(TypeError, match="df"):
        ARFamily(t).params_dim()
    # pinning it is the documented way out
    family = ARFamily(t, fixed={"df": 5.0})
    assert family.params_dim() == 2


def test_fixed_names_are_validated():
    with pytest.raises(ValueError, match="not parameters"):
        ARFamily(norm, fixed={"nonsense": 1.0})
    with pytest.raises(ValueError, match="nothing for the conditioner"):
        ARFamily(norm, fixed={"loc": 0.0, "scale": 1.0})


def test_spline_family_reports_the_same_width_as_the_flow_config():
    """Both pack the same spline, so both must need the same parameter count."""
    from probjax.nn.generative.nflows.config import RationalQuadraticSplineConfig

    for num_bins in (4, 8, 16):
        assert (
            ARFamily.spline(num_bins).params_dim()
            == RationalQuadraticSplineConfig(num_bins=num_bins).params_dim()
        )


# ---------------------------------------------------------------------------
# Autoregressive structure -- the property everything else rests on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_dim", [2, 4, 6])
def test_parameters_of_dimension_i_depend_only_on_earlier_inputs(input_dim):
    model = MADE(input_dim, nnx.Rngs(0))
    _perturb(model, jax.random.PRNGKey(1))

    params_dim = model.family.params_dim()
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
    model = CategoricalAutoregressive(
        input_dim, nnx.Rngs(0), num_categories=num_categories
    )
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
    model = MADE(4, nnx.Rngs(0))
    _perturb(model, jax.random.PRNGKey(2))
    params_dim = model.family.params_dim()
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
            [l + scale * jax.random.normal(k, jnp.shape(l)) for l, k in zip(leaves, keys)],
        ),
    )


# ---------------------------------------------------------------------------
# Density, sampling, training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_logpdf_and_sampling_are_finite(family):
    model = AutoregressiveModel(3, family, nnx.Rngs(0))
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
    "build",
    [
        pytest.param(lambda r: MADE(2, r), id="made"),
        pytest.param(lambda r: MixtureAutoregressive(2, r, num_components=6), id="mixture"),
        pytest.param(lambda r: SplineAutoregressive(2, r, num_bins=6), id="spline"),
        pytest.param(lambda r: HistogramAutoregressive(2, r, num_bins=16), id="histogram"),
    ],
)
def test_trained_density_is_normalised(build):
    """A wrong conditional or a leaky mask shows up as mass != 1."""
    from tests.test_nflow_density_estimation import crescent

    model = build(nnx.Rngs(0))
    train = crescent(jax.random.PRNGKey(0), 4000)
    losses = model.fit(jax.random.PRNGKey(1), train, num_steps=400, batch_size=512)
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

    model = MixtureAutoregressive(2, nnx.Rngs(0), num_components=8)
    train, test = crescent(jax.random.PRNGKey(0), 8000), crescent(jax.random.PRNGKey(1), 2000)
    model.fit(jax.random.PRNGKey(2), train, num_steps=800, batch_size=512)

    nll = float(-jnp.mean(jax.vmap(model._logpdf)(test)))
    optimum = OPTIMAL_NLL["crescent"]
    assert optimum - 0.1 < nll < optimum + 0.15, f"NLL {nll} vs optimum {optimum}"


def test_context_conditioning_changes_the_density():
    model = MADE(3, nnx.Rngs(0), context_features=2)
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


def test_as_dist_round_trips_through_the_exported_sampler():
    model = MADE(3, nnx.Rngs(0))
    dist = model.as_dist()

    samples = dist.rvs(jax.random.PRNGKey(0), (16,))
    assert samples.shape == (16, 3)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(dist.logpdf(samples)))
    # a second, different batch size must reuse the same export
    assert dist.rvs(jax.random.PRNGKey(1), (5,)).shape == (5, 3)


def test_bounded_family_rejects_out_of_range_data():
    model = HistogramAutoregressive(
        2, nnx.Rngs(0), num_bins=8, low=-1.0, high=1.0, tails=False
    )
    with pytest.raises(ValueError, match="bounded support"):
        model.loss(None, jnp.array([[0.0, 5.0]]))
    assert jnp.isfinite(model.loss(None, jnp.array([[0.0, 0.5]])))


def test_tailed_family_accepts_any_range():
    model = HistogramAutoregressive(2, nnx.Rngs(0), num_bins=8, low=-1.0, high=1.0)
    assert jnp.isfinite(model.loss(None, jnp.array([[0.0, 50.0]])))


# ---------------------------------------------------------------------------
# Discrete data
# ---------------------------------------------------------------------------


def test_categorical_head_starts_uniform_and_samples_in_range():
    num_categories, input_dim = 4, 3
    model = CategoricalAutoregressive(
        input_dim, nnx.Rngs(0), num_categories=num_categories
    )
    assert model.family.params_dim() == num_categories

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

    model = CategoricalAutoregressive(2, nnx.Rngs(0), num_categories=num_categories)
    losses = model.fit(jax.random.PRNGKey(2), data, num_steps=800, batch_size=512)
    assert jnp.all(jnp.isfinite(losses))

    # the learned p(x1 | x0) should concentrate on (x0 + 1) % K
    for value in range(num_categories):
        probe = jnp.array([[value, 0]])
        params = model.predict_params(probe)
        probs = model.family.unpack(params[:, 1, :])["probs"][0]
        assert jnp.argmax(probs) == (value + 1) % num_categories
        assert probs[(value + 1) % num_categories] > 0.6


def test_marginal_of_the_first_dimension_matches_the_data():
    """The first conditional has no inputs, so it must fit the marginal exactly."""
    n = 20_000
    data = jnp.stack(
        [
            jax.random.normal(jax.random.PRNGKey(0), (n,)) * 0.5 + 2.0,
            jax.random.normal(jax.random.PRNGKey(1), (n,)),
        ],
        -1,
    )
    model = MADE(2, nnx.Rngs(0))
    model.fit(jax.random.PRNGKey(2), data, num_steps=600, batch_size=512)

    params = model.predict_params(jnp.zeros(2))
    natural = model.family.unpack(params[0])
    assert jnp.allclose(natural["loc"], 2.0, atol=0.15)
    assert jnp.allclose(natural["scale"], 0.5, atol=0.15)


def test_conditioner_output_width_matches_the_family():
    """Regression guard: the head must emit exactly input_dim * params_dim."""
    for family in (ARFamily.normal(), ARFamily.mixture(7), ARFamily.spline(5)):
        model = AutoregressiveModel(4, family, nnx.Rngs(0))
        flat = model.conditioner(jnp.zeros(4))
        assert flat.shape == (4 * family.params_dim(),)
        assert model.predict_params(jnp.zeros(4)).shape == (4, family.params_dim())


def test_custom_conditioner_depth_is_respected():
    model = AutoregressiveModel(
        3,
        ARFamily.normal(),
        nnx.Rngs(0),
        conditioner=MLPARConditionerConfig(hidden_dims=(16, 16, 16)),
    )
    assert jnp.isfinite(model.loss(None, jnp.zeros((4, 3))))
