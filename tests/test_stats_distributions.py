import itertools

import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from probjax.stats import (
    # Discrete distributions
    bernoulli,
    beta,
    binomial,
    categorical,
    cauchy,
    chi2,
    dirac,
    dirichlet,
    expon,
    gamma,
    geometric,
    # Higher-order distributions
    indep,
    laplace,
    mixture,
    multivariate_normal,
    # Continuous distributions
    norm,
    pareto,
    poisson,
    t,
    transformed,
    truncnorm,
    uniform,
    vonmises,
)
from probjax.stats.constraint_registry import biject_to, transform_to
from probjax.stats.constraints import (
    simplex,
    strict_positive,
    symmetric_positive_definite_matrix,
)
from probjax.stats.divergences import (
    kl_divergence,
    max_slice_wasserstein_distance,
    sliced_wasserstein_distance,
    wasserstein_distance,
)
from probjax.stats.divergences.wasserstein import (
    _1d_wasserstein_without_cdf,
    __max_slice_wasserstein_generic,
    __sliced_wasserstein_generic,
)

CONTINUOUS_DIST = [
    norm,
    gamma,
    beta,
    expon,
    laplace,
    uniform,
    chi2,
    t,
    cauchy,
    dirichlet,
    multivariate_normal,
]
DISCRETE_DIST = [bernoulli, binomial, categorical, poisson, geometric, dirac]
SPECIAL_DIST = [indep, transformed, mixture]

FIT_TEST_CASES = [
    {
        "seed": 0,
        "dist": norm,
        "params": {"loc": jnp.array(0.5), "scale": jnp.array(1.3)},
        "param_order": ["loc", "scale"],
        "num_samples": 8000,
        "rtol": 0.1,
        "atol": 0.05,
    },
    {
        "seed": 1,
        "dist": expon,
        "params": {"rate": jnp.array(2.5)},
        "param_order": ["rate"],
        "num_samples": 8000,
        "rtol": 0.1,
        "atol": 0.05,
    },
    {
        "seed": 2,
        "dist": gamma,
        "params": {"alpha": jnp.array(3.0), "beta": jnp.array(1.5)},
        "param_order": ["alpha", "beta"],
        "num_samples": 20000,
        "rtol": {"alpha": 0.15, "beta": 0.15},
        "atol": {"alpha": 0.2, "beta": 0.1},
    },
    {
        "seed": 3,
        "dist": cauchy,
        "params": {"loc": jnp.array(0.2), "scale": jnp.array(0.8)},
        "param_order": ["loc", "scale"],
        "num_samples": 50000,
        "rtol": {"loc": 0.2, "scale": 0.6},
        "atol": {"loc": 0.2, "scale": 0.5},
    },
    {
        "seed": 4,
        "dist": dirichlet,
        "params": {"alpha": jnp.array([2.0, 4.0, 3.0])},
        "param_order": ["alpha"],
        "num_samples": 10000,
        "rtol": {"alpha": 0.2},
        "atol": {"alpha": 0.2},
    },
    {
        "seed": 5,
        "dist": multivariate_normal,
        "params": {
            "loc": jnp.array([0.3, -0.7]),
            "cov": jnp.array([[1.2, 0.4], [0.4, 1.5]]),
        },
        "param_order": ["loc", "cov"],
        "num_samples": 12000,
        "rtol": {"loc": 0.1, "cov": 0.2},
        "atol": {"loc": 0.1, "cov": 0.2},
    },
    {
        "seed": 6,
        "dist": bernoulli,
        "params": {"p": jnp.array(0.35)},
        "param_order": ["p"],
        "num_samples": 4000,
        "rtol": 0.05,
        "atol": 0.02,
    },
    {
        "seed": 7,
        "dist": binomial,
        "params": {"n": 10, "probs": jnp.array(0.45)},
        "param_order": ["n", "probs"],
        "num_samples": 5000,
        "rtol": {"n": 0.0, "probs": 0.05},
        "atol": {"n": 0.0, "probs": 0.02},
        "fit_kwargs": lambda case: {"n": case["params"]["n"]},
    },
    {
        "seed": 8,
        "dist": poisson,
        "params": {"rate": jnp.array(4.5)},
        "param_order": ["rate"],
        "num_samples": 6000,
        "rtol": 0.05,
        "atol": 0.05,
    },
    {
        "seed": 9,
        "dist": geometric,
        "params": {"p": jnp.array(0.3)},
        "param_order": ["p"],
        "num_samples": 8000,
        "rtol": 0.05,
        "atol": 0.02,
        "sample_fn": lambda key, params, num: (
            jax.random.geometric(key, params["p"], shape=(num,)) - 1
        ),
    },
    {
        "seed": 10,
        "dist": categorical,
        "params": {"probs": jnp.array([0.1, 0.3, 0.2, 0.4])},
        "param_order": ["probs"],
        "num_samples": 15000,
        "rtol": {"probs": 0.1},
        "atol": {"probs": 0.05},
        "fit_kwargs": lambda case: {"num_classes": case["params"]["probs"].shape[-1]},
        "sample_fn": lambda key, params, num: jax.random.categorical(
            key, jnp.log(params["probs"]), shape=(num,)
        ),
    },
]


# Helper functions
def sample_and_check_shape(dist, key, sample_shape, *args, **kwargs):
    """Check if sampling and shape handling works correctly."""
    sample = dist.rvs(key, *args, shape=sample_shape, **kwargs)
    assert sample.shape == sample_shape + dist.batch_shape + dist.event_shape, (
        "Sample shape mismatch"
    )
    return sample


def check_mean_and_var(dist, key, *args, **kwargs):
    """Check if mean and variance are computed correctly."""
    sample = dist.rvs(key, *args, shape=(50000,), **kwargs)
    mean = jnp.mean(sample, axis=0)
    var = jnp.var(sample, axis=0)
    std = jnp.sqrt(var)

    assert mean.shape == dist.batch_shape + dist.event_shape, "Mean shape mismatch"
    assert var.shape == dist.batch_shape + dist.event_shape, "Variance shape mismatch"

    try:
        true_mean = dist.mean(*args, **kwargs)
        true_var = dist.var(*args, **kwargs)
        true_std = dist.std(*args, **kwargs)

        finite_true_mean = jnp.isfinite(true_mean)
        mean_to_compare = jnp.where(finite_true_mean, mean, true_mean)

        finite_true_var = jnp.isfinite(true_var)
        var_to_compare = jnp.where(finite_true_var, var, true_var)

        finite_true_std = jnp.isfinite(true_std)
        std_to_compare = jnp.where(finite_true_std, std, true_std)

        assert jnp.allclose(
            true_mean[finite_true_mean],
            mean_to_compare[finite_true_mean],
            atol=0.2,
            rtol=1.0,
        ), "Mean is not close to sample mean"

        assert jnp.allclose(
            true_var[finite_true_var],
            var_to_compare[finite_true_var],
            atol=0.2,
            rtol=1.0,
        ), "Variance is not close to sample variance"

        assert jnp.allclose(
            true_std[finite_true_std],
            std_to_compare[finite_true_std],
            atol=0.3,
            rtol=1.0,
        ), "Standard deviation is not close to sample standard deviation"

    except NotImplementedError:
        pass


def check_cdf_icdf(dist, key, *args, **kwargs):
    """Check if CDF and ICDF (PPF) are computed correctly."""
    sample = dist.rvs(key, *args, shape=(10000,), **kwargs)
    eval_points = dist.rvs(key, *args, shape=(10,), **kwargs)

    empirical_cdf = jnp.mean(sample[:, None] <= eval_points[None, :], axis=0)

    try:
        cdf = dist.cdf(eval_points, *args, **kwargs)
        assert cdf.shape == eval_points.shape, "CDF shape mismatch"
        assert jnp.isfinite(cdf).all(), "CDF is not finite for all samples"
        assert jnp.allclose(empirical_cdf, cdf, atol=0.1, rtol=0.5), (
            "CDF is not close to empirical cdf"
        )

        try:
            icdf = dist.ppf(cdf, *args, **kwargs)
            assert icdf.shape == eval_points.shape, "ICDF shape mismatch"
            assert jnp.isfinite(icdf).all(), "ICDF is not finite for all samples"
            assert jnp.allclose(eval_points, icdf, atol=0.1, rtol=0.5), (
                "ICDF is not close to sample"
            )
        except NotImplementedError:
            pass
    except NotImplementedError:
        pass


def check_mode(dist, key, *args, **kwargs):
    """Check if mode is computed correctly."""
    sample = dist.rvs(key, *args, shape=(1000,), **kwargs)
    try:
        log_prob = dist.logpdf(sample, *args, **kwargs)
        mode = sample[jnp.argmax(log_prob)]
        mode_log_prob = dist.logpdf(mode, *args, **kwargs)
        est_mode_log_prob = dist.logpdf(dist.mode(*args, **kwargs), *args, **kwargs)

        assert mode.shape == dist.batch_shape + dist.event_shape, "Mode shape mismatch"
        assert jnp.isfinite(mode).all(), "Mode is not finite"

        assert jnp.all(mode_log_prob <= est_mode_log_prob) | jnp.allclose(
            dist.mode(*args, **kwargs), mode, atol=0.1, rtol=0.1
        ), "Mode is not close to sample mode or has a higher log_prob"
        assert mode_log_prob <= dist.logpdf(
            dist.mode(*args, **kwargs), *args, **kwargs
        ), "Mode log_prob is not maximum"
    except NotImplementedError:
        pass


def _get_tol(case, tol_key, name, default):
    value = case.get(tol_key)
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return value


def _case_id(case):
    dist = case["dist"]
    return getattr(dist, "name", dist.__class__.__name__)


def init_dist(dist, key, shape=(1,)):
    """Initialize a distribution with random parameters."""
    if dist == uniform:
        x = jax.random.normal(key, shape)
        return dist(x, x + 1)
    elif dist == multivariate_normal:
        key, subkey = jax.random.split(key)
        loc = jax.random.normal(subkey, shape)
        key, subkey = jax.random.split(key)
        cov = jnp.atleast_2d(jax.random.normal(subkey, shape))
        cov = jnp.einsum('...ij,...kj->...ik', cov, cov)
        return dist(loc, cov)
    elif hasattr(dist, 'parameters'):
        keys = jax.random.split(key, len(dist.parameters))
        kwargs = {}
        for (name, constraint), key in zip(dist.parameters.items(), keys, strict=False):
            transform = transform_to(constraint)
            kwargs[name] = transform(jax.random.normal(key, shape))
        return dist(**kwargs)
    return dist


@pytest.mark.parametrize("case", FIT_TEST_CASES, ids=_case_id)
def test_distribution_fit_estimators(case):
    """Ensure distribution-specific fit routines recover parameters from samples."""
    dist = case["dist"]
    params = {k: v for k, v in case["params"].items()}
    seed = case.get("seed", 0)
    key = jax.random.PRNGKey(seed)
    num_samples = case.get("num_samples", 5000)
    sample_shape = (num_samples,)
    sample_fn = case.get("sample_fn")
    if sample_fn is not None:
        data = sample_fn(key, params, num_samples)
    else:
        data = dist.rvs(key, **params, shape=sample_shape)

    fit_kwargs = case.get("fit_kwargs")
    if callable(fit_kwargs):
        fit_kwargs = fit_kwargs(case)
    fit_kwargs = fit_kwargs or {}

    fitted = dist.fit(data, **fit_kwargs)
    if not isinstance(fitted, tuple):
        fitted = (fitted,)

    for idx, name in enumerate(case["param_order"]):
        expected = params[name]
        fitted_val = fitted[idx]
        expected_arr = jnp.asarray(expected)
        fitted_arr = jnp.asarray(fitted_val)
        assert expected_arr.shape == fitted_arr.shape, (
            f"{_case_id(case)} parameter '{name}' shape mismatch: "
            f"{fitted_arr.shape} vs {expected_arr.shape}"
        )
        atol = _get_tol(case, "atol", name, 0.1)
        rtol = _get_tol(case, "rtol", name, 0.1)
        assert jnp.allclose(fitted_arr, expected_arr, atol=atol, rtol=rtol), (
            f"{_case_id(case)} fit mismatch for parameter '{name}'"
        )


@pytest.mark.parametrize(
    "dist",
    CONTINUOUS_DIST + DISCRETE_DIST + SPECIAL_DIST,
    ids=lambda x: getattr(x, 'name', x.__class__.__name__),
)
def test_distribution_class_attributes(dist):
    """Test that all distributions have required class attributes."""
    assert hasattr(dist, 'parameters') and isinstance(dist.parameters, dict), (
        "Missing parameters"
    )
    assert hasattr(dist, 'support') and callable(dist.support), "Missing support"
    assert hasattr(dist, 'rvs') and callable(dist.rvs), "Missing rvs"


def test_frozen_init_rejects_invalid_arguments():
    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        norm(foo=1.0)

    with pytest.raises(TypeError, match="expected at most 2 positional arguments"):
        norm(0.0, 1.0, 2.0)

    with pytest.raises(TypeError, match="multiple values for argument 'loc'"):
        norm(0.0, loc=1.0)


def test_constraint_bijections_round_trip():
    positive_value = jnp.array(2.5)
    simplex_value = jnp.array([0.2, 0.3, 0.5])
    covariance = jnp.array([[2.0, 0.3], [0.3, 1.0]])

    for constraint, value in (
        (strict_positive, positive_value),
        (simplex, simplex_value),
        (symmetric_positive_definite_matrix, covariance),
    ):
        transform = biject_to(constraint)
        assert jnp.allclose(transform(transform.inv(value)), value, atol=1e-6)


def test_frozen_parameter_round_trip():
    dist = norm(loc=jnp.array(1.5), scale=jnp.array(2.0))

    assert set(dist.params) == {"loc", "scale"}
    assert jnp.allclose(dist.unconstrained_params["scale"], jnp.log(2.0))

    rebuilt = norm.from_params(
        norm.params_from_unconstrained(dist.unconstrained_params)
    )
    assert jnp.allclose(rebuilt.loc, dist.loc)
    assert jnp.allclose(rebuilt.scale, dist.scale)


def test_nested_mixture_parameter_round_trip():
    dist = mixture(
        jnp.array([0.25, 0.75]),
        [norm(-1.0, 0.5), norm(2.0, 1.5)],
    )

    unconstrained = dist.unconstrained_params
    flat, unravel = ravel_pytree(unconstrained)
    rebuilt = mixture.from_params(mixture.params_from_unconstrained(unravel(flat)))

    assert flat.shape == (6,)
    assert jnp.allclose(rebuilt.mixing_probs, dist.mixing_probs)
    assert jnp.allclose(rebuilt.components[0].scale, dist.components[0].scale)


def test_fit_params_names_analytic_fit_results():
    data = jnp.array([-2.0, 0.0, 1.0, 3.0])
    loc, scale = norm.fit(data)

    fitted = norm.fit_params(data)

    assert set(fitted) == {"loc", "scale"}
    assert jnp.allclose(fitted["loc"], loc)
    assert jnp.allclose(fitted["scale"], scale)


def test_frozen_init_rejects_invalid_special_distribution_keywords():
    base_dist = norm(0.0, 1.0)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        categorical(jnp.array([1.0]), foo=True)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        indep(base_dist, foo=True)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        mixture(jnp.array([1.0]), [base_dist], foo=True)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        transformed(base_dist, lambda x: x, foo=True)


def test_transformed_accepts_inverse_and_logdet_frozen_keyword():
    def inverse_and_logdet(y):
        return y, jnp.zeros_like(y)

    dist = transformed(
        norm(0.0, 1.0),
        lambda x: x,
        inverse_and_logdet=inverse_and_logdet,
    )

    assert dist.kwds["inverse_and_logdet"] is inverse_and_logdet


@pytest.mark.parametrize("dist", CONTINUOUS_DIST + DISCRETE_DIST, ids=lambda x: x.name)
def test_base_distribution(dist, shape=(1,), seed=0):
    """Test basic functionality of distributions."""
    key = jax.random.PRNGKey(seed)
    p = init_dist(dist, key, shape)

    # Test sampling and shape handling
    sample_and_check_shape(p, key, shape)

    # Test mean and variance
    check_mean_and_var(p, key)

    # Test mode
    check_mode(p, key)

    # Test CDF and ICDF
    check_cdf_icdf(p, key)

    # Test PyTree functionality
    flatten_p, tree_p = jax.tree_util.tree_flatten(p)
    q = jax.tree_util.tree_unflatten(tree_p, flatten_p)
    assert jnp.allclose(p.rvs(key, shape=shape), q.rvs(key, shape=shape)), (
        "PyTree reconstruction mismatch"
    )


@pytest.mark.parametrize("dist", CONTINUOUS_DIST + DISCRETE_DIST, ids=lambda x: x.name)
def test_independent_distribution(dist, shape=(2,), seed=0):
    """Test independent distribution functionality."""
    # Skip over multivariate dists
    if dist == multivariate_normal or dist == dirichlet or dist == categorical:
        return

    key = jax.random.PRNGKey(seed)
    p = init_dist(dist, key, shape)

    p = indep(p)

    # Test sampling and shape handling
    sample_and_check_shape(p, key, shape)

    # Test mean and variance
    check_mean_and_var(p, key)

    # Test mode
    check_mode(p, key)

    # Test PyTree functionality
    flatten_p, tree_p = jax.tree_util.tree_flatten(p)
    q = jax.tree_util.tree_unflatten(tree_p, flatten_p)
    assert jnp.allclose(
        p.rvs(key, shape=shape), q.rvs(key, shape=shape), atol=0.01, rtol=0.01
    ), "PyTree reconstruction mismatch"


@pytest.mark.parametrize(
    "dist1, dist2",
    list(itertools.combinations(CONTINUOUS_DIST + DISCRETE_DIST, 2)),
    ids=lambda x: f"{x.name}",
)
def test_mixed_independent_distribution(dist1, dist2, shape=(1,), seed=0):
    """Test mixed independent distribution functionality."""
    if dist1 == multivariate_normal or dist1 == dirichlet or dist1 == categorical:
        return
    if dist2 == multivariate_normal or dist2 == dirichlet or dist2 == categorical:
        return

    key = jax.random.PRNGKey(seed)

    p1 = init_dist(dist1, key, shape)
    p2 = init_dist(dist2, key, shape)

    try:
        p = indep(p1, p2)
    except AssertionError:
        return

    sample_and_check_shape(p, key, shape)
    check_mean_and_var(p, key)
    check_mode(p, key)


@pytest.mark.parametrize("dist", CONTINUOUS_DIST + DISCRETE_DIST, ids=lambda x: x.name)
def test_mixture_distribution(dist, shape=(1,), seed=0):
    """Test mixture distribution functionality."""
    key = jax.random.PRNGKey(seed)
    p1 = init_dist(dist, jax.random.PRNGKey(seed + 42), shape=shape)
    p2 = init_dist(dist, jax.random.PRNGKey(seed + 420000), shape=shape)

    p = mixture(jnp.array([0.5, 0.5]), [p1, p2])

    sample_and_check_shape(p, key, shape)
    check_mean_and_var(p, key)
    check_mode(p, key)


def test_mixture_em_gaussian(seed: int = 0):
    """Ensure the EM routine recovers a simple Gaussian mixture."""
    key = jax.random.PRNGKey(seed)
    true_probs = jnp.array([0.35, 0.65])
    comp1 = norm(-1.0, 0.6)
    comp2 = norm(2.0, 0.8)
    mix = mixture(true_probs, [comp1, comp2])

    data = mix.rvs(key, shape=(5000,))

    init_components = [norm(-2.0, 1.5), norm(1.0, 1.2)]
    rng = jax.random.PRNGKey(seed + 123)
    est_probs, est_components = mixture.fit(
        data,
        init_components,
        max_iter=150,
        tol=1e-5,
        rng_key=rng,
    )

    est_means = jnp.array([comp.mean() for comp in est_components])
    est_scales = jnp.array([comp.std() for comp in est_components])
    true_means = jnp.array([comp1.mean(), comp2.mean()])
    true_scales = jnp.array([comp1.std(), comp2.std()])

    order_est = jnp.argsort(est_means)
    order_true = jnp.argsort(true_means)

    est_probs = est_probs[order_est]
    est_means = est_means[order_est]
    est_scales = est_scales[order_est]

    true_probs_sorted = true_probs[order_true]
    true_means = true_means[order_true]
    true_scales = true_scales[order_true]

    assert jnp.allclose(est_probs, true_probs_sorted, atol=0.12)
    assert jnp.allclose(est_means, true_means, atol=0.2)
    assert jnp.allclose(est_scales, true_scales, atol=0.2)


def _angle_distance(a, b):
    return jnp.abs(jnp.arctan2(jnp.sin(a - b), jnp.cos(a - b)))


def test_mixture_em_vonmises(seed: int = 0):
    """EM should recover parameters of a von Mises mixture."""
    import numpy as np
    from scipy.stats import vonmises as scipy_vonmises

    rng_np = np.random.default_rng(seed)
    true_probs = jnp.array([0.45, 0.55])
    loc1, kappa1 = -1.8, 4.0
    loc2, kappa2 = 1.8, 5.0
    comp1 = vonmises(loc1, kappa1)
    comp2 = vonmises(loc2, kappa2)

    n = 6000
    labels = rng_np.choice(2, size=n, p=np.array(true_probs))
    samples = np.where(
        labels == 0,
        scipy_vonmises.rvs(kappa1, loc=loc1, size=n, random_state=rng_np),
        scipy_vonmises.rvs(kappa2, loc=loc2, size=n, random_state=rng_np),
    )
    data = jnp.array(samples)

    init_components = [vonmises(-1.0, 2.5), vonmises(2.3, 2.5)]
    rng = jax.random.PRNGKey(seed + 456)
    est_probs, est_components = mixture.fit(
        data,
        init_components,
        mixing_probs_init=true_probs,
        max_iter=300,
        tol=1e-5,
        rng_key=rng,
    )

    est_locs = jnp.array([comp.args[0] for comp in est_components])
    est_kappas = jnp.array([comp.args[1] for comp in est_components])
    true_locs = jnp.array([comp1.args[0], comp2.args[0]])
    true_kappas = jnp.array([comp1.args[1], comp2.args[1]])

    order_est = jnp.argsort(est_locs)
    order_true = jnp.argsort(true_locs)

    est_probs = est_probs[order_est]
    est_locs = est_locs[order_est]
    est_kappas = est_kappas[order_est]

    true_probs_sorted = true_probs[order_true]
    true_locs = true_locs[order_true]
    true_kappas = true_kappas[order_true]

    baseline_ll = jnp.mean(mixture.logpdf(data, true_probs, init_components))
    final_ll = jnp.mean(mixture.logpdf(data, est_probs, list(est_components)))

    assert final_ll > baseline_ll
    assert jnp.all(est_probs > 0.05)
    assert jnp.all(_angle_distance(est_locs, true_locs) < 1.0)
    assert jnp.all(est_kappas > 0.5)


def test_mixture_em_multivariate_gaussian(seed: int = 0):
    """EM should recover a simple multivariate Gaussian mixture."""
    key = jax.random.PRNGKey(seed)
    true_probs = jnp.array([0.4, 0.6])
    loc1 = jnp.array([0.5, -0.2])
    cov1 = jnp.array([[0.8, 0.1], [0.1, 0.6]])
    loc2 = jnp.array([-1.0, 1.3])
    cov2 = jnp.array([[0.5, -0.2], [-0.2, 0.9]])
    comp1 = multivariate_normal(loc1, cov=cov1)
    comp2 = multivariate_normal(loc2, cov=cov2)
    mix = mixture(true_probs, [comp1, comp2])

    data = mix.rvs(key, shape=(4000,))

    init_components = [
        multivariate_normal(loc=jnp.zeros(2), cov=jnp.eye(2)),
        multivariate_normal(loc=jnp.array([1.5, -1.0]), cov=jnp.eye(2)),
    ]
    rng = jax.random.PRNGKey(seed + 321)
    est_probs, est_components = mixture.fit(
        data,
        init_components,
        max_iter=150,
        tol=1e-4,
        rng_key=rng,
    )

    est_means = jnp.stack([comp.args[0] for comp in est_components])
    est_covs = jnp.stack([comp.args[1] for comp in est_components])

    true_means = jnp.stack([loc1, loc2])
    true_covs = jnp.stack([cov1, cov2])

    order_est = jnp.argsort(est_means[:, 0])
    order_true = jnp.argsort(true_means[:, 0])

    est_probs = est_probs[order_est]
    est_means = est_means[order_est]
    est_covs = est_covs[order_est]

    true_probs_sorted = true_probs[order_true]
    true_means = true_means[order_true]
    true_covs = true_covs[order_true]

    assert jnp.allclose(est_probs, true_probs_sorted, atol=0.12)
    assert jnp.allclose(est_means, true_means, atol=0.35)
    assert jnp.allclose(est_covs, true_covs, atol=0.4)


def test_truncnorm_sampling_with_bounds(seed: int = 0):
    """Ensure truncated normal sampling respects bounds and remains finite."""
    key = jax.random.PRNGKey(seed)
    loc = jnp.array(0.5)
    scale = jnp.array(0.8)
    lower = jnp.array(-1.2)
    upper = jnp.array(1.7)
    dist = truncnorm(loc, scale, lower, upper)

    samples = dist.rvs(key, shape=(4000,))
    assert samples.shape == (4000,)
    assert jnp.all(jnp.isfinite(samples)), "Samples must be finite"
    assert jnp.all(samples >= lower - 1e-6)
    assert jnp.all(samples <= upper + 1e-6)


def test_pareto_sampling_matches_moment(seed: int = 0):
    """Pareto sampler should respect the minimum and produce the correct mean."""
    key = jax.random.PRNGKey(seed)
    scale = jnp.array(0.0)
    tail = jnp.array(2.5)
    dist = pareto(scale, tail)

    samples = dist.rvs(key, shape=(32000,))
    assert samples.shape == (32000,)
    assert jnp.all(samples >= scale), "Pareto samples must be >= scale parameter"

    expected_mean = dist.mean()
    if not jnp.isfinite(expected_mean):
        return
    empirical_mean = jnp.mean(samples)
    assert jnp.allclose(
        empirical_mean,
        expected_mean,
        atol=0.15 * float(expected_mean),
        rtol=0.15,
    )


def test_geometric_rvs_and_logpdf_alignment(seed: int = 0):
    """Geometric sampler and logpdf should align on support and probabilities."""
    key = jax.random.PRNGKey(seed)
    prob = jnp.array(0.37)
    dist = geometric(prob)

    samples = dist.rvs(key, shape=(10000,))
    assert samples.min() >= 0, "Geometric samples must be non-negative"
    assert samples.dtype == jnp.int32

    values = jnp.arange(5, dtype=prob.dtype)
    expected_logpdf = jnp.log(prob) + values * jnp.log1p(-prob)
    observed_logpdf = dist.logpdf(values)
    assert jnp.allclose(observed_logpdf, expected_logpdf, atol=1e-6, rtol=1e-6)

    assert dist.logpdf(jnp.array(-1.0)) == -jnp.inf


@pytest.mark.parametrize("dist", CONTINUOUS_DIST)
def test_transformed_distribution(dist, shape=(1,), seed=0):
    """Test transformed distribution functionality."""
    key = jax.random.PRNGKey(seed)
    p = init_dist(dist, key, shape=shape)

    # Test sampling and shape handling
    sample_and_check_shape(p, key, shape)

    # Test PyTree functionality
    flatten_p, tree_p = jax.tree_util.tree_flatten(p)
    q = jax.tree_util.tree_unflatten(tree_p, flatten_p)
    assert jnp.allclose(p.rvs(key, shape=shape), q.rvs(key, shape=shape)), (
        "PyTree reconstruction mismatch"
    )


@pytest.mark.parametrize(
    "dist1, dist2",
    itertools.combinations(CONTINUOUS_DIST + DISCRETE_DIST, 2),
    ids=lambda x: f"{x.name}",
)
def test_kl_divergence(dist1, dist2, shape=(1,), seed=0):
    """Test KL divergence computation."""
    key1 = jax.random.PRNGKey(seed)
    key2 = jax.random.PRNGKey(seed + 420000)
    p = init_dist(dist1, key1)
    q = init_dist(dist2, key2)

    try:
        dist = kl_divergence(p, q, mc_samples=10000, key=key1)
    except (AssertionError, ValueError, NotImplementedError):
        return

    # Monte Carlo estimation of KL divergence
    samples = p.rvs(key1, shape=(10000,))
    log_ratio = p.logpdf(samples) - q.logpdf(samples)
    dist_mc = jnp.mean(log_ratio)

    assert dist.shape == p.batch_shape, "KL divergence shape mismatch"
    assert jnp.allclose(dist, dist_mc, atol=0.1, rtol=0.1), (
        "MC KL divergence is not close to analytic KL divergence"
    )


@pytest.mark.parametrize(
    "dist1, dist2", itertools.combinations(CONTINUOUS_DIST + DISCRETE_DIST, 2)
)
def test_wasserstein_distance(dist1, dist2, shape=(1,), seed=0):
    """Test Wasserstein distance computation."""
    key1 = jax.random.PRNGKey(seed)
    key2 = jax.random.PRNGKey(seed + 420000)
    p = init_dist(dist1, key1)
    q = init_dist(dist2, key2)

    try:
        dist = wasserstein_distance(p, q, mc_samples=1000, key=key1)
    except (AssertionError, ValueError, NotImplementedError):
        return

    # Monte Carlo estimation of Wasserstein distance
    samples_p = p.rvs(key1, shape=(1000,))
    samples_q = q.rvs(key2, shape=(1000,))
    dist_mc = _1d_wasserstein_without_cdf(samples_p, samples_q)

    assert dist.shape == p.batch_shape, "Wasserstein distance shape mismatch"
    assert jnp.allclose(dist, dist_mc, atol=0.1, rtol=0.1), (
        "MC Wasserstein distance is not close to analytic Wasserstein distance"
    )


@pytest.mark.parametrize(
    "dist1, dist2", itertools.combinations(CONTINUOUS_DIST + DISCRETE_DIST, 2)
)
def test_sliced_wasserstein_distance(dist1, dist2, shape=(1,), seed=0):
    """Test Sliced Wasserstein distance computation."""
    key1 = jax.random.PRNGKey(seed)
    key2 = jax.random.PRNGKey(seed + 420000)
    p = init_dist(dist1, key1)
    q = init_dist(dist2, key2)

    try:
        dist = sliced_wasserstein_distance(p, q, mc_samples=1000, key=key1)
    except (AssertionError, ValueError, NotImplementedError):
        return

    # Monte Carlo estimation of Sliced Wasserstein distance
    samples_p = p.rvs(key1, shape=(1000,))
    samples_q = q.rvs(key2, shape=(1000,))
    dist_mc = __sliced_wasserstein_generic(
        samples_p, samples_q, num_slices=100, key=key1
    )

    assert dist.shape == p.batch_shape, "Sliced Wasserstein distance shape mismatch"
    assert jnp.allclose(dist, dist_mc, atol=0.1, rtol=0.1), (
        "MC Sliced Wasserstein distance is not close to analytic Sliced Wasserstein distance"
    )


@pytest.mark.parametrize(
    "dist1, dist2", itertools.combinations(CONTINUOUS_DIST + DISCRETE_DIST, 2)
)
def test_max_slice_wasserstein_distance(dist1, dist2, shape=(1,), seed=0):
    """Test Max Sliced Wasserstein distance computation."""
    key1 = jax.random.PRNGKey(seed)
    key2 = jax.random.PRNGKey(seed + 420000)
    p = init_dist(dist1, key1)
    q = init_dist(dist2, key2)

    try:
        dist = max_slice_wasserstein_distance(p, q, mc_samples=1000, key=key1)
    except (AssertionError, ValueError, NotImplementedError):
        return

    # Monte Carlo estimation of Max Sliced Wasserstein distance
    samples_p = p.rvs(key1, shape=(1000,))
    samples_q = q.rvs(key2, shape=(1000,))
    dist_mc = __max_slice_wasserstein_generic(samples_p, samples_q, key=key1)

    assert dist.shape == p.batch_shape, "Max Sliced Wasserstein distance shape mismatch"
    assert jnp.allclose(dist, dist_mc, atol=0.1, rtol=0.1), (
        "MC Max Sliced Wasserstein distance is not close to analytic Max Sliced Wasserstein distance"
    )


# =============================================================================
# Directional distribution tests (Watson, Bingham)
# =============================================================================

from probjax.stats.continuous import bingham, watson


def _unit_vector(x):
    return x / jnp.linalg.norm(x)


def test_watson_samples_live_on_sphere():
    key = jax.random.PRNGKey(0)
    mean_direction = _unit_vector(jnp.array([0.3, -0.4, 1.2]))
    samples = watson.rvs(key, mean_direction=mean_direction, kappa=2.5, shape=(128,))
    norms = jnp.linalg.norm(samples, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-6)


def test_watson_logpdf_symmetry_and_uniform_limit():
    mean_direction = _unit_vector(jnp.array([0.0, 0.0, 1.0]))
    x = _unit_vector(jnp.array([1.0, 0.5, -0.25]))
    logpdf_pos = watson.logpdf(x, mean_direction, kappa=3.0)
    logpdf_neg = watson.logpdf(-x, mean_direction, kappa=3.0)
    assert jnp.allclose(logpdf_pos, logpdf_neg, atol=1e-6)

    logpdf_uniform = watson.logpdf(x, mean_direction, kappa=0.0)
    logpdf_uniform_ref = watson.logpdf(
        _unit_vector(jnp.array([0.1, -0.3, 0.95])), mean_direction, kappa=0.0
    )
    assert jnp.allclose(logpdf_uniform, logpdf_uniform_ref, atol=1e-6)


def test_watson_natural_parameter_shape():
    mean_direction = _unit_vector(jnp.array([0.1, 0.3, 0.9]))
    kappa = jnp.array([2.0, 5.0])  # batched concentration
    params = watson.natural_parameters(
        mean_direction=jnp.broadcast_to(mean_direction, (2, 3)),
        kappa=kappa,
    )
    assert params.shape == (2, 3)


def test_bingham_samples_live_on_sphere():
    key = jax.random.PRNGKey(1)
    orientation = jnp.eye(3)
    concentration = jnp.array([-3.0, 0.5, 2.0])
    samples = bingham.rvs(
        key, orientation=orientation, concentration=concentration, shape=(64,)
    )
    norms = jnp.linalg.norm(samples, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-6)


def test_bingham_logpdf_symmetry_and_uniform_limit():
    orientation = jnp.eye(3)
    concentration = jnp.array([-2.0, 0.0, 1.0])
    x = _unit_vector(jnp.array([0.2, -0.7, 0.65]))
    logpdf_pos = bingham.logpdf(x, orientation, concentration)
    logpdf_neg = bingham.logpdf(-x, orientation, concentration)
    assert jnp.allclose(logpdf_pos, logpdf_neg, atol=1e-6)

    logpdf_uniform = bingham.logpdf(x, orientation, jnp.zeros_like(concentration))
    logpdf_uniform_ref = bingham.logpdf(
        _unit_vector(jnp.array([-0.5, 0.3, 0.81])),
        orientation,
        jnp.zeros_like(concentration),
    )
    assert jnp.allclose(logpdf_uniform, logpdf_uniform_ref, atol=1e-6)


def test_bingham_natural_parameter_shape():
    orientation = jnp.stack([jnp.eye(3), jnp.eye(3)], axis=0)
    concentration = jnp.stack(
        [jnp.array([-2.0, 0.0, 1.0]), jnp.array([0.5, -0.3, -0.2])],
        axis=0,
    )
    params = bingham.natural_parameters(
        orientation=orientation, concentration=concentration
    )
    assert params.shape == (2, 3, 3)
    assert jnp.allclose(params, jnp.swapaxes(params, -1, -2))


def test_watson_fit_estimates_direction():
    key = jax.random.PRNGKey(2)
    true_mu = _unit_vector(jnp.array([0.1, -0.4, 1.0]))
    samples = watson.rvs(key, mean_direction=true_mu, kappa=5.0, shape=(512,))
    mu_hat, kappa_hat = watson.fit(samples)
    alignment = jnp.abs(jnp.dot(mu_hat, true_mu))
    assert alignment > 0.95
    assert kappa_hat > 0


def test_bingham_fit_estimates_orientation():
    key = jax.random.PRNGKey(3)
    orientation = jnp.eye(3)
    concentration = jnp.array([-3.0, -1.0, 0.0])
    samples = bingham.rvs(
        key, orientation=orientation, concentration=concentration, shape=(512,)
    )
    orientation_hat, concentration_hat = bingham.fit(samples)
    overlap = jnp.abs(orientation_hat.T @ orientation)
    assert jnp.all(jnp.max(overlap, axis=1) > 0.8)
    assert concentration_hat.shape == concentration.shape
