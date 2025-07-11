import itertools

import jax
import jax.numpy as jnp
import pytest

from probjax.stats import (
    rv_generic,
    rv_continuous,
    rv_discrete,
    rv_continuous_frozen,
    rv_discrete_frozen,
    # Continuous distributions
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
    # Discrete distributions
    bernoulli,
    binomial,
    categorical,
    poisson,
    geometric,
    dirac,
    empirical,
    # Higher-order distributions
    independent,
    transformed,
    mixture,
)

from probjax.stats.constraint_registry import transform_to
from probjax.stats.divergences import (
    kl_divergence,
    wasserstein_distance,
    sliced_wasserstein_distance,
    max_slice_wasserstein_distance,
)
from probjax.stats.divergences.wasserstein import (
    _1d_wasserstein_without_cdf,
    __sliced_wasserstein_generic,
    __max_slice_wasserstein_generic,
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
SPECIAL_DIST = [independent, transformed, mixture]

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
        for (name, constraint), key in zip(dist.parameters.items(), keys):
            transform = transform_to(constraint)
            kwargs[name] = transform(jax.random.normal(key, shape))
        return dist(**kwargs)
    return dist

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

    p = independent(p)

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
        p = independent(p1, p2)
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
    if isinstance(p, rv_discrete):
        log_ratio = p.logpmf(samples) - q.logpmf(samples)
    else:
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
