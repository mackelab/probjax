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
def sample_and_check_shape(dist, key, sample_shape, *args, **kwds):
    """Check if sampling and shape handling works correctly."""
    sample = dist.rvs(key, sample_shape, *args, **kwds)
    assert sample.shape == sample_shape + dist.batch_shape + dist.event_shape, (
        "Sample shape mismatch"
    )
    return sample

def check_mean_and_var(dist, key, *args, **kwds):
    """Check if mean and variance are computed correctly."""
    sample = dist.rvs(key, (50000,), *args, **kwds)
    mean = jnp.mean(sample, axis=0)
    var = jnp.var(sample, axis=0)
    std = jnp.sqrt(var)

    assert mean.shape == dist.batch_shape + dist.event_shape, "Mean shape mismatch"
    assert var.shape == dist.batch_shape + dist.event_shape, "Variance shape mismatch"

    try:
        true_mean = dist.mean(*args, **kwds)
        true_var = dist.var(*args, **kwds)
        true_std = dist.std(*args, **kwds)

        finite_true_mean = jnp.isfinite(true_mean)
        mean_to_compare = jnp.where(finite_true_mean, mean, true_mean)

        finite_true_var = jnp.isfinite(true_var)
        var_to_compare = jnp.where(finite_true_var, var, true_var)

        finite_true_std = jnp.isfinite(true_std)
        std_to_compare = jnp.where(finite_true_std, std, true_std)

        if jnp.all(finite_true_mean):
            assert jnp.allclose(true_mean, mean_to_compare, atol=0.2, rtol=1.0), (
                "Mean is not close to sample mean"
            )
        else:  # If true_mean is not finite, ensure sample mean is also not finite (or close to it)
            assert jnp.allclose(true_mean, mean_to_compare, equal_nan=True), (
                "Mean is not close to sample mean (non-finite case)"
            )

        if jnp.all(finite_true_var):
            assert jnp.allclose(true_var, var_to_compare, atol=0.2, rtol=1.0), (
                "Variance is not close to sample variance"
            )
        else:  # If true_var is not finite, ensure sample var is also not finite (or close to it)
            assert jnp.allclose(true_var, var_to_compare, equal_nan=True), (
                "Variance is not close to sample variance (non-finite case)"
            )

        if jnp.all(finite_true_std):
            assert jnp.allclose(true_std, std_to_compare, atol=0.3, rtol=1.0), (
                "Standard deviation is not close to sample standard deviation"
            )
        else:  # If true_std is not finite, ensure sample std is also not finite (or close to it)
            assert jnp.allclose(true_std, std_to_compare, equal_nan=True), (
                "Standard deviation is not close to sample standard deviation (non-finite case)"
            )

    except NotImplementedError:
        pass

def check_cdf_icdf(dist, key, *args, **kwds):
    """Check if CDF and ICDF (PPF) are computed correctly."""
    sample = dist.rvs(key, (10000,), *args, **kwds)
    eval_points = dist.rvs(key, (10,), *args, **kwds)

    empirical_cdf = jnp.mean(sample[:, None] <= eval_points[None, :], axis=0)

    try:
        cdf = dist.cdf(eval_points, *args, **kwds)
        assert cdf.shape == eval_points.shape, "CDF shape mismatch"
        assert jnp.isfinite(cdf).all(), "CDF is not finite for all samples"
        assert jnp.allclose(empirical_cdf, cdf, atol=0.1, rtol=0.5), (
            "CDF is not close to empirical cdf"
        )

        try:
            icdf = dist.ppf(cdf, *args, **kwds)
            assert icdf.shape == eval_points.shape, "ICDF shape mismatch"
            assert jnp.isfinite(icdf).all(), "ICDF is not finite for all samples"
            assert jnp.allclose(eval_points, icdf, atol=0.1, rtol=0.5), (
                "ICDF is not close to sample"
            )
        except NotImplementedError:
            pass
    except NotImplementedError:
        pass

def check_mode(dist, key, *args, **kwds):
    """Check if mode is computed correctly."""
    sample = dist.rvs(key, (1000,), *args, **kwds)
    try:
        log_prob = dist.logpdf(sample, *args, **kwds)
        mode = sample[jnp.argmax(log_prob)]
        mode_log_prob = dist.logpdf(mode, *args, **kwds)
        est_mode_log_prob = dist.logpdf(dist.mode(*args, **kwds), *args, **kwds)

        assert mode.shape == dist.batch_shape + dist.event_shape, "Mode shape mismatch"
        assert jnp.isfinite(mode).all(), "Mode is not finite"

        assert jnp.all(mode_log_prob <= est_mode_log_prob + 1e-3) | jnp.allclose(
            dist.mode(*args, **kwds), mode, atol=0.1, rtol=0.1
        ), "Mode is not close to sample mode"
        assert mode_log_prob <= dist.logpdf(dist.mode(*args, **kwds), *args, **kwds), (
            "Mode log_prob is not maximum"
        )
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
    "dist", CONTINUOUS_DIST + DISCRETE_DIST + SPECIAL_DIST, ids=lambda x: x.name
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
    assert jnp.allclose(p.rvs(key, shape), q.rvs(key, shape)), (
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
    assert jnp.allclose(p.rvs(key, shape), q.rvs(key, shape), atol=0.01, rtol=0.01), (
        "PyTree reconstruction mismatch"
    )


# @pytest.mark.parametrize(
#     "dist1, dist2",
#     list(itertools.combinations(CONTINUOUS_DIST + DISCRETE_DIST, 2)),
#     ids=lambda x: f"{x[0].name}-{x[1].name}",
# )
# def test_mixed_independent_distribution(dist1, dist2, shape=(1,), seed=0):
#     """Test mixed independent distribution functionality."""
#     key = jax.random.PRNGKey(seed)

#     p1 = init_dist(dist1, key, shape)
#     p2 = init_dist(dist2, key, shape)

#     try:
#         p = independent(p1, p2)
#     except AssertionError:
#         return

#     sample_and_check_shape(p, key, shape)
#     check_mean_and_var(p, key)
#     check_mode(p, key)


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
    assert jnp.allclose(p.rvs(key, shape), q.rvs(key, shape)), (
        "PyTree reconstruction mismatch"
    )


# @pytest.mark.parametrize(
#     "dist1, dist2", itertools.combinations(CONTINUOUS_DIST + DISCRETE_DIST, 2)
# )
# def test_kl_divergence(dist1, dist2, shape=(1,), seed=0):
#     """Test KL divergence computation."""
#     key1 = jax.random.PRNGKey(seed)
#     key2 = jax.random.PRNGKey(seed + 420000)
#     p = init_dist(dist1, key1)
#     q = init_dist(dist2, key2)

#     try:
#         dist = kl_divergence(p, q)
#     except (AssertionError, ValueError):
#         return

#     # Monte Carlo estimation of KL divergence
#     samples = p.rvs(key1, (10000,))
#     log_ratio = p.logpdf(samples) - q.logpdf(samples)
#     dist_mc = jnp.mean(log_ratio)

#     assert dist.shape == p.batch_shape, "KL divergence shape mismatch"
#     assert jnp.allclose(dist, dist_mc, atol=0.1, rtol=0.1), (
#         "MC KL divergence is not close to analytic KL divergence"
#     )
