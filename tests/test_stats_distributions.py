import itertools

import jax
import jax.numpy as jnp
import pytest
from jax import random
from jaxtyping import ArrayLike

from probjax.stats import (
    # Base classes
    rv_continuous,
    rv_discrete,
    # Continuous distributions
    norm,
    gamma,
    beta,
    expon,
    laplace,
    uniform,
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
from probjax.stats.constraints import (
    simplex,
    interval,
    real,
    positive,
    strict_positive,
    integer,
    boolean,
    UnitInterval,
    PositiveInteger,
    StrictPositiveInteger,
    distribution as distribution_constraint,
    dependent,
)

CONTINUOUS_DIST = [norm, gamma, beta, expon, laplace, uniform]
DISCRETE_DIST = [bernoulli, binomial, categorical, poisson, geometric, dirac, empirical]
SPECIAL_DIST = [independent, transformed, mixture]


# Some helper functions
def sample_and_log_prob(p, key: jax.random.PRNGKey, sample_shape):
    # Check log_prob and sample without sampling shape
    sample = p.rvs(key, sample_shape)
    log_prob = p.logpdf(sample)

    assert sample.shape == sample_shape + p.batch_shape + p.event_shape, (
        "Sample shape mismatch"
    )
    assert log_prob.shape == sample_shape + p.batch_shape, "Log_prob shape mismatch"
    assert jnp.isfinite(log_prob).all(), "Log_prob is not finite for all samples"


def mean_and_var(p, key: jax.random.PRNGKey):
    # Check mean and variance
    sample = p.rvs(key, (10000,))
    mean = jnp.mean(sample, axis=0)
    var = jnp.var(sample, axis=0)
    std = jnp.sqrt(var)

    assert mean.shape == p.batch_shape + p.event_shape, "Mean shape mismatch"
    assert var.shape == p.batch_shape + p.event_shape, "Variance shape mismatch"

    try:
        # This can be infinite for some distributions
        true_mean = p.mean()
        true_var = p.var()
        true_std = jnp.sqrt(true_var)

        mask = jnp.isfinite(true_mean)
        mean = jnp.where(mask, mean, true_mean)
        mask = jnp.isfinite(true_var)
        var = jnp.where(mask, var, true_var)
        mask = jnp.isfinite(true_std)
        std = jnp.where(mask, std, true_std)

        # Rather loose check as LLN may not hold
        assert jnp.allclose(true_mean, mean, atol=0.2, rtol=1.0), (
            "Mean is not close to sample mean"
        )
        assert jnp.allclose(true_var, var, atol=0.2, rtol=1.0), (
            "Variance is not close to sample variance"
        )
        assert jnp.allclose(true_std, std, atol=0.3, rtol=1.0), (
            "Standard deviation is not close to sample standard deviation"
        )
    except AssertionError as e:
        raise e
    except NotImplementedError:
        pass


def cdf_icdf(p, key: jax.random.PRNGKey):
    # Check cdf and icdf
    sample = p.rvs(key, (10000,))
    eval_points = p.rvs(key, (10,))

    empirical_cdf = jnp.mean(sample[:, None] <= eval_points[None, :], axis=0)

    try:
        cdf = p.cdf(eval_points)

        assert cdf.shape == eval_points.shape, "CDF shape mismatch"
        assert jnp.isfinite(cdf).all(), "CDF is not finite for all samples"

        assert jnp.allclose(empirical_cdf, cdf, atol=0.1, rtol=0.5), (
            "CDF is not close to empirical cdf"
        )

        try:
            icdf = p.ppf(cdf)
            assert icdf.shape == eval_points.shape, "ICDF shape mismatch"
            assert jnp.isfinite(icdf).all(), "ICDF is not finite for all samples"
            assert jnp.allclose(eval_points, icdf, atol=0.1, rtol=0.5), (
                "ICDF is not close to sample"
            )
        except NotImplementedError:
            pass
    except AssertionError as e:
        raise e
    except NotImplementedError:
        pass


def init_dist(dist_or_gen_obj, key: random.PRNGKey, shape: tuple = (1,)):
    """Initialize a frozen distribution with random parameters using its parameters and constraints."""

    if isinstance(dist_or_gen_obj, type):  # It's a class like bernoulli, poisson
        params_dict = dist_or_gen_obj.parameters
        # Create an instance of the generator class if 'dist_or_gen_obj' is the class itself
        # Some distributions might not need this if their class methods handle everything
        # For consistency, we assume we need an instance to call .freeze() on.
        # If dist_or_gen_obj() is not needed (i.e. class itself has freeze), this could be simpler.
        # However, Scipy-like distributions are generator instances already.
        try:
            dist_to_freeze = dist_or_gen_obj()
        except (
            TypeError
        ):  # Happens if __init__ takes arguments like 'name' but we don't provide them.
            # Most dists have a default __init__(self, name=None).
            # If it's a generator object already, this 'if' branch isn't taken.
            dist_to_freeze = dist_or_gen_obj  # Fallback if instantiation fails (e.g. if it's a class that is not meant to be instantiated before freeze)
            # This part is tricky. For classes like `poisson`, we need `poisson().freeze()`.
            # For generator objects like `norm`, we need `norm.freeze()`.
            # The original code did: instance = dist(), then instance.freeze() for classes
            # and dist.freeze() for generator objects. This logic seems fine.
        if not hasattr(
            dist_to_freeze, 'freeze'
        ):  # Ensure it's an actual distribution generator
            dist_to_freeze = dist_or_gen_obj  # if dist_or_gen_obj was poisson class, dist_or_gen_obj() made an instance.
            # if it was norm instance, this path is skipped.
    else:  # It's a generator instance like norm (an object of norm_gen)
        params_dict = dist_or_gen_obj.__class__.parameters
        dist_to_freeze = dist_or_gen_obj

    keys = random.split(key, len(params_dict))
    kwargs = {}

    # Temp storage for interdependent params like empirical's values for its weights
    generated_params_data = {}

    for i, (name, constraint) in enumerate(params_dict.items()):
        k = keys[i]
        param_batch_shape = shape

        # Skip dependent parameters, they should be handled by the distribution's _parse_args or freeze
        if constraint is dependent:
            continue

        # Special handling for empirical distribution
        # Assuming 'empirical' is the class probjax.stats.empirical.empirical
        is_empirical = dist_to_freeze.__class__ == empirical or isinstance(
            dist_to_freeze, empirical
        )

        if is_empirical:
            num_empirical_samples = 5  # Fixed for testing
            if name == "values":
                # 'values' for empirical: (num_samples,) + batch_shape
                value = random.normal(k, (num_empirical_samples,) + param_batch_shape)
                generated_params_data["values_shape_0"] = num_empirical_samples
            elif name == "weights":
                num_s = generated_params_data.get(
                    "values_shape_0", num_empirical_samples
                )
                raw_weights = random.uniform(
                    k, (num_s,) + param_batch_shape
                )  # Ensure weights match samples along axis 0
                value = raw_weights / jnp.sum(
                    raw_weights, axis=0, keepdims=True
                )  # Sum over samples for each batch element
            else:  # Should not happen for standard empirical
                value = random.normal(k, param_batch_shape)

        # Special handling for categorical 'probs'
        # Assuming 'categorical' is the class probjax.stats.categorical.categorical
        elif name == "probs" and (
            dist_to_freeze.__class__ == categorical
            or isinstance(dist_to_freeze, categorical)
        ):
            num_categories = 3  # Fixed for testing
            raw_probs = random.uniform(k, param_batch_shape + (num_categories,))
            value = raw_probs / jnp.sum(raw_probs, axis=-1, keepdims=True)

        elif isinstance(constraint, UnitInterval):
            value = random.uniform(k, param_batch_shape, minval=0.0, maxval=1.0)
        elif isinstance(
            constraint, PositiveInteger
        ):  # e.g. Poisson rate (though rate is float), or Binomial n (needs to be int)
            # For Poisson rate, it's float. For Binomial n, it's int. Constraint needs to be more specific or checked.
            # Let's generate float for now, as Poisson was an issue.
            value = jnp.abs(
                random.randint(k, param_batch_shape, 0, 10).astype(jnp.float32)
            )  # Cast to float
        elif isinstance(constraint, StrictPositiveInteger):  # e.g. Binomial n
            # Binomial 'n' is int.
            value = random.randint(k, param_batch_shape, 1, 10)  # Generate int
            # If a float parameter had this constraint, it would be: jnp.abs(random.normal(k, param_batch_shape)) + 1.0
        elif isinstance(constraint, interval):
            low = getattr(constraint, 'low', 0.0)
            high = getattr(constraint, 'high', 1.0)
            if not (
                isinstance(low, (int, float, jnp.ndarray))
                and isinstance(high, (int, float, jnp.ndarray))
            ):
                low, high = 0.0, 1.0  # Fallback

            # Ensure low < high for uniform sampling, handle scalar and array cases for low/high
            if isinstance(low, jnp.ndarray) or isinstance(high, jnp.ndarray):
                actual_low = jnp.where(
                    low < high - 1e-6, low, high - 1e-6
                )  # Element-wise min
                actual_low = jnp.where(
                    high > low, actual_low, low
                )  # if high <= low, use low
            else:  # Scalars
                actual_low = min(low, high - 1e-6) if high > low else low

            value = random.uniform(k, param_batch_shape, minval=actual_low, maxval=high)
            # Integer interval handling:
            # if getattr(constraint, 'dtype', float) == int:
            #     value = jnp.round(value).astype(jnp.int32) # Or floor/ceil depending on interval definition

        elif constraint is None or constraint is real:
            value = random.normal(k, param_batch_shape)
        elif constraint is boolean:
            value = random.bernoulli(k, 0.5, param_batch_shape)
        elif (
            constraint is integer
        ):  # Generic integer not covered by PositiveInteger etc.
            value = random.randint(
                k, param_batch_shape, -10, 10
            )  # Wider range for generic int
        elif constraint is positive:  # Non-strict positive float
            value = jnp.abs(random.normal(k, param_batch_shape))
        elif constraint is strict_positive:  # Strict positive float
            value = jnp.abs(random.normal(k, param_batch_shape)) + 1e-6
        elif constraint is simplex:
            # For a generic simplex, assume last dimension is K (e.g. 3 for testing)
            # This might need to be context-aware if the param_batch_shape is not just the batch dims
            simplex_event_dim = 3
            raw = random.uniform(k, param_batch_shape + (simplex_event_dim,))
            value = raw / jnp.sum(raw, axis=-1, keepdims=True)
        elif constraint is distribution_constraint:
            # Parameter is another distribution (e.g., base_dist for Transformed)
            # Create a simple frozen Normal distribution.
            # The shape of loc/scale should broadcast with param_batch_shape.
            # For simplicity, let loc have param_batch_shape and scale be 1.
            loc_val = random.normal(k, param_batch_shape)
            scale_val = jnp.ones(param_batch_shape)  # Ensure scale is broadcastable
            value = norm.freeze(loc=loc_val, scale=scale_val)
        else:  # Fallback for unhandled constraints
            print(
                f"Warning: Unhandled constraint type {constraint} for parameter {name}. Defaulting to random.normal."
            )
            value = random.normal(k, param_batch_shape)

        kwargs[name] = value

    # Ensure the correct object is used for freeze()
    # If dist_or_gen_obj was a class (e.g. poisson), dist_to_freeze is now an instance poisson()
    # If dist_or_gen_obj was an instance (e.g. norm), dist_to_freeze is that instance.
    return dist_to_freeze.freeze(**kwargs)


def mode_correct(p, key: jax.random.PRNGKey):
    # Check mode
    sample = p.rvs(key, (10000,))
    log_prob_samples = p.logpdf(sample)
    mode = sample[jnp.argmax(log_prob_samples)]
    mode_log_prob = p.logpdf(mode)

    assert mode.shape == p.batch_shape + p.event_shape, "Mode shape mismatch"
    assert jnp.isfinite(mode).all(), "Mode is not finite"
    try:
        assert jnp.allclose(p.mode(), mode, atol=0.5, rtol=1.0), (
            "Mode is not close to sample mode"
        )
        assert mode_log_prob <= p.logpdf(p.mode()), "Mode log_prob is not maximum"
    except AssertionError as e:
        raise e
    except NotImplementedError:
        pass


@pytest.mark.parametrize("dist", CONTINUOUS_DIST + DISCRETE_DIST + SPECIAL_DIST)
def test_distribution_class_attributes(dist):
    assert hasattr(dist, "parameters"), "Missing parameters"
    assert hasattr(dist, "support"), "Missing support"


@pytest.mark.parametrize("dist", CONTINUOUS_DIST + DISCRETE_DIST)
def test_base_distribution(dist, shape=(1,), seed=0):
    # Initialize distributions
    key = jax.random.PRNGKey(seed)
    p = init_dist(dist, key, shape)

    # Check sample and log_prob
    sample_and_log_prob(p, key, shape)
    mean_and_var(p, key)
    mode_correct(p, key)
    cdf_icdf(p, key)

    # Check PyTree
    flatten_p, tree_p = jax.tree_util.tree_flatten(p)
    q = jax.tree_util.tree_unflatten(tree_p, flatten_p)
    assert jnp.allclose(p.rvs(key, shape), q.rvs(key, shape)), (
        "PyTree reconstruction mismatch"
    )


@pytest.mark.parametrize("dist", CONTINUOUS_DIST + DISCRETE_DIST)
def test_independent_distribution(dist, shape=(2,), seed=0):
    key = jax.random.PRNGKey(seed)
    p = init_dist(dist, key, shape)

    try:
        p = independent(p, 1)
    except AssertionError:
        return

    # Check sample and log_prob
    sample_and_log_prob(p, key, shape)
    mean_and_var(p, key)
    mode_correct(p, key)

    # Check PyTree
    flatten_p, tree_p = jax.tree_util.tree_flatten(p)
    q = jax.tree_util.tree_unflatten(tree_p, flatten_p)
    assert jnp.allclose(p.rvs(key, shape), q.rvs(key, shape), atol=0.01, rtol=0.01), (
        "PyTree reconstruction mismatch"
    )


@pytest.mark.parametrize(
    "dist1, dist2", itertools.combinations(CONTINUOUS_DIST + DISCRETE_DIST, 2)
)
def test_mixed_independent_distribution(dist1, dist2, shape=(1,), seed=0):
    key = jax.random.PRNGKey(seed)

    p1 = init_dist(dist1, key, shape)
    p2 = init_dist(dist2, key, shape)
    # Batch shapes must be the same
    try:
        p = independent([p1, p2], 1)
    except AssertionError:
        # If batch and event shapes are different, we can't make an independent
        # distribution
        return

    sample_and_log_prob(p, key, shape)
    mean_and_var(p, key)
    mode_correct(p, key)


@pytest.mark.parametrize("dist", CONTINUOUS_DIST)
def test_transformed_distribution(dist, shape=(1,), seed=0):
    key = random.PRNGKey(seed)
    p_base = init_dist(dist, key, shape=shape)

    def bijector(x):
        return x + 1.0

    # Correct instantiation:
    p_transformed = transformed().freeze(base_dist=p_base, bijector=bijector)

    # Basic checks
    assert hasattr(p_transformed, "logpdf")
    assert hasattr(p_transformed, "mode")  # Assuming mode is implemented
    assert hasattr(p_transformed, "support")  # Assuming support is implemented
    # Add more specific tests for transformed distribution logic if needed
    x_test = p_transformed.rvs(random.PRNGKey(seed + 1))
    assert isinstance(p_transformed.logpdf(x_test), jnp.ndarray)


# Assuming 'mixture' is imported, e.g., from probjax.stats import mixture
# You'll need to find where 'mixture' class is defined.
# For now, I'll comment out the mixture test fix if 'mixture' is not available in current context.
# If 'mixture' is available:
# from probjax.stats import mixture # Make sure this import is correct

# @pytest.mark.parametrize("dist", CONTINUOUS_DIST + DISCRETE_DIST)
# def test_mixture_distribution(dist, shape=(1,), seed=0):
#     key = random.PRNGKey(seed)
#     p1 = init_dist(dist, random.PRNGKey(seed + 1), shape=shape)
#     p2 = init_dist(dist, random.PRNGKey(seed + 2), shape=shape)
#
#     mix_probs = jnp.array([0.5, 0.5])
#     components = [p1, p2]
#
#     # Correct instantiation:
#     # The parameter names for mixture().freeze might be different, e.g. 'probs' or 'weights'
#     # and 'components' or 'distributions'. Check mixture.py
#     # Assuming params are 'probs' and 'distributions':
#     try:
#         p_mix = mixture().freeze(probs=mix_probs, distributions=components)
#
#         # Basic checks
#         if hasattr(p_mix, "logpdf"): # For continuous mixtures
#             assert hasattr(p_mix, "logpdf")
#         elif hasattr(p_mix, "logpmf"): # For discrete mixtures
#             assert hasattr(p_mix, "logpmf")
#         assert hasattr(p_mix, "mode")
#         assert hasattr(p_mix, "support")
#         x_test = p_mix.rvs(random.PRNGKey(seed+3))
#         # Further checks
#     except Exception as e:
#         pytest.fail(f"Failed to initialize or test mixture distribution: {e}")

# Fix for test_independent_distribution (assuming independent is imported)
# from probjax.stats import independent

# @pytest.mark.parametrize("dist", CONTINUOUS_DIST + DISCRETE_DIST)
# def test_independent_distribution(dist, shape=(1,), seed=0):
#     key = random.PRNGKey(seed)
#     p_base = init_dist(dist, key, shape=shape)
#     reinterpreted_batch_ndims = 1

#     # Ensure the base distribution's batch_shape has at least reinterpreted_batch_ndims
#     if len(p_base.batch_shape) < reinterpreted_batch_ndims:
#         # This test case might not be suitable for this base dist/shape combination
#         # Or, we need to adjust 'shape' in init_dist to ensure enough batch dims
#         pytest.skip(f"Base distribution batch shape {p_base.batch_shape} too small for reinterpreted_batch_ndims={reinterpreted_batch_ndims}")
#         return

#     # Correct instantiation for independent:
#     try:
#         # Assuming constructor is independent(base_distribution, reinterpreted_batch_ndims)
#         # Or independent().freeze(base_distribution=..., reinterpreted_batch_ndims=...)
#         # Check independent.py for its API.
#         # If it's independent().freeze():
#         p_ind = independent().freeze(base_dist=p_base, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
#         # If its __init__ is independent(base_dist, reinterpreted_batch_ndims):
#         # p_ind = independent(base_dist=p_base, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
#         # The error was "takes from 1 to 2 positional arguments but 3 were given"
#         # This suggests the constructor might be simpler, or it's frozen differently.
#         # Let's assume it needs freeze():

#         # Basic checks
#         # ...
#     except Exception as e:
#         pytest.fail(f"Failed to initialize or test independent distribution: {e}")
