import pytest
import jax
import jax.numpy as jnp

from probjax.distributions import continuous
from probjax.distributions import discrete
from probjax.distributions.independent import Independent
from probjax.distributions.mixture import Mixture
from probjax.distributions.transformed_distribution import TransformedDistribution
from probjax.distributions import Distribution
from probjax.distributions.constraints import Constraint

from probjax.distributions.constraint_registry import transform_to


CONTINOUS_DIST = [getattr(continuous, name) for name in continuous.__all__]
DISCRETE_DIST = [getattr(discrete, name) for name in discrete.__all__]
SPECIAL_DIST = [Independent, Mixture, TransformedDistribution]

# Some helper functions
def sample_and_log_prob(p: Distribution, key: jax.random.PRNGKey, sample_shape):
    # Check log_prob and sample without sampling shape
    sample = p.sample(key, sample_shape)
    log_prob = p.log_prob(sample)

    assert sample.shape == sample_shape + p.batch_shape + p.event_shape, "Sample shape mismatch"
    assert log_prob.shape == sample_shape + p.batch_shape, "Log_prob shape mismatch"
    assert jnp.isfinite(log_prob).all(), "Log_prob is not finite for all samples"

def mean_and_var(p: Distribution, key: jax.random.PRNGKey):
    # Check mean and variance
    sample = p.sample(key, (10000,))
    mean = jnp.mean(sample, axis=0)
    var = jnp.var(sample, axis=0)
    std = jnp.sqrt(var)

    assert mean.shape == p.batch_shape + p.event_shape, "Mean shape mismatch"
    assert var.shape == p.batch_shape + p.event_shape, "Variance shape mismatch"
    assert jnp.isfinite(mean).all(), "Mean is not finite"
    assert jnp.isfinite(var).all(), "Variance is not finite"
    assert jnp.isfinite(std).all(), "Standard deviation is not finite"
    try:
        assert jnp.allclose(p.mean, mean, atol=0.1), "Mean is not close to sample mean"
        assert jnp.allclose(p.variance, var, atol=0.1), "Variance is not close to sample variance"
        assert jnp.allclose(p.stddev, std, atol=0.1), "Standard deviation is not close to sample standard deviation"
    except AssertionError as e:
        raise e 
    except NotImplementedError:
        pass

def mode_correct(p: Distribution, key: jax.random.PRNGKey):
    # Check mode
    sample = p.sample(key, (10000,))
    log_prob_samples = p.log_prob(sample)
    mode = sample[jnp.argmax(log_prob_samples)]
    mode_log_prob = p.log_prob(mode)

    assert mode.shape == p.batch_shape + p.event_shape, "Mode shape mismatch"
    assert jnp.isfinite(mode).all(), "Mode is not finite"
    try:
        assert jnp.allclose(p.mode, mode, atol=0.5), "Mode is not close to sample mode"
        assert mode_log_prob <= p.log_prob(mode), "Mode log_prob is not maximum"
    except AssertionError as e:
        raise e 
    except NotImplementedError:
        pass


@pytest.mark.parametrize("dist", CONTINOUS_DIST + DISCRETE_DIST + SPECIAL_DIST)
def test_distribution_class_attributes(dist: type[Distribution]):
    assert hasattr(dist, "arg_constraints") and isinstance(
        dist.arg_constraints, dict
    ), "Missing arg_constraints"
    assert hasattr(dist, "support") and isinstance(
        dist.support, Constraint
    ), "Missing support"
    assert hasattr(dist, "has_rsample") and isinstance(
        dist.has_rsample, bool
    ), "Missing has_rsample"

@pytest.mark.parametrize("dist", CONTINOUS_DIST + DISCRETE_DIST)
def test_base_distribution(dist: type[Distribution], shape = (1,), seed= 0):
    # Initialize distributions
    key = jax.random.PRNGKey(seed)
    kwargs = dict([(name, transform_to(constraint)(jax.random.normal(key,shape)*3)) for name, constraint in dist.arg_constraints.items()])
    p = dist(**kwargs)

    # Check batch and event shapes
    assert p.batch_shape + p.event_shape == shape, "Shapes do not match"

    # Check sample and log_prob
    sample_and_log_prob(p, key, shape)
    mean_and_var(p, key)
    mode_correct(p, key)