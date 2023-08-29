import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array

from .distribution import Distribution
from .constraints import real, positive, unit_interval, interval

__all__ = ["TruncatedNormal"]

from jax.tree_util import register_pytree_node_class
from jax.scipy.stats import truncnorm


@register_pytree_node_class
class TruncatedDistribution(Distribution):
    def __init__(self, base_dist, left_bound, right_bound):
        self.base_dist = base_dist

        assert left_bound < right_bound, "Left bound must be less than right bound"
        self.left_bound = left_bound
        self.right_bound = right_bound

        batch_shape = base_dist.batch_shape
        event_shape = base_dist.event_shape

        self.arg_constraints["base_dist"] = None
        self.arg_constraints["left_bound"] = real
        self.arg_constraints["right_bound"] = real

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        cdf_left = self.base_dist.cdf(self.left_bound)
        cdf_right = self.base_dist.cdf(self.right_bound)
        average_acceptance_probability = (cdf_right - cdf_left).prod(-len(self.event_shape)).mean()
        assert average_acceptance_probability < 1e-4, "Average acceptance probability is too low"
        num_samples = (sample_shape + self.batch_shape).prod()
        expected_required_samples = num_samples / average_acceptance_probability

        total_samples = []
        while len(total_samples) < num_samples:
            key, subkey = random.split(key)
            samples = self.base_dist.sample(subkey, (expected_required_samples,))
            accepted_samples = jnp.where(
                (samples >= self.left_bound) & (samples <= self.right_bound)
            )
            total_samples.append(samples[accepted_samples])
        total_samples = jnp.concatenate(total_samples, axis=0)
        return total_samples[:num_samples].reshape(shape)
    
    def log_prob(self, value):
        return self.base_dist.log_prob(value) - jnp.log(self.cdf(self.right_bound) - self.cdf(self.left_bound))
        
