from typing import Dict, Optional, Any, Tuple


import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax.tree_util import register_pytree_node_class

__all__ = ["Distribution"]


@register_pytree_node_class
class Distribution:
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    arg_constraints: Dict[str, Any] = {}
    has_rsample = False

    def __init__(
        self,
        batch_shape: tuple = tuple(),
        event_shape: tuple = tuple(),
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape

        super().__init__()

    @property
    def batch_shape(self) -> tuple:
        """
        Returns the shape over which parameters are batched.
        """
        return self._batch_shape

    @property
    def event_shape(self) -> tuple:
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape

    @property
    def mean(self) -> jnp.array:
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def mode(self) -> jnp.array:
        """
        Returns the mode of the distribution.
        """
        raise NotImplementedError(f"{self.__class__} does not implement mode")

    @property
    def variance(self) -> jnp.array:
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError

    @property
    def stddev(self) -> jnp.array:
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()

    def sample(self, key, sample_shape: tuple = tuple()) -> jnp.array:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        return self.rsample(key, sample_shape)

    def rsample(self, key, sample_shape: tuple = tuple()) -> jnp.array:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError

    def log_prob(self, value: jnp.array) -> jnp.array:
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (array):
        """
        raise NotImplementedError

    def cdf(self, value: jnp.array) -> jnp.array:
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (array):
        """
        raise NotImplementedError

    def icdf(self, value: jnp.array) -> jnp.array:
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (array):
        """
        raise NotImplementedError

    def entropy(self) -> jnp.array:
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            array of shape batch_shape.
        """
        raise NotImplementedError

    def perplexity(self) -> jnp.array:
        """
        Returns perplexity of distribution, batched over batch_shape.

        Returns:
            array of shape batch_shape.
        """
        return jnp.exp(self.entropy())

    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            [
                "{}: {}".format(
                    p,
                    self.__dict__[p]
                    if self.__dict__[p].size == 1
                    else self.__dict__[p].size,
                )
                for p in param_names
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"


    # JAX jit requires this
    def tree_flatten(self):
        print("Distribution flattened")
        return tuple(getattr(self, param) for param in self.arg_constraints.keys()), None


    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(cls.arg_constraints.keys(), children)))
