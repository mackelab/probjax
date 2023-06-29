import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array
from typing import Sequence

from .exponential_family import Distribution
from .constraints import real, positive, simplex

__all__ = ["Mixture"]

from jax.tree_util import register_pytree_node_class
from jax.scipy.stats import norm


@register_pytree_node_class
class Mixture(Distribution):
    arg_constraints = {"mixing_probs": simplex}

    def __init__(self, mixing_probs: Array, components: Sequence[Distribution]):
        self.mixing_probs = mixing_probs
        self.components = components

        num_components = mixing_probs.shape[-1]
        assert num_components == len(components)
        batch_shape = mixing_probs.shape[:-1]
        event_shape = components[0].event_shape
        self.support = components[0].support
        for i, component in enumerate(components):
            assert component.batch_shape == batch_shape
            assert component.event_shape == event_shape
            component_args = component.arg_constraints
            for arg in component_args:
                self.arg_constraints[arg + f"_{i}"] = component_args[arg]


        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        mixture_idx = random.categorical(key, self.mixing_probs, shape=shape[:-1])
        mixture_idx = jnp.expand_dims(mixture_idx, axis=-1)
        samples = random.normal(key, shape=shape)
        return jnp.take_along_axis(self.loc + self.scale * samples, mixture_idx, axis=-2)

