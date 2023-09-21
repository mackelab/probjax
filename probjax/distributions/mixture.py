import jax
import jax.numpy as jnp
from jax import random
from jax import lax
from jax.scipy.special import erfinv, erf

from jaxtyping import Array
from typing import Sequence

from probjax.distributions.exponential_family import Distribution

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

        self.arg_constraints["components"] = None
        num_components = mixing_probs.shape[-1]
        assert num_components == len(
            components
        ), "Number of components must match number of mixing probabilities"
        batch_shape = mixing_probs.shape[:-1]
        event_shape = components[0].event_shape

        self.support = components[
            0
        ].support  # Pick the component with the largest support TODO

        
        for i, component in enumerate(components):
            assert (
                component.batch_shape == batch_shape
            ), "Batch shape of components must match"
            assert (
                component.event_shape == event_shape
            ), "Event shape of components must match"
            component_args = component.arg_constraints
            for arg in component_args:
                self.arg_constraints[arg + f"_{i}"] = component_args[arg]

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        key_comp, key_sample, key_permute = random.split(key, 3)
        shape = sample_shape + self.batch_shape + self.event_shape
        mixture_idx = random.categorical(
            key_comp, self.mixing_probs, shape=shape[:-1] if len(shape) > 1 else shape
        )
        components, num_samples = jnp.unique(mixture_idx, return_counts=True)

        total_samples = []
        for i in range(len(self.components)):
            component_idx = components[i]
            component = self.components[component_idx]
            component_sample = component.sample(
                key_sample, sample_shape=(num_samples[i],)
            )
            total_samples.append(component_sample)

        total_samples = jnp.concatenate(total_samples, axis=0)[
            random.permutation(key_permute, jnp.arange(shape[0]))
        ]
        return total_samples

    def rsample(self, key, sample_shape: tuple = ...) -> Array:
        raise NotImplementedError(
            "Mixture does not support reparameterized sampling, can be done -> implicit reparam."
        )

    def log_prob(self, value):
        log_probs = []
        for i, component in enumerate(self.components):
            log_prob = component.log_prob(value)
            log_probs.append(log_prob + jnp.log(self.mixing_probs[..., i]))
        return jax.scipy.special.logsumexp(jnp.stack(log_probs, axis=-1), axis=-1)

    def cdf(self, value):
        cdf_components = []
        for i, component in enumerate(self.components):
            cdf_component = component.cdf(value)
            cdf_components.append(cdf_component)
        cdf_components = jnp.stack(cdf_components, axis=-1)
        return jnp.sum(cdf_components * self.mixing_distribution.probs, axis=-1)

    def icdf(self, value):
        icdf_components = []
        for i, component in enumerate(self.components):
            icdf_component = component.icdf(value)
            icdf_components.append(icdf_component)
        icdf_components = jnp.stack(icdf_components, axis=-1)
        return jnp.sum(icdf_components * self.mixing_distribution.probs, axis=-1)

    def mean(self):
        mean_components = []
        for i, component in enumerate(self.components):
            mean_component = component.mean()
            mean_components.append(mean_component)
        mean_components = jnp.stack(mean_components, axis=-1)
        return jnp.sum(mean_components * self.mixing_distribution.probs, axis=-1)

    def variance(self):
        variance_components = []
        for i, component in enumerate(self.components):
            variance_component = component.variance()
            variance_components.append(variance_component)
        variance_components = jnp.stack(variance_components, axis=-1)
        return jnp.sum(variance_components * self.mixing_distribution.probs, axis=-1)

    # Each distribution will be registered as a PyTree
    def tree_flatten(self):
        flat_components, tree_components = jax.tree_util.tree_flatten(self.components)
        return (
            (self.mixing_probs,) + tuple(flat_components),
            [tree_components],
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tree_components = aux_data[0]
        return cls(
            children[0], jax.tree_util.tree_unflatten(tree_components, children[1:])
        )

    def __repr__(self) -> str:
        return (
            "Mixture"
            + "("
            + "mixing_probs="
            + self.mixing_probs.__repr__()
            + ", components="
            + self.components.__repr__()
            + ")"
        )


class MixtureSameFamily(Distribution):
    def __init__(self, mixing_probs: Array, components: Distribution):
        self.mixing_probs = mixing_probs
        self.components = components

        self.arg_constraints["components"] = None
        num_components = mixing_probs.shape[-1]
        batch_shape = mixing_probs.shape[:-1]
        event_shape = components.event_shape
        assert (
            num_components == components.batch_shape[-1]
        ), "Batchdim of components must match number of mixing probabilities"

        self.support = components.support
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, key, sample_shape=()):
        pass
