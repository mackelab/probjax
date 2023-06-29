from typing import Dict, Optional, Any, Tuple, Callable


import jax
import jax.numpy as jnp
import jax.random as jrandom

from probjax.distributions.distribution import Distribution

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class ConditionalDistribution:

    params: Dict[str, Any] = {}

    def __init__(self, context_shape: tuple, event_shape: tuple, conditionor:Callable, distribution: type[Distribution], params = {}) -> None:
        self._conditionor = conditionor
        self._distribution = distribution
        self.params = params

        self._context_shape = context_shape
        self._event_shape = event_shape
        self.support = distribution.support

        super().__init__()

    @property
    def context_shape(self) -> tuple:
        """
        Returns the shape over which parameters are batched.
        """
        return self._context_shape
    
    @property
    def event_shape(self) -> tuple:
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape
    
    def __repr__(self) -> str:
        return f"{self._distribution.__name__}(array({self._event_shape})|array({self._context_shape}))"
    
    def __call__(self, context) -> Distribution:
        out = self._conditionor(self.params, context)
        return self._distribution(*out)
    
    def tree_flatten(self):
        return (self.params,), (self._conditionor, self._distribution)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children[0])