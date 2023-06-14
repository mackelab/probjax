from typing import Dict, Optional, Any, Tuple, Callable


import jax
import jax.numpy as jnp
import jax.random as jrandom

from probjax.distributions.distribution import Distribution

class ConditionalDistribution:

    def __init__(self, conditionor:Callable, distribution: type[Distribution]) -> None:
        
        self._conditionor = conditionor
        self._distribution = distribution

        super().__init__()
    
    def __call__(self, *args, **kwargs):
        out = self._conditionor(*args, **kwargs)
        return self._distribution(out)