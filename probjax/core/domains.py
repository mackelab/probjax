from typing import Any
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
from jax.core import Jaxpr

from jaxtyping import PyTree, Array, Float, Int, Bool
from typing import Union

from abc import abstractmethod

from ..distributions.constraints import real, integer, boolean, interval, positive, negative, unit_interval, unit_square

from jax.core import Primitive
from jax.lax import tanh_p, exp_p, log_p, add_p, sub_p, mul_p, div_p
from .utils import BaseRules, BaseInterpreter

class DomainRules(BaseRules):

    def _default_rule(self, prim: Primitive, *args, **kwargs) -> Any:
        return real
    
rules = DomainRules()



def base_transformer(output_domain):
    def transformer(prim, *args, **params):
        # Simplest implemenation -> Can be more specific for certain input domains
        return output_domain
    
    return transformer

rules[tanh_p] = base_transformer(unit_square)
rules[exp_p] = base_transformer(positive)
rules[log_p] = base_transformer(real)
rules[add_p] = base_transformer(real)
rules[add_p] = base_transformer(real)
rules[mul_p] = base_transformer(real)
rules[div_p] = base_transformer(real)



class DomainInterpreter(BaseInterpreter):
    rules = rules



    def _get_output(self, jaxpr: Jaxpr) -> Any:
        return self.env




