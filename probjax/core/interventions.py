from typing import Any, Callable, Iterable, Optional
from jax import lax
import jax.numpy as jnp
import jax.random as jrandom
from jax.core import Primitive, Jaxpr, JaxprEqn, ClosedJaxpr
from jax._src.util import safe_map

from probjax.core.utils import BaseInterpreter, BaseRules, jaxpr_returning_const, remove_closed_jaxpr_vars_with_suffix
from probjax.core.custom_primitives.random_variable import rv_p
from jax import linear_util as lu
from jax import tree_util
from jax import api_util






class InterventionRules(BaseRules):
    def __init__(self, interventions: dict) -> None:
        self.interventions = interventions
        self.__dict__[rv_p] = self._rv_rule  # type: ignore

    def _rv_rule(self, prim: Primitive, *args, **kwargs) -> Any:
        name = kwargs["name"]
        if name in self.interventions:
            
            sampling_fn_jaxpr = kwargs.pop("sampling_fn_jaxpr")
            log_prob_fn_jaxpr = kwargs.pop("log_prob_fn_jaxpr")
            
            
            out_aval = sampling_fn_jaxpr.out_avals[0]
            out_const  = self.interventions[name]
            assert out_aval.shape == out_const.shape, f"Shape mismatch: {out_aval.shape} != {out_const.shape}. Intervention must have same shape as random variable."
            assert out_aval.dtype == out_const.dtype, f"Dtype mismatch: {out_aval.dtype} != {out_const.dtype}. Intervention must have same dtype as random variable."

            new_sampling_fn_jaxpr, out_tree = jaxpr_returning_const(out_const, invars=sampling_fn_jaxpr.jaxpr.invars)
            # Maybe remove unnecessary variables from jaxpr in future
            #new_log_prob_fn_jaxpr = remove_closed_jaxpr_vars_with_suffix(log_prob_fn_jaxpr)
 
            kwargs["sampling_fn_jaxpr"] = new_sampling_fn_jaxpr
            kwargs["log_prob_fn_jaxpr"] = log_prob_fn_jaxpr 
            out = rv_p.bind(*args,  **kwargs)

            return tree_util.tree_unflatten(out_tree, out)
        else:
            return prim.bind(*args, **kwargs)


class InterventionInterpreter(BaseInterpreter):
    def __init__(self, interventions: dict) -> None:
        self.rules = InterventionRules(interventions)
