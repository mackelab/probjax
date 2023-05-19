from typing import Any, Callable, Iterable, Optional
from jax import lax
import jax.numpy as jnp
import jax.random as jrandom
from jax.core import Primitive, Jaxpr, JaxprEqn, ClosedJaxpr
from jax._src.util import safe_map

from probjax.core.utils import BaseInterpreter, BaseRules
from probjax.core.random_variable import rv_p
from jax import linear_util as lu
from jax import tree_util
from jax import api_util

from jax._src.interpreters import partial_eval as pe
from jax._src.lax.control_flow import (
    _initial_style_open_jaxpr,
    _initial_style_jaxprs_with_common_consts,
)
from jax._src.lax.control_flow.common import (
    _abstractify,
    _avals_short,
    _check_tree_and_avals,
    _initial_style_jaxprs_with_common_consts,
    _make_closed_jaxpr,
    _prune_zeros,
    _typecheck_param,
    allowed_effects,
)




class InterventionRules(BaseRules):
    def __init__(self, interventions: dict) -> None:
        self.interventions = interventions
        self.__dict__[rv_p] = self._rv_rule  # type: ignore

    def _rv_rule(self, prim: Primitive, *args, **kwargs) -> Any:
        name = kwargs["name"]

        if name in self.interventions:

            def sample(*args, **kwargs):
                val = self.interventions[name]
                #val_broadcasted = jnp.broadcast_to(val, args[-1].shape[:-1])
                return val
            
            operands = [
                jrandom.PRNGKey(0),
            ]
            sampling_ops, sampling_ops_tree = tree_util.tree_flatten(operands)
            sampling_ops_avals = tuple(map(_abstractify, sampling_ops))
            sampling_jaxpr, sampling_consts, sampling_out_trees = _initial_style_open_jaxpr(
                sample, sampling_ops_tree, sampling_ops_avals
            )
            sampling_fn_jaxpr = ClosedJaxpr(pe.convert_constvars_jaxpr(sampling_jaxpr), ())
            num_sampling_consts = kwargs["num_sampling_consts"]
            kwargs["sampling_fn_jaxpr"] = sampling_fn_jaxpr
            kwargs["num_sampling_consts"] = len(sampling_consts)
 
            out = rv_p.bind(*sampling_consts, *args[num_sampling_consts:],  **kwargs)
            return tree_util.tree_unflatten(sampling_out_trees, out)
        else:
            return prim.bind(*args, **kwargs)


class InterventionInterpreter(BaseInterpreter):
    def __init__(self, interventions: dict) -> None:
        self.rules = InterventionRules(interventions)
