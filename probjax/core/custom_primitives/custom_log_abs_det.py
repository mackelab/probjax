import jax
import jax.numpy as jnp
from jax import core

from jax import linear_util as lu
from functools import partial, update_wrapper

from jax.tree_util import tree_flatten, tree_unflatten

from jax.core import Primitive, CallPrimitive

from jax._src.lax.control_flow import _initial_style_jaxpr

from typing import Any, Callable
from jax._src.util import safe_map
from jax._src.api_util import argnums_partial_except, flatten_fun

from jax.interpreters import mlir

# This probably should be a call primitive

custom_logdet_call_p = Primitive("custom_inverse_call_p")
custom_logdet_call_p.multiple_results = True


@custom_logdet_call_p.def_impl
def custom_inverse_call_impl(*args, forward_jaxpr, inverse_jaxpr, **params):
    with core.new_sublevel():
        ans = core.eval_jaxpr(forward_jaxpr.jaxpr, forward_jaxpr.literals, *args)
    return ans


@custom_logdet_call_p.def_abstract_eval
def custom_inverse_call_abstract_eval(*args, forward_jaxpr, inverse_jaxpr, **params):
    with core.new_sublevel():
        return forward_jaxpr.out_avals


def custom_inverse_call_lowering(ctx, *args, forward_jaxpr, inverse_jaxpr, **params):
    return mlir.core_call_lowering(ctx, *args, name="forward_call", call_jaxpr=forward_jaxpr)


mlir.register_lowering(custom_logdet_call_p, custom_inverse_call_lowering)


def seperate_args(args, static_argnums):
    if static_argnums is None:
        return args, ()
    else:
        static_argnums = sorted(static_argnums)
        static_args = tuple(args[i] for i in static_argnums)
        dynamic_args = tuple(args[i] for i in range(len(args)) if i not in static_argnums)
        return dynamic_args, static_args

class custom_logabsdet:
    def __init__(self, fun: Callable, static_argnums = None) -> None:
        update_wrapper(self, fun)
        self.fun = fun
        self.static_argnums = static_argnums

    def definv(self, inv_fun: Callable) -> Callable:
        self.inv_fun = inv_fun
        return inv_fun

    def inv(self, *args):
        return self.inv_fun(*args)

    def __call__(self, *args) -> Any:
        name = getattr(self.fun, "__name__", str(self.fun))
        if not self.inv_fun:
            msg = f"No inverse defined for custom_inverse function {name} using definv."
            raise AttributeError(msg)
        inv_name = getattr(self.inv_fun, "__name__", str(self.inv_fun))

        f = lu.wrap_init(self.fun)
        f, dyn_args = argnums_partial_except(f, self.static_argnums, args, allow_invalid=False)
        args_flat, in_tree = tree_flatten(dyn_args)
        jax_tree_fun, out_tree = flatten_fun(f, in_tree)


        in_avals = tuple(safe_map(core.get_aval, args_flat))
        in_avals = tuple(safe_map(core.raise_to_shaped, in_avals))
        forward_jaxpr, consts_f, out_tree_f = _initial_style_jaxpr(
            self.fun, in_tree, in_avals
        )
        out_avals = tuple(forward_jaxpr.out_avals)
        out_avals = tuple(safe_map(core.raise_to_shaped, out_avals))
        inverse_jaxpr, consts_i, out_tree_i = _initial_style_jaxpr(
            self.inv_fun, out_tree_f, out_avals
        )

        out_flat = custom_inverse_call_p.bind(
            *args_flat,
            forward_jaxpr=forward_jaxpr,
            inverse_jaxpr=inverse_jaxpr,
            in_tree=in_tree,
        )
        return tree_unflatten(out_tree_f, out_flat)
