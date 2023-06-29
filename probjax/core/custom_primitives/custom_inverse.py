import jax
import jax.numpy as jnp
from jax import core

from jax import linear_util as lu
from functools import partial, update_wrapper

from jax.tree_util import tree_flatten, tree_unflatten

from jax.core import Primitive, CallPrimitive
from jax._src.util import weakref_lru_cache, cache


from typing import Any, Callable
from jax._src.util import safe_map
from jax._src.api_util import (
    flatten_fun_nokwargs,
    argnums_partial_except,
    flatten_fun,
    flatten_fun_nokwargs,
    shaped_abstractify,
)

from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe

custom_inverse_call_p = Primitive("custom_inverse_call_p")
custom_inverse_call_p.multiple_results = True


@custom_inverse_call_p.def_impl
def custom_inverse_call_impl(*args, forward_jaxpr, inverse_jaxpr, **params):
    with core.new_sublevel():
        ans = core.eval_jaxpr(forward_jaxpr.jaxpr, forward_jaxpr.literals, *args)
    return ans


@custom_inverse_call_p.def_abstract_eval
def custom_inverse_call_abstract_eval(*args, forward_jaxpr, inverse_jaxpr, **params):
    with core.new_sublevel():
        return forward_jaxpr.out_avals


def custom_inverse_call_lowering(ctx, *args, forward_jaxpr, inverse_jaxpr, **params):
    return mlir.core_call_lowering(
        ctx, *args, name="forward_call", call_jaxpr=forward_jaxpr
    )


mlir.register_lowering(custom_inverse_call_p, custom_inverse_call_lowering)


class custom_inverse:
    def __init__(self, fun: Callable, static_argnums=None) -> None:
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
        f, dyn_args = argnums_partial_except(
            f, self.static_argnums, args, allow_invalid=False
        )
        args_flat, in_tree = tree_flatten(dyn_args)
        jax_tree_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
        in_avals = tuple(safe_map(shaped_abstractify, args_flat))
        debug = pe.debug_info(self.fun, in_tree, out_tree, False, name or "<unknown>")
        jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(
            jax_tree_fun, in_avals, debug
        )
        forward_jaxpr = core.ClosedJaxpr(jaxpr, consts)
        out_tree = out_tree()

        f_inv = lu.wrap_init(self.inv_fun)
        f_inv, _ = argnums_partial_except(
            f_inv, self.static_argnums, args, allow_invalid=False
        )
        jax_tree_inv_fun, out_tree_inv = flatten_fun_nokwargs(f_inv, in_tree)
        debug = pe.debug_info(
            self.inv_fun, in_tree, out_tree_inv, False, inv_name or "<unknown>"
        )
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(jax_tree_inv_fun, in_avals, debug)
        inverse_jaxpr = core.ClosedJaxpr(jaxpr, consts)

        out_flat = custom_inverse_call_p.bind(
            *args_flat,
            forward_jaxpr=forward_jaxpr,
            inverse_jaxpr=inverse_jaxpr,
            in_tree=in_tree,
        )
        return tree_unflatten(out_tree, out_flat)
