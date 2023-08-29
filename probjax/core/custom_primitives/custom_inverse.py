import jax

import jax.numpy as jnp

jax.numpy.set_printoptions(precision=3, suppress=True)
from jax import core

from jax import linear_util as lu
from functools import partial, update_wrapper

from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves, tree_map
from jax.interpreters import ad, batching
from jax._src import ad_util

from jax.core import Primitive, CallPrimitive
from jax._src.util import weakref_lru_cache, cache


from typing import Any, Callable
from jax._src.util import safe_map
from jax._src.api_util import (
    flatten_fun_nokwargs,
    argnums_partial,
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


def custom_inverse_jvp(primals, tangents, forward_jaxpr, inverse_jaxpr, **params):
    nonzeros = [type(t) is not ad_util.Zero for t in tangents]
    forward_jvp_jaxpr, forward_out_nz = ad.jvp_jaxpr(
        forward_jaxpr, nonzeros, instantiate=False
    )
    nonzero_tangents = [t for t in tangents if type(t) is not ad_util.Zero]
    forward_jvp_jaxpr_ = pe.convert_constvars_jaxpr(forward_jvp_jaxpr.jaxpr)

    new_primals, new_tangent = core.eval_jaxpr(
        forward_jvp_jaxpr_, forward_jvp_jaxpr.consts, *primals, *nonzero_tangents
    )

    return new_primals, new_tangent


def batch_custom_inverse_call(
    spmd_axis_name, axis_size, axis_name, main_type, args, dims, **params
):
    forward_jaxpr = params.pop("forward_jaxpr")
    inverse_jaxpr = params.pop("inverse_jaxpr")

    # We have to batch the jaxprs. For that lets first get the invals and outvals
    in_avals1 = forward_jaxpr.in_avals
    out_avals1 = forward_jaxpr.out_avals

    in_avals2 = inverse_jaxpr.in_avals
    out_avals2 = inverse_jaxpr.out_avals

    # We will batch all the inputs and outputs  (maybe do not batch consts ... )
    in_batched1 = [True] * len(in_avals1)
    out_batched1 = [True] * len(out_avals1)

    in_batched2 = [True] * len(in_avals2)
    out_batched2 = [True] * len(out_avals2)

    # Applies the batching for the jaxprs
    args = [batching.bdim_at_front(x, d, axis_size) for x, d in zip(args, dims)]

    # Batched jaxprs
    batched_forward_fn, out_size1 = batching.batch_jaxpr(
        forward_jaxpr,
        axis_size,
        in_batched1,
        out_batched1,
        axis_name,
        spmd_axis_name,
        main_type,
    )
    batched_inverse_fn, _ = batching.batch_jaxpr(
        inverse_jaxpr,
        axis_size,
        in_batched2,
        out_batched2,
        axis_name,
        spmd_axis_name,
        main_type,
    )

    # Update jaxprs with batched ones
    out = custom_inverse_call_p.bind(
        *args,
        forward_jaxpr=batched_forward_fn,
        inverse_jaxpr=batched_inverse_fn,
        **params,
    )

    # Outdim
    out_dims = [0 if b else batching.not_mapped for b in out_size1]

    return out, out_dims


def custom_inverse_transpose(*args, **kwargs):
    return ad.call_transpose(custom_inverse_call_p, *args, **kwargs)


batching.spmd_axis_primitive_batchers[custom_inverse_call_p] = batch_custom_inverse_call
batching.axis_primitive_batchers[custom_inverse_call_p] = partial(
    batch_custom_inverse_call, None
)
ad.primitive_transposes[custom_inverse_call_p] = custom_inverse_transpose
ad.primitive_jvps[custom_inverse_call_p] = custom_inverse_jvp


# TODO: Add support other tracer support!


class custom_inverse:
    def __init__(self, fun: Callable, static_argnums=None) -> None:
        update_wrapper(self, fun)
        self.fun = fun
        self.static_argnums = static_argnums

    def definv(self, inv_fun: Callable) -> Callable:
        def wrapped_inv(*args):
            return inv_fun(*args), jnp.nan

        self.inv_fun = inv_fun
        self.inv_fun_and_log_det = wrapped_inv
        return wrapped_inv

    def definv_and_logdet(self, inv_fun_and_log_det: Callable) -> Callable:
        self.inv_fun_and_log_det = inv_fun_and_log_det
        self.inv_fun = lambda *args, **kwargs: inv_fun_and_log_det(*args, **kwargs)[0]
        return inv_fun_and_log_det

    def inv(self, *args, **kwargs):
        return self.inv_fun(*args, **kwargs)

    def inv_and_logdet(self, *args, **kwargs):
        return self.inv_fun_and_log_det( *args, **kwargs)

    def __call__(self, *args, **params) -> Any:
        name = getattr(self.fun, "__name__", str(self.fun))
        if not self.inv_fun:
            msg = f"No inverse defined for custom_inverse function {name} using definv."
            raise AttributeError(msg)
        inv_name = getattr(self.inv_fun, "__name__", str(self.inv_fun))

        f = lu.wrap_init(self.fun, params=params)
        if self.static_argnums is None:
            dyn_args = args
        else:
            dyn_args = (i for i in range(len(args)) if i not in self.static_argnums)
            f, dyn_args = argnums_partial(
                f, dyn_args, args, require_static_args_hashable=False
            )

        args_flat, in_tree = tree_flatten(dyn_args)
        jax_tree_fun, out_tree = flatten_fun_nokwargs(f, in_tree)  # type: ignore
        in_avals = tuple(safe_map(shaped_abstractify, args_flat))
        debug = pe.debug_info(self.fun, in_tree, out_tree, False, name or "<unknown>")
        jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(
            jax_tree_fun, in_avals, debug
        )
        forward_jaxpr = core.ClosedJaxpr(jaxpr, consts)
        out_tree = out_tree()

        f_inv = lu.wrap_init(self.inv_fun_and_log_det, params=params)
        if self.static_argnums is None:
            dyn_args = args
        else:
            dyn_args = (i for i in range(len(args)) if i not in self.static_argnums)
            f_inv, dyn_args = argnums_partial(
                f_inv, dyn_args, args, require_static_args_hashable=False
            )
        jax_tree_inv_fun, out_tree_inv = flatten_fun_nokwargs(f_inv, in_tree)  # type: ignore
        debug = pe.debug_info(
            self.inv_fun_and_log_det,
            in_tree,
            out_tree_inv,
            False,
            inv_name or "<unknown>",
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
