from functools import update_wrapper
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import core
from jax._src import ad_util
from jax._src import linear_util as lu
from jax._src.api_util import argnums_partial, debug_info, flatten_fun_nokwargs
from jax._src.core import shaped_abstractify
from jax._src.util import safe_map, safe_zip
from jax.extend.core import ClosedJaxpr, Primitive
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_flatten, tree_unflatten

# Use safe_map and safe_zip.
map = safe_map
zip = safe_zip

jax.numpy.set_printoptions(precision=3, suppress=True)


class custom_inverse:
    """Provides a mechanism to define a custom inverse and optional log-determinant for a function."""

    def __init__(self, fun: Callable, inv_argnum=0, static_argnums=None) -> None:
        """Initialize a custom inverse wrapper with optional static arguments."""
        update_wrapper(self, fun)
        self.fun = fun
        self.static_argnums = static_argnums
        self.inv_argnum = inv_argnum
        self.inv_fun = None  # Will be set via definv / definv_and_logdet
        self.inv_fun_and_log_det = None  # Will be set via definv_and_logdet

        # If we want to invert a function, then it should also be able to compute the
        # value and the logdet
        self.value_and_logdet_fun = None  # Will be set via defvalue_and_logdet

    def definv(self, inv_fun: Callable) -> Callable:
        """Define an inverse function without returning a log-determinant."""

        def _wrapped_inv(*args, **kwargs):
            # You can choose how to return a log-det if needed.
            return inv_fun(*args, **kwargs), jnp.nan

        self.inv_fun = inv_fun
        self.inv_fun_and_log_det = _wrapped_inv
        return _wrapped_inv

    def defvalue_and_logdet(self, value_and_logdet_fun: Callable) -> Callable:
        self.value_and_logdet_fun = value_and_logdet_fun
        if not hasattr(self, "value_and_logdet_fun"):
            self.value_and_logdet_fun = lambda *args, **kwargs: value_and_logdet_fun(
                *args, **kwargs
            )[0]
        return value_and_logdet_fun

    def definv_and_logdet(self, inv_fun_and_log_det: Callable) -> Callable:
        self.inv_fun_and_log_det = inv_fun_and_log_det
        if not hasattr(self, "inv_fun"):
            self.inv_fun = lambda *args, **kwargs: inv_fun_and_log_det(*args, **kwargs)[
                0
            ]
        return inv_fun_and_log_det

    def inv(self, *args, **kwargs):
        return self.inv_fun(*args, **kwargs)

    def inv_and_logdet(self, *args, **kwargs):
        return self.inv_fun_and_log_det(*args, **kwargs)

    def value_and_logdet(self, *args, **kwargs):
        return self.value_and_logdet_fun(*args, **kwargs)

    def __call__(self, *args, **params) -> Any:
        name = getattr(self.fun, "__name__", str(self.fun))
        if not self.inv_fun_and_log_det:
            msg = f"No inverse defined for custom_inverse function {name} using definv."
            raise AttributeError(msg)

        # Wrap forward and inverse functions with any static parameters.
        info = debug_info(
            "Trace for inverse of custom_inverse function", self.fun, (), {}
        )
        f = lu.wrap_init(self.fun, params=params, debug_info=info)
        f_inv = lu.wrap_init(self.inv_fun_and_log_det, params=params, debug_info=info)

        # Determine which arguments are dynamic.
        if self.static_argnums is None:
            dyn_args = args
            dyn_args_index = tuple(range(len(args)))
        else:
            dyn_args_index = tuple(
                i for i in range(len(args)) if i not in self.static_argnums
            )
            f, dyn_args = argnums_partial(
                f, dyn_args_index, args, require_static_args_hashable=True
            )
            f_inv, _ = argnums_partial(
                f_inv, dyn_args_index, args, require_static_args_hashable=True
            )

        # Flatten the dynamic args and compute abstract values.
        args_flat, in_tree = tree_flatten(dyn_args)
        in_avals = tuple(map(shaped_abstractify, args_flat))

        # Trace the forward jaxpr eagerly.
        f_flat, out_tree_fn = flatten_fun_nokwargs(f, in_tree)
        jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(f_flat, in_avals)
        forward_jaxpr = ClosedJaxpr(jaxpr, consts)
        out_tree = out_tree_fn()

        # Create a thunk for the inverse jaxpr. Notice we delay the call to trace the inverse.
        def lazy_inverse_jaxpr():
            f_inv_flat, _ = flatten_fun_nokwargs(f_inv, in_tree)
            inv_in_avals = list(in_avals)
            # Replace the abstract value of the inversion target with the forward output’s.
            i = dyn_args_index.index(self.inv_argnum)
            inv_in_avals[i] = out_avals[0]
            jaxpr_inv, _, consts_inv = pe.trace_to_jaxpr_dynamic(
                f_inv_flat, inv_in_avals
            )
            return ClosedJaxpr(jaxpr_inv, consts_inv)

        # Bind the forward jaxpr and the lazy inverse thunk.
        out_flat = custom_inverse_call_p.bind(
            *args_flat,
            forward_jaxpr=forward_jaxpr,
            inverse_jaxpr=lazy_inverse_jaxpr,  # delayed realization
            in_tree=in_tree,
            inv_argnum=dyn_args_index.index(self.inv_argnum),
        )

        return tree_unflatten(out_tree, out_flat)


def custom_inverse_call_impl(*args, forward_jaxpr, inverse_jaxpr, **params):
    # In a normal (non-differentiation) context we only need the forward jaxpr.
    ans = core.eval_jaxpr(forward_jaxpr.jaxpr, forward_jaxpr.literals, *args)
    return ans


def custom_inverse_call_abstract_eval(*args, forward_jaxpr, inverse_jaxpr, **params):
    return forward_jaxpr.out_avals


def custom_inverse_call_lowering(ctx, *args, forward_jaxpr, inverse_jaxpr, **params):
    # If lowering requires the inverse, ensure it is realized.
    if callable(inverse_jaxpr):
        inverse_jaxpr = inverse_jaxpr()
    return mlir.core_call_lowering(
        ctx, *args, name="forward_call", call_jaxpr=forward_jaxpr
    )


def process_jvp(forward_jaxpr, tangents):
    nonzeros = [type(t) is not ad_util.Zero for t in tangents]
    forward_jvp_jaxpr, _ = ad.jvp_jaxpr(forward_jaxpr, nonzeros, instantiate=False)
    nonzero_tangents = [t for t in tangents if type(t) is not ad_util.Zero]
    return forward_jvp_jaxpr, nonzero_tangents


def custom_inverse_jvp(primals, tangents, forward_jaxpr, inverse_jaxpr, **params):
    # Realize the inverse jaxpr if it is still a thunk.
    if callable(inverse_jaxpr):
        inverse_jaxpr = inverse_jaxpr()
    forward_jvp_jaxpr, nonzero_tangents = process_jvp(forward_jaxpr, tangents)
    new_primals, new_tangent = core.eval_jaxpr(
        forward_jvp_jaxpr.jaxpr, forward_jvp_jaxpr.consts, *primals, *nonzero_tangents
    )
    return [new_primals], [new_tangent]


def batch_custom_inverse_call(axis_data, args, in_dims, **params):
    forward_jaxpr = params.pop("forward_jaxpr")
    inverse_jaxpr = params.pop("inverse_jaxpr")
    # Batch the arguments.
    args = [
        batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0 else x
        for x, d in zip(args, in_dims)
    ]
    in_batched = [d is not batching.not_mapped for d in in_dims]
    batched_forward_fn, out_size1 = batching.batch_jaxpr(
        forward_jaxpr,
        axis_data,
        in_batched,
        False,
    )
    # Realize the inverse jaxpr if needed for batching.
    batched_inverse_fn, _ = batching.batch_jaxpr(
        inverse_jaxpr if not callable(inverse_jaxpr) else inverse_jaxpr(),
        axis_data,
        in_batched,
        False,
    )
    out = custom_inverse_call_p.bind(
        *args,
        forward_jaxpr=batched_forward_fn,
        inverse_jaxpr=batched_inverse_fn,
        **params,
    )
    out_dims = [0 if b else batching.not_mapped for b in out_size1]
    return out, out_dims


def custom_inverse_transpose(*args, **kwargs):
    return ad.call_transpose(custom_inverse_call_p, *args, **kwargs)


# Define the custom primitive.
custom_inverse_call_p = Primitive("custom_inverse_call_p")
custom_inverse_call_p.multiple_results = True
custom_inverse_call_p.def_impl(custom_inverse_call_impl)
custom_inverse_call_p.def_abstract_eval(custom_inverse_call_abstract_eval)
mlir.register_lowering(custom_inverse_call_p, custom_inverse_call_lowering)
batching.fancy_primitive_batchers[custom_inverse_call_p] = batch_custom_inverse_call
ad.primitive_transposes[custom_inverse_call_p] = custom_inverse_transpose
ad.primitive_jvps[custom_inverse_call_p] = custom_inverse_jvp
