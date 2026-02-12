from functools import lru_cache
from threading import local
from typing import Any, cast

import jax
from jax._src import linear_util as lu
from jax._src.api_util import debug_info, flatten_fun_nokwargs
from jax._src.core import shaped_abstractify
from jax._src.interpreters import ad as ad_src
from jax._src.interpreters import batching as batching_src
from jax._src.interpreters import partial_eval as pe_src
from jax.extend.core import ClosedJaxpr, Primitive
from jax.interpreters import ad, batching, mlir
from jax.tree_util import tree_flatten, tree_unflatten

from probjax.core.custom_primitives.call_primitive import (
    call_abstract_eval,
    call_impl,
    call_lowering,
    jvp_from_forward_jaxpr,
)
from probjax.core.custom_primitives.common import ensure_hashable, has_tracer


class NameStack(local):
    def __init__(self):
        self.counts = {}

    def get_name(self, prefix="rv"):
        if prefix not in self.counts:
            self.counts[prefix] = 0
        count = self.counts[prefix]
        name = f"{prefix}_{count}"
        self.counts[prefix] += 1
        return name


name_stack = NameStack()


@lru_cache(maxsize=4096)
def _trace_random_variable_jaxpr(
    rvs_fn,
    in_tree,
    in_avals,
    shape,
    kwds_items,
):
    kwds = dict(kwds_items)

    def f_dyn(*flat_inputs):
        inputs = tree_unflatten(in_tree, flat_inputs)
        key, *args = inputs
        return rvs_fn(key, *args, shape=shape, **kwds)

    info = debug_info("random_variable forward", rvs_fn, (), {})
    f_wrapped = lu.wrap_init(f_dyn, debug_info=info)
    f_flat, out_tree_thunk = flatten_fun_nokwargs(f_wrapped, in_tree)
    jaxpr, _, consts = pe_src.trace_to_jaxpr_dynamic(f_flat, in_avals)
    _ = out_tree_thunk()
    return ClosedJaxpr(jaxpr, consts)


def _rv_impl(
    *flat_inputs,
    forward_jaxpr,
    in_tree,
    shape,
    dist,
    name,
    rvs_fn,
    logpdf_fn,
    kwds_items,
):
    return call_impl(
        *flat_inputs,
        forward_jaxpr=forward_jaxpr,
        single_result=True,
        in_tree=in_tree,
        shape=shape,
        dist=dist,
        name=name,
        rvs_fn=rvs_fn,
        logpdf_fn=logpdf_fn,
        kwds_items=kwds_items,
    )


def _rv_abstract_eval(
    *flat_avals,
    forward_jaxpr,
    in_tree,
    shape,
    dist,
    name,
    rvs_fn,
    logpdf_fn,
    kwds_items,
):
    return call_abstract_eval(
        *flat_avals,
        forward_jaxpr=forward_jaxpr,
        single_result=True,
        in_tree=in_tree,
        shape=shape,
        dist=dist,
        name=name,
        rvs_fn=rvs_fn,
        logpdf_fn=logpdf_fn,
        kwds_items=kwds_items,
    )


def _rv_lowering(ctx, *mlir_args, **params):
    params = dict(params)
    forward_jaxpr = params.pop("forward_jaxpr")
    lowering_name = params.pop("name", "random_variable")
    return call_lowering(
        ctx,
        *mlir_args,
        forward_jaxpr=forward_jaxpr,
        lowering_name=lowering_name,
        **params,
    )


def _rv_batching_rule(batched_args, batch_dims, **params):
    params = dict(params)
    rvs_fn = params["rvs_fn"]
    logpdf_fn = params["logpdf_fn"]
    shape = params.get("shape", ())
    kwds = dict(params.get("kwds_items", ()))

    def rvs_fn_with_shape(key, *args, shape=shape, **kwargs):
        del kwargs
        return rvs_fn(key, *args, shape=shape, **kwds)

    rvs_fn_batched = jax.vmap(rvs_fn_with_shape, in_axes=batch_dims)
    if any(d is not batching.not_mapped for d in batch_dims[1:]):
        logpdf_fn_batched = jax.vmap(logpdf_fn, in_axes=batch_dims[1:])
    else:
        logpdf_fn_batched = logpdf_fn

    key, *args = batched_args
    out = rv_p.bind(
        key,
        *args,
        dist=params.get("dist"),
        shape=shape,
        name=params.get("name"),
        rvs_fn=rvs_fn_batched,
        logpdf_fn=logpdf_fn_batched,
        kwds=kwds,
    )
    return out, 0


def _rv_jvp(primals, tangents, **params):
    primal_outs, tangent_outs = jvp_from_forward_jaxpr(
        params["forward_jaxpr"],
        primals,
        tangents,
    )
    return primal_outs[0], tangent_outs[0]


def _rv_transpose_rule(*args, **kwargs):
    call_transpose = getattr(ad_src, "call_transpose")
    return call_transpose(rv_p, *args, **kwargs)


class RandomVariableCallPrimitive(Primitive):
    def __init__(self):
        super().__init__("random_variable")

    def bind(
        self,
        key,
        *args,
        shape=(),
        dist=None,
        name=None,
        rvs_fn=None,
        logpdf_fn=None,
        kwds=None,
        **params,
    ):
        if dist is None and (rvs_fn is None or logpdf_fn is None):
            raise ValueError("dist must be provided when rvs_fn/logpdf_fn are not set")

        dist_obj = cast(Any, dist)
        if rvs_fn is None:
            rvs_fn = dist_obj.rvs
        if logpdf_fn is None:
            logpdf_fn = dist_obj.logpdf

        if kwds is None:
            kwds = {}
        if name is None:
            prefix = getattr(dist_obj, "name", "rv")
            name = name_stack.get_name(prefix)

        args, kwds = dist_obj._parse_args(*args, **kwds)
        kwds_items = tuple(
            sorted((k, ensure_hashable(v, f"kwds['{k}']")) for k, v in kwds.items())
        )

        if not has_tracer((key, args, kwds)):
            return rvs_fn(key, *args, shape=shape, **kwds)

        dyn_inputs = (key, *args)
        flat_inputs, in_tree = tree_flatten(dyn_inputs)
        in_avals = tuple(map(shaped_abstractify, flat_inputs))
        forward_jaxpr = _trace_random_variable_jaxpr(
            rvs_fn,
            in_tree,
            in_avals,
            shape,
            kwds_items,
        )

        return super().bind(
            *flat_inputs,
            forward_jaxpr=forward_jaxpr,
            in_tree=in_tree,
            shape=shape,
            dist=dist_obj,
            name=name,
            rvs_fn=rvs_fn,
            logpdf_fn=logpdf_fn,
            kwds_items=kwds_items,
            **params,
        )


rv_p = RandomVariableCallPrimitive()
call_rv_p = rv_p

rv_p.def_impl(_rv_impl)
rv_p.def_abstract_eval(_rv_abstract_eval)
batching.primitive_batchers[rv_p] = _rv_batching_rule
mlir.register_lowering(rv_p, _rv_lowering)
ad.primitive_jvps[rv_p] = _rv_jvp
ad.primitive_transposes[rv_p] = _rv_transpose_rule


__all__ = [
    "NameStack",
    "call_rv_p",
    "name_stack",
    "rv_p",
]
