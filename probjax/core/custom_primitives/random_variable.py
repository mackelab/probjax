from functools import lru_cache
from threading import local
from typing import Any, cast

from jax._src.core import shaped_abstractify
from jax._src.interpreters import ad as ad_src
from jax.extend.core import ClosedJaxpr, Primitive
from jax.interpreters import ad, batching, mlir
from jax.tree_util import tree_flatten, tree_unflatten

from probjax.core.custom_primitives.call_primitive import (
    call_abstract_eval,
    call_impl,
    call_lowering,
    jvp_from_forward_jaxpr,
)
from probjax.core.custom_primitives.common import (
    batch_closed_jaxpr,
    ensure_hashable,
    has_tracer,
    move_mapped_axes_to_front,
    trace_to_closed_jaxpr,
)
from probjax.core.custom_primitives.contracts import parse_random_variable_call_params


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

    fn_name = getattr(rvs_fn, "__name__", str(rvs_fn))
    forward_jaxpr, _, _ = trace_to_closed_jaxpr(
        f_dyn,
        in_tree=in_tree,
        in_avals=in_avals,
        debug_name="random_variable forward",
        const_context=f"random_variable forward ({fn_name})",
    )
    return forward_jaxpr


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
    _ = parse_random_variable_call_params(
        {
            "forward_jaxpr": forward_jaxpr,
            "in_tree": in_tree,
            "shape": shape,
            "dist": dist,
            "name": name,
            "rvs_fn": rvs_fn,
            "logpdf_fn": logpdf_fn,
            "kwds_items": kwds_items,
        }
    )
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
    _ = parse_random_variable_call_params(
        {
            "forward_jaxpr": forward_jaxpr,
            "in_tree": in_tree,
            "shape": shape,
            "dist": dist,
            "name": name,
            "rvs_fn": rvs_fn,
            "logpdf_fn": logpdf_fn,
            "kwds_items": kwds_items,
        }
    )
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


def _rv_batching_rule(axis_data, batched_args, batch_dims, **params):
    params = dict(params)
    forward_jaxpr = params["forward_jaxpr"]

    new_args, in_axes, any_batched = move_mapped_axes_to_front(batched_args, batch_dims)
    if not any_batched:
        out = Primitive.bind(rv_p, *new_args, **params)
        return out, batching.not_mapped

    batched_forward_jaxpr, out_axes = batch_closed_jaxpr(
        forward_jaxpr, axis_data, in_axes
    )

    rebound_params = dict(params, forward_jaxpr=batched_forward_jaxpr)
    out = Primitive.bind(rv_p, *new_args, **rebound_params)
    out_axis = out_axes[0] if out_axes else batching.not_mapped
    return out, out_axis


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
        forward_jaxpr=None,
        in_tree=None,
        kwds_items=None,
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
            kwds = {} if kwds_items is None else dict(kwds_items)
        if name is None:
            prefix = getattr(dist_obj, "name", "rv")
            name = name_stack.get_name(prefix)

        args, parsed_kwds = dist_obj._parse_args(*args, **kwds)
        if kwds_items is None:
            kwds_items = tuple(
                sorted(
                    (k, ensure_hashable(v, f"kwds['{k}']"))
                    for k, v in parsed_kwds.items()
                )
            )

        if not has_tracer((key, args, parsed_kwds)):
            return rvs_fn(key, *args, shape=shape, **parsed_kwds)

        dyn_inputs = (key, *args)
        flat_inputs, dyn_tree = tree_flatten(dyn_inputs)
        if in_tree is None:
            in_tree = dyn_tree

        if forward_jaxpr is None:
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
batching.fancy_primitive_batchers[rv_p] = _rv_batching_rule
mlir.register_lowering(rv_p, _rv_lowering)
ad.primitive_jvps[rv_p] = _rv_jvp
ad.primitive_transposes[rv_p] = _rv_transpose_rule


__all__ = [
    "NameStack",
    "call_rv_p",
    "name_stack",
    "rv_p",
]
