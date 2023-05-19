from jax.core import (
    CallPrimitive,
    ClosedJaxpr,
    Primitive,
    new_sublevel,
    get_aval,
    raise_to_shaped,
    eval_jaxpr,
)
import jax
import jax.random as jrandom
from jax import tree_util
from jax import linear_util as lu
from jax import api_util
from typing import Hashable, Callable
from jax._src import effects
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters.batching import batch_jaxpr

from functools import partial


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

from probjax.distributions.distribution import Distribution

__all__ = ["rv"]


def _sample_distribution(dist: Distribution, key, *args, shape=(), **kwargs):
    return dist.sample(key, *args, sample_shape=shape, **kwargs)


def _log_prob_distribution(dist: Distribution, value, *args, **kwargs):
    return dist.log_prob(value=value, *args, **kwargs)


def unzip2(*args, **params):
    num_sample_consts = params["num_sampling_consts"]
    num_log_prob_consts = params["num_log_prob_consts"]
    mode = params["mode"]
    if mode == "sampling":
        consts = args[:num_sample_consts]
    else:
        consts = args[num_sample_consts : num_sample_consts + num_log_prob_consts]

    args = args[num_sample_consts + num_log_prob_consts :]
    return consts, args


def rv(dist: Distribution, name: Hashable) -> Callable:
    """This takes a distribution and returns a function that samples from that distribution.


    Args:
        dist (Distribution): Distribution of random variable
        name (Hashable): Name of random variable

    Returns:
        Callable: Sampling function
    """

    def sample_fn(key, *args, shape=(), **kwargs):
        return _sample_distribution(dist, key, *args, shape=shape, **kwargs)

    def log_prob_fn(value, *args, **kwargs):
        return _log_prob_distribution(dist, value, *args, **kwargs)

    def wrapped(*args, **kwargs):
        operands = [
            jrandom.PRNGKey(0),
        ]
        sampling_ops, sampling_ops_tree = tree_util.tree_flatten(operands)
        sampling_ops_avals = tuple(map(_abstractify, sampling_ops))
        sampling_jaxpr, sampling_consts, sampling_out_trees = _initial_style_open_jaxpr(
            sample_fn, sampling_ops_tree, sampling_ops_avals
        )
        sampling_fn_jaxpr = ClosedJaxpr(pe.convert_constvars_jaxpr(sampling_jaxpr), ())
        sampling_out_avals = sampling_fn_jaxpr.out_avals

        log_prob_operands = sampling_out_avals
        log_prob_ops, log_prob_ops_tree = tree_util.tree_flatten(log_prob_operands)
        log_prob_ops_avals = tuple(log_prob_ops)  # Is already abstract
        log_prob_jaxpr, log_prob_consts, log_prob_out_trees = _initial_style_open_jaxpr(
            log_prob_fn, log_prob_ops_tree, log_prob_ops_avals
        )
        log_prob_fn_jaxpr = ClosedJaxpr(pe.convert_constvars_jaxpr(log_prob_jaxpr), ())
        log_prob_out_avals = log_prob_fn_jaxpr.out_avals

        num_sampling_consts = len(sampling_consts)
        num_log_prob_consts = len(log_prob_consts)

        consts = sampling_consts + log_prob_consts

        out = rv_p.bind(
            *consts,
            *args,
            name=name,
            sampling_fn_jaxpr=sampling_fn_jaxpr,
            num_sampling_consts=num_sampling_consts,
            log_prob_fn_jaxpr=log_prob_fn_jaxpr,
            num_log_prob_consts=num_log_prob_consts,
            mode="sampling",
            **kwargs
        )

        return tree_util.tree_unflatten(sampling_out_trees, out)

    return wrapped


def _rv_impl(*args, **params):
    sampling_fn_jaxpr = params.get("sampling_fn_jaxpr")
    log_prob_fn_jaxpr = params.get("log_prob_fn_jaxpr")
    mode = params.get("mode")
    consts, args = unzip2(*args, **params)
    if mode == "sampling":
        with new_sublevel():
            out = eval_jaxpr(
                sampling_fn_jaxpr.jaxpr, sampling_fn_jaxpr.literals, *consts, *args
            )
    elif mode == "log_prob":
        with new_sublevel():
            out = eval_jaxpr(
                log_prob_fn_jaxpr.jaxpr, log_prob_fn_jaxpr.literals, *consts, *args
            )
    else:
        raise NotImplementedError

    return out


def _rv_abstract_eval(*args, **params):
    mode = params.get("mode")
    if mode == "sampling":
        out_types = params.pop("sampling_fn_jaxpr").out_avals
    elif mode == "log_prob":
        out_types = params.pop("log_prob_fn_jaxpr").out_avals
    else:
        raise NotImplementedError

    return out_types


# JIT support
def _rv_lowering(
    ctx, *args, name, sampling_fn_jaxpr, log_prob_fn_jaxpr, **params
):
    mode = params.get("mode")
    consts, args = unzip2(*args, **params)
    if mode == "sampling":
        call_jaxpr = sampling_fn_jaxpr
    else:
        call_jaxpr = log_prob_fn_jaxpr
    return mlir.core_call_lowering(
        ctx, *consts, *args, name=name, call_jaxpr=call_jaxpr
    )


def _rv_transpose_rule(*args, **kwargs):
    return ad.call_transpose(rv_p, *args, **kwargs)


# def _rv_batching_rule(
#     spmd_axis_name, axis_size, axis_name, main_type, args, dims, **params
# ):
#     sampling_fn_jaxpr = params.pop("sampling_fn_jaxpr")
#     log_prob_fn_jaxpr = params.pop("log_prob_fn_jaxpr")

#     print(args,dims)
#     # consts, args = unzip2(*args, **params)
#     # consts_dim, args_dim = dims[: len(consts)], dims[len(consts) :]
#     # dims = consts_dim + args_dim
#     # args = consts + args

#     in_avals1 = sampling_fn_jaxpr.in_avals
#     out_avals1 = sampling_fn_jaxpr.out_avals

#     in_avals2 = log_prob_fn_jaxpr.in_avals
#     out_avals2 = log_prob_fn_jaxpr.out_avals

#     in_batched1 = [True] * len(in_avals1)
#     out_batched1 = [True] * len(out_avals1)

#     in_batched2 = [True] * len(in_avals2)
#     out_batched2 = [True] * len(out_avals2)

#     args = [batching.bdim_at_front(x, d, axis_size) for x, d in zip(args, dims)]

#     batched_sampling_fn, out_size1 = batch_jaxpr(
#         sampling_fn_jaxpr,
#         axis_size,
#         in_batched1,
#         out_batched1,
#         axis_name,
#         spmd_axis_name,
#         main_type,
#     )

#     print(args)
#     print(batched_sampling_fn)

#     batched_log_prob_fn, out_size2 = batch_jaxpr(
#         log_prob_fn_jaxpr,
#         axis_size,
#         in_batched1,
#         out_batched1,
#         axis_name,
#         spmd_axis_name,
#         main_type,
#     )

#     out = eval_jaxpr(
#         batched_sampling_fn.jaxpr, batched_sampling_fn.literals, *args
#     )

#     return out, [0,]

def _rv_batching_rule(spmd_axis_name, axis_size, axis_name, main_type, args, dims, **params):
    sampling_fn_jaxpr = params.pop("sampling_fn_jaxpr")
    log_prob_fn_jaxpr = params.pop("log_prob_fn_jaxpr")

    

    in_avals1 = sampling_fn_jaxpr.in_avals
    out_avals1 = sampling_fn_jaxpr.out_avals

    in_avals2 = log_prob_fn_jaxpr.in_avals
    out_avals2 = log_prob_fn_jaxpr.out_avals

    in_batched1  = [True] * len(in_avals1)
    out_batched1 = [True] * len(out_avals1)

    in_batched2  = [True] * len(in_avals2)
    out_batched2 = [True] * len(out_avals2)

    args = [
        batching.bdim_at_front(x, d, axis_size) for x, d in zip(args, dims)]
    
    batched_sampling_fn, out_size1 = batch_jaxpr(sampling_fn_jaxpr, axis_size,in_batched1, out_batched1, axis_name, spmd_axis_name, main_type)

    batched_log_prob_fn, out_size2 = batch_jaxpr(log_prob_fn_jaxpr,  axis_size, in_batched2, out_batched2, axis_name, spmd_axis_name, main_type)

    const1 = args[:params.get("num_sampling_consts")]
    const2 = args[params.get("num_sampling_consts"):params.get("num_sampling_consts")+params.get("num_log_prob_consts")]
    args = args[params.get("num_sampling_consts")+params.get("num_log_prob_consts"):]

    out = eval_jaxpr(batched_sampling_fn.jaxpr, batched_sampling_fn.literals, *const1,*args)

    return out, [0,]


rv_p = Primitive("random_variable")
rv_p.multiple_results = True
rv_p.def_impl(_rv_impl)
rv_p.def_abstract_eval(_rv_abstract_eval)
batching.spmd_axis_primitive_batchers[rv_p] = _rv_batching_rule
batching.axis_primitive_batchers[rv_p] = partial(_rv_batching_rule, None)
mlir.register_lowering(rv_p, _rv_lowering)
ad.primitive_transposes[rv_p] = _rv_transpose_rule
