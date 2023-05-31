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


from probjax.core.utils import _sampling_logprobs_jaxprs_with_common_consts

from probjax.distributions.distribution import Distribution

__all__ = ["rv"]


def _sample_distribution(dist: Distribution, key, *args, shape=(), **kwargs):
    return dist.sample(key, *args, sample_shape=shape, **kwargs)


def _log_prob_distribution(dist: Distribution, value, *args, **kwargs):
    return dist.log_prob(value=value, *args, **kwargs)



def rv(dist: Distribution, name: Hashable) -> Callable:
    """This takes a distribution and returns a function that samples from that distribution.


    Args:
        dist (Distribution): Distribution of random variable
        name (Hashable): Name of random variable

    Returns:
        Callable: Sampling function
    """

    def sample_fn(key, *args, **kwargs):
        return _sample_distribution(dist, key, *args, **kwargs)

    def log_prob_fn(value, *args, **kwargs):
        return _log_prob_distribution(dist, value, *args, **kwargs)

    def wrapped(*args, **kwargs):

        [sampling_fn_jaxpr, log_prob_fn_jaxpr], consts, out_trees = _sampling_logprobs_jaxprs_with_common_consts(sample_fn, log_prob_fn)

        out = rv_p.bind(
            *consts,
            *args,
            name=name,
            sampling_fn_jaxpr=sampling_fn_jaxpr,
            log_prob_fn_jaxpr=log_prob_fn_jaxpr,
            dist = type(dist),
            **kwargs
        )

        return tree_util.tree_unflatten(out_trees[0], out)

    return wrapped


def _rv_impl(*args, **params):

    with new_sublevel():
        call_jaxpr = params.get("sampling_fn_jaxpr")
        return eval_jaxpr(call_jaxpr.jaxpr, call_jaxpr.literals, *args)


def _rv_abstract_eval(*args, **params):
    with new_sublevel():
        call_jaxpr = params.get("sampling_fn_jaxpr")
        return call_jaxpr.out_avals


# JIT support
def _rv_lowering(
    ctx, *args, name, sampling_fn_jaxpr, log_prob_fn_jaxpr, **params
):
    call_jaxpr = sampling_fn_jaxpr
    return mlir.core_call_lowering(
        ctx, *args, name=name, call_jaxpr=call_jaxpr
    )


def _rv_transpose_rule(*args, **kwargs):
    return ad.call_transpose(rv_p, *args, **kwargs)


def _rv_batching_rule(spmd_axis_name, axis_size, axis_name, main_type, args, dims, **params):
    sampling_fn_jaxpr = params.pop("sampling_fn_jaxpr")
    log_prob_fn_jaxpr = params.pop("log_prob_fn_jaxpr")

    
    # We have to batch the jaxprs. For that lets first get the invals and outvals
    in_avals1 = sampling_fn_jaxpr.in_avals
    out_avals1 = sampling_fn_jaxpr.out_avals

    in_avals2 = log_prob_fn_jaxpr.in_avals
    out_avals2 = log_prob_fn_jaxpr.out_avals

    # We will batch all the inputs and outputs  (maybe do not batch consts ... )
    in_batched1  = [True] * len(in_avals1)
    out_batched1 = [True] * len(out_avals1)

    in_batched2  = [True] * len(in_avals2)
    out_batched2 = [True] * len(out_avals2)


    # Applies the batching for the jaxprs
    args = [
        batching.bdim_at_front(x, d, axis_size) for x, d in zip(args, dims)]
    

    # Batched jaxprs
    batched_sampling_fn, out_size1 = batch_jaxpr(sampling_fn_jaxpr, axis_size,in_batched1, out_batched1, axis_name, spmd_axis_name, main_type)
    batched_log_prob_fn, _ = batch_jaxpr(log_prob_fn_jaxpr,  axis_size, in_batched2, out_batched2, axis_name, spmd_axis_name, main_type)

    # Update jaxprs with batched ones
    out = rv_p.bind(
            *args,
            sampling_fn_jaxpr=batched_sampling_fn,
            log_prob_fn_jaxpr=batched_log_prob_fn,
            **params
        )
    
    # Outdim 
    out_dims = [0 if b else batching.not_mapped for b in out_size1]

    return out, out_dims


rv_p = Primitive("random_variable")
rv_p.multiple_results = True
rv_p.def_impl(_rv_impl)
rv_p.def_abstract_eval(_rv_abstract_eval)
batching.spmd_axis_primitive_batchers[rv_p] = _rv_batching_rule
batching.axis_primitive_batchers[rv_p] = partial(_rv_batching_rule, None)
mlir.register_lowering(rv_p, _rv_lowering)
ad.primitive_transposes[rv_p] = _rv_transpose_rule
