from jax.core import CallPrimitive, Primitive
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

from probjax.distributions.distribution import Distribution

__all__ = ["rv"]

rv_p = CallPrimitive("random_variable")


def _sample_distribution(dist: Distribution, key, shape=()):
    return dist.sample(key, sample_shape=shape)


def _log_prob_distribution(dist: Distribution, value, *args, **kwargs):
    return dist.log_prob(value=value, *args, **kwargs)


# Default implementation
@rv_p.def_impl
def _rv_p_impl(f: Callable, *args, **params) -> Callable:
    with jax.core.new_sublevel():
        return f.call_wrapped(*args)


# JIT support
def _rv_lowering(ctx, *args, name, call_jaxpr, scope=None, **_):
    return mlir.core_call_lowering(ctx, *args, name=name, call_jaxpr=call_jaxpr)


mlir.register_lowering(rv_p, _rv_lowering)


def _rv_transpose_rule(*args, **kwargs):
    return ad.call_transpose(rv_p, *args, **kwargs)


ad.primitive_transposes[rv_p] = _rv_transpose_rule

from copy import deepcopy


def rv(dist: Distribution, name: Hashable) -> Callable:
    """This takes a distribution and returns a function that samples from that distribution.


    Args:
        dist (Distribution): Distribution of random variable
        name (Hashable): Name of random variable

    Returns:
        Callable: Sampling function
    """
    
    def wrapped(*args, **kwargs):
        def sample(key, shape=()):
            return _sample_distribution(dist, key, shape)
        
        dummy_key = jrandom.PRNGKey(0)
        dummy_sample = dist.sample(dummy_key)

        def log_prob(value, *args, **kwargs):
            return _log_prob_distribution(dist, value, *args, **kwargs)

        fun2 = lu.wrap_init(log_prob)
        flat_args2, in_tree2 = tree_util.tree_flatten(dummy_sample)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun2, in_tree2)
        log_prob_jaxpr = jax.make_jaxpr(flat_fun.call_wrapped)(flat_args2)


        fun = lu.wrap_init(sample, kwargs)
        flat_args, in_tree = tree_util.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        out_flat = rv_p.bind(
            flat_fun,
            *flat_args,
            name=name,
            dist=type(dist),
            log_prob_jaxpr=log_prob_jaxpr,
            mode = "sampling",
        )
        return tree_util.tree_unflatten(out_tree(), out_flat)

    return wrapped
