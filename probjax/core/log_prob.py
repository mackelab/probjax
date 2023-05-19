from jax.core import CallPrimitive, Primitive
import jax

from jax import tree_util
from jax import linear_util as lu
from jax import api_util
from typing import Hashable, Callable
from jax._src import effects
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir

from probjax.distributions.distribution import Distribution

__all__ = ["log_prob"]

log_prob_p = CallPrimitive("log_prob")


def _log_prob_distribution(dist: Distribution, value, *args, **kwargs):
    return dist.log_prob(value=value, *args, **kwargs)


# Default implementation
@log_prob_p.def_impl
def _log_prob_p_impl(f: Callable, *args, **_) -> Callable:
    with jax.core.new_sublevel():
        return f.call_wrapped(*args)


# # JIT support
# def _rv_lowering(ctx, *args, name, call_jaxpr, scope=None, **_):
#     return mlir.core_call_lowering(ctx, *args, name=name, call_jaxpr=call_jaxpr)


# mlir.register_lowering(rv_p, _rv_lowering)


# def _rv_transpose_rule(*args, **kwargs):
#     return ad.call_transpose(rv_p, *args, **kwargs)


# ad.primitive_transposes[rv_p] = _rv_transpose_rule


def log_prob(dist: Distribution) -> Callable:
    """This takes a distribution and returns a function that samples from that distribution.


    Args:
        dist (Distribution): Distribution of random variable
        name (Hashable): Name of random variable

    Returns:
        Callable: Sampling function
    """

    def wrapped(*args, **kwargs):
        def log_prob(value, *args, **kwargs):
            return _log_prob_distribution(dist, value, *args, **kwargs)
        fun = lu.wrap_init(log_prob, kwargs)
        flat_args, in_tree = tree_util.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        out_flat = log_prob_p.bind(flat_fun, *flat_args,dist=type(dist))
        return tree_util.tree_unflatten(out_tree(), out_flat)

    return wrapped
