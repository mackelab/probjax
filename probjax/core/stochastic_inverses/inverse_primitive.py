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
from jax._src import api_util


from probjax.core.utils import BaseRules


RULES = {}


def register_inverse_rule(
    prim: Primitive,
):
    """Registers a rule for a primitive.

    Args:
        prim (Primitive): Primitive
        rule (Callable): Rule to apply
    """

    def make_rule(fun):

        def inverse_rule(known_invals, known_outvals):
            if all(known_invals):


        RULES[prim] = fun

    return make_rule


def stochastic_inverse(
    prim, known_invals, known_outvals, constraints_invals, constraints_outvals
):
    """Computes the inverse of a primitive. If the primitive is not invertible, it stochasticly samples from the inverse distribution."""

    inverse_fn = RULES[prim](
        known_invals, known_outvals, constraints_invals, constraints_outvals
    )

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(inverse_fn, **kwargs)
        flat_args, in_tree = tree_util.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        out_flat = inverse_p.bind(flat_fun, *flat_args)
        return tree_util.tree_unflatten(out_tree, out_flat)

    return wrapped


def _inverse_impl(*args, **kwargs):
    pass


inverse_p = Primitive("inverse")
inverse_p.multiple_results = True
inverse_p.def_impl(_inverse_impl)


# Inverse rules base class
